"""
oracleGeneration.py

The actual GPT Image 2 call that was the missing piece all last session --
every test up to now used Oracle images supplied by hand. This is what
produces them live, in the pipeline, from a given Original photo.

ENDPOINT CHOICE: OpenAI's Image API has two endpoints -- images.generate
(text-to-image, nothing to correct FROM) and images.edit (modify an
existing image per a prompt). This pipeline is correcting an existing
photo, not generating a new scene, so images.edit is the right one --
POST https://api.openai.com/v1/images/edits, model "gpt-image-2".

STDLIB ONLY, NO SDK -- same discipline as level0_scene_classifier.py's
own documented reasoning (an SDK dependency that isn't in requirements.txt
degrades silently to a missing-module failure in production, confirmed
as a real incident there). images.edit needs multipart/form-data (it
takes a real image file, not JSON), which urllib doesn't build for you --
_encode_multipart below does it by hand rather than reaching for
`requests` or the `openai` package.

PROMPT SELECTION: takes is_exterior directly rather than re-deriving
scene type -- level0_scene_classifier.py's resolve_scene_type() is
already the authoritative signal for this in the pipeline; this module
should be a consumer of that decision, not a second source of truth for
it. Callers must pass the SAME is_exterior this pipeline run already
computed.

PROMPT VERSIONING: exterior uses v2 (Smart_Correct_Exterior_Oracle_v2.docx)
-- NOT v1. v1 was tested this session and found to overshoot on sky and
vegetation saturation (measured on a real photo: sky +122%, vegetation
+45%, both violating v1's own stated rules). v2 explicitly tightens Sky,
Lawn, and Trees-and-Shrubs by name and adds a general Critical Rule
("if a correction would make the property appear in better physical
condition than the captured photo supports, do not perform it"). Keep
using v2 going forward; only revisit if a real batch shows it still
overshoots (see handoff: re-check oracleCorrection.py's saturation-cap
trigger rate once this is live -- if v2 is holding, the caps should fire
much less often than they did against v1's output).

INTERIOR now ALSO uses v2 (Smart_Correct_Oracle_Interior_v2.docx) -- NOT
v1, same reasoning as exterior but the OPPOSITE failure direction. v1's
language was restrained-by-default ("carefully balanced exposure,"
"subtle local contrast," "do not... flatten contrast") with no explicit
permission to recover aggressively -- confirmed on a real photo (Flux
Kontext, IMG_8310): the interior result was a near-total no-op across
the ENTIRE room (wall/fireplace/carpet all within ~1-3 L points of the
raw Original), not selective protection of one dark object. Compare to
the same model's exterior test on the same session, which DID apply a
real (if imperfect) correction -- same model, same prompt-writing
discipline, different outcome, traced to the exterior prompt having an
explicit "Dynamic Range: recover shadow/highlight/midtone" instruction
that v1 interior never had. v2 interior adds that instruction directly,
plus a dedicated "Furniture and Materials in Shadow -- Recover, Do Not
Protect" section naming the exact failure this codebase already
documented once, in level2_vision_regions.py's own prompt, and had not
carried over here: "a dark-toned object sitting in an underexposed room
is, in the overwhelming majority of cases, underexposed furniture...
not a genuinely black material that must be protected." Keep using v2
going forward; only revisit if a real batch still under-corrects.

TIMEOUT AND RETRY: image generation can take up to ~2 minutes for a
complex prompt (this is OpenAI's own stated guidance, not a guess) --
sized very differently from this codebase's Vision calls, which are
never given more than 30s. Retries use exponential backoff WITH jitter,
also per OpenAI's own guidance -- jitter specifically matters here
because a fleet of Railway workers retrying in lockstep after a shared
rate-limit response would all land on the rate limiter at the same
instant otherwise.

ORG VERIFICATION GOTCHA: OpenAI gates the GPT Image model family behind
API Organization Verification in the developer console. If every call
fails identically on the very first attempt, check that BEFORE assuming
the code is wrong -- it will look like an auth failure, not a "not
verified yet" failure, from this module's vantage point.

LABELING: every image this module produces is, by definition, the
Oracle Scene Render -- a digitally altered image. That's not a caveat
for this module to enforce (see oracleCorrection.py's own note on this),
it's a fact about anything this function returns, and it must be labeled
wherever shown, per the settled position this project already reached.
"""

import os
import io
import json
import time
import random
import base64
import logging
import mimetypes
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

ORACLE_GENERATION_ENABLED = os.environ.get("ORACLE_GENERATION_ENABLED", "true").lower() not in ("false", "0", "")
# NOTE: confirm this is the exact env var name used when the OpenAI key was
# added to Railway -- "OPENAI_API_KEY" is the OpenAI SDK's own conventional
# name and the most likely match, but this wasn't confirmed against the
# actual Railway config this session. If generate_oracle_scene() reports
# missing_OPENAI_API_KEY despite the key being added, check the Railway
# variable name matches this exactly before assuming anything else is wrong.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ORACLE_MODEL = os.environ.get("ORACLE_GENERATION_MODEL", "gpt-image-2")
TIMEOUT_SECONDS = int(os.environ.get("ORACLE_GENERATION_TIMEOUT_SECONDS", "150"))  # generation can
# take up to ~2 min per OpenAI's own guidance -- do not shrink this to match
# the Vision calls' timeouts elsewhere in this codebase, it's a different
# kind of call with a materially different latency profile.
MAX_RETRIES = int(os.environ.get("ORACLE_GENERATION_MAX_RETRIES", "3"))
BASE_BACKOFF_SECONDS = 2.0
IMAGE_SIZE = os.environ.get("ORACLE_GENERATION_SIZE", "1536x1024")  # closest supported size to
# this pipeline's typical 4:3/3:2 listing-photo aspect ratios; revisit if a
# real batch shows OpenAI's supported size list doesn't fit well.
IMAGE_QUALITY = os.environ.get("ORACLE_GENERATION_QUALITY", "high")


# ---- Prompt text, embedded directly (not read from the .docx at runtime) ----
# Sourced verbatim from Smart_Correct_Oracle_v1-Prompt.docx (interior) and
# Smart_Correct_Exterior_Oracle_v2.docx (exterior, v2 -- see module
# docstring for why v2 and not v1). Embedded as constants so this module
# has no runtime dependency on the docx files or a document-reading step --
# same self-containment reasoning as every other prompt-driven module in
# this codebase keeping its prompt as a Python string, not an external file.
# IF EITHER PROMPT DOCX IS REVISED AGAIN, THIS MUST BE UPDATED TO MATCH --
# there is no automated sync between the docx and this constant.

INTERIOR_ORACLE_PROMPT = """Smart Correct Oracle v3

PRIMARY ROLE
Treat the uploaded image as a real photograph of an existing room, captured under real, unedited lighting conditions -- often underexposed, mixed-light, or high dynamic range.
Produce the same photograph corrected to MLS BRIGHT -- the bright, clean, evenly-lit, high-key style standard for professional real estate listing photography. This is a specific, recognizable industry look, not a vague "improvement": walls and ceilings read bright and clean, shadows are opened up so no area of the room reads dark or hidden, whites are crisp, and the room overall reads as inviting, spacious, and fully lit -- the way a top-tier real estate photography service (the kind MLS listings are known for) delivers, not a restrained or partial correction.
This is a photographic correction task, not a redesign, staging, or relighting task -- but correction means MLS Bright, fully realized, not a conservative or subtle version of it.

Identity Preservation
Preserve exactly:
* Architecture
* Camera position
* Perspective
* Composition
* Lens characteristics
* Furniture
* Decor
* Materials
* Finishes
* Paint colors
* Wood colors
* Fabric colors
* Exterior view
* Landscaping
* Fixtures
* Window coverings
* Artwork
Do not move, add, remove, replace, redesign, or restyle anything.

Dynamic Range -- MLS Bright, Fully Realized
Recover shadow detail, highlight detail, and midtone separation throughout the entire frame to reach true MLS Bright -- not a partial or conservative lift. A real estate photographer shooting this room professionally would use HDR bracketing, fill flash, or strobe specifically to eliminate deep shadow and blown highlight in a single frame, producing the bright, open, evenly-lit look every MLS listing photo is expected to have. Match that result directly: a fully, evenly, BRIGHTLY resolved room -- not a moderately-improved version of the underexposed capture, and not a cautious middle ground between the raw file and MLS Bright.
Do not:
* change the time of day
* introduce new sunlight
* add a light source that was not present
* invent shadows or reflections that the captured data does not support
Do:
* fully resolve underexposed areas into their real, visible material and color, using the sensor data that is genuinely present, however faint
* push exposure and shadow recovery as far as true MLS Bright requires -- treat "the data is barely visible in the raw file" as a reason to recover it aggressively and brightly, not a reason to leave it dim or partially corrected

Furniture and Materials in Shadow -- Recover, Do Not Protect
This is the single most common failure in this task, so it is stated directly: a dark-toned object sitting in an underexposed room is, in the overwhelming majority of cases, underexposed furniture that needs the same MLS Bright correction as the rest of the room -- not a genuinely black material that must be protected from correction. Reserve any hesitation to lift a dark object ONLY for the rare case of a room that is otherwise already well-exposed, where a single object is genuinely, intentionally black under good light. In every other case -- which is most real estate photos before correction -- recover full wood grain, carving detail, upholstery texture, and true material color in dark furniture with the same brightness and confidence applied to the rest of the room. A carved wood chair back should show its actual carving, clearly and brightly lit, not a silhouette with a few highlight edges. Do not let apparent darkness in the source file be mistaken for the object's real, intended appearance.

Ceiling
Treat the ceiling as one continuous painted surface. Fully resolve to bright, even, MLS-Bright white, while preserving the warm practical-light glow surrounding any fixture and believable luminance gradients from natural daylight direction.

Walls
Fully resolve wall color and exposure into one true, bright, even, continuous tone across the whole surface, matching MLS Bright standards -- a real wall does not change color or brightness partway across a room, and the corrected image should not either. Neutralize mixed color casts from combined daylight and interior lighting. Preserve subtle texture.

Windows
Recover only naturally visible detail. Do not invent scenery beyond the window. Do not create impossible exterior exposure. Maintain believable glass.

Carpet and Flooring
Fully recover natural texture and true tone at MLS Bright levels, including in areas currently in shadow. Increase local contrast as needed to reveal real texture. Maintain depth. Avoid an artificial HDR appearance.

Wood Furniture
Fully recover true wood tone and grain at full brightness, including on pieces currently underexposed toward black. Reveal real carving, joinery, and surface detail wherever the sensor captured it, however faint in the raw file. Do not create synthetic gloss or texture that was not actually captured.

Contrast
Produce the appearance of expert professional MLS photo processing -- a fully, evenly, BRIGHTLY resolved image, not a subtle adjustment. Avoid the specific failure modes of overdone HDR: halos, glow, excessive clarity, excessive dehaze, exaggerated sharpening. Avoiding those artifacts is about technique and quality, not a reason to limit how bright or fully corrected the room becomes.

Color
Maintain realistic color fidelity: neutral whites, natural wood tone, accurate wall color, believable warm practical lighting, balanced daylight -- all at full MLS Bright exposure levels.

Critical Rule
If any real, captured detail in this photo -- in furniture, walls, materials, or shadowed areas -- remains hidden in shadow, dim, or crushed toward black when the sensor data supports recovering it to MLS Bright, the correction is incomplete. The test is not "did I change too much" -- it is "does this read as true MLS Bright, professionally shot and edited, with every material and surface fully and brightly resolved." Under-correction that leaves the room dim or partially corrected is the failure mode to avoid here, the same way over-correction that invents data not supported by the capture is the failure mode to avoid elsewhere.

Final Objective
The finished image should appear indistinguishable from a professionally photographed and professionally edited MLS-Bright architectural real estate photograph of this exact room -- fully, confidently, brightly corrected, with every material and surface reading true and clearly, not a cautious partial improvement over the raw capture. Improve only photographic quality. Do not improve the property."""


EXTERIOR_ORACLE_PROMPT = """Smart Correct Exterior Oracle v2

PRIMARY ROLE
Treat the uploaded image as a real exterior photograph of an existing property.
Produce the same photograph after it has been captured and professionally post-processed by a world-class architectural real estate photographer.
This is a photographic optimization task only. It is NOT a beautification task.

Identity Preservation
Preserve exactly: Architecture, Roof, Stucco, Brick, Stone, Concrete, Driveway, Sidewalk, Street, Curbs, Landscaping, Lawn, Trees, Shrubs, Flower beds, Utility boxes, Mailboxes, Vehicles, Shadows, Camera position, Perspective, Lens, Composition, Time of day, Weather, Season, Exterior view.
Do not move, remove, add, replace or redesign anything.

Photographic Philosophy
Assume the photographer: exposed correctly, used professional dynamic range recovery, selected proper white balance, minimized lens flare, corrected lens distortion, balanced highlights and shadows, optimized local contrast, delivered a premium MLS-quality photograph. Improve only the photographic recording. Never improve the property itself.

Oracle Generation Objective
Assume this image will be used only as a photographic oracle to guide a deterministic reconstruction engine. Favor physically plausible photographic corrections over aesthetically pleasing improvements. If uncertainty exists, preserve documentary evidence rather than optimize appearance. The oracle should represent the maximum photographic quality achievable from the captured scene -- not an idealized version of the property.

Physical Evidence Rule
Treat every visible object and surface as documentary evidence. Improve only the recording of that evidence. Never improve: condition, health, maintenance, cleanliness, age, quality, appearance.

Sky
Treat the sky as physical evidence. Recover only information already supported by the captured photograph.
Do NOT: deepen blue, increase saturation, remove haze, create clouds, improve weather, increase drama, alter atmospheric conditions, change sun position, change season, create sunset, create twilight.
Recover only photographic information supported by the captured image, including tone, texture, color, and local contrast. Do not synthesize new scene information.

Lawn
Treat the lawn as documentary evidence. Preserve: color, health, irrigation, density, mowing pattern, seasonal appearance, worn areas, brown areas, patchiness.
Do NOT: green the lawn, improve health, increase density, repair dead spots, synthesize blades, increase saturation, improve maintenance. Only improve photographic recording.

Trees and Shrubs
Preserve: canopy size, branch structure, leaf density, seasonal color, health, pruning condition.
Do NOT: increase foliage, improve health, increase fullness, remove dead branches, add leaves, increase saturation, create richer greens. Recover only naturally captured detail.
Preserve relative shadow density throughout the scene. Shadow recovery should maintain smooth tonal transitions without increasing perceived contrast between adjacent illuminated and shaded surfaces.

Exterior Walls
Recover: texture, tonal separation, shadow detail, highlight detail. Preserve: stains, discoloration, fading, wear, imperfections. Never improve condition.

Roof
Recover only: texture, highlight rolloff, shadow detail. Never: recolor, clean, replace, repair, improve tile condition.

Driveway / Sidewalk
Preserve: stains, cracks, patches, discoloration, repairs, tire marks. Improve only: exposure, contrast, texture.

Windows
Recover only detail supported by the captured exposure. Do not: remove reflections, invent interiors, invent exterior scenery, change tint, clean glass, reduce dirt, improve condition.

Vehicles
Treat every vehicle as documentary evidence. Never: repaint, clean, reposition, replace, remove reflections. Recover only captured detail.

Mixed Lighting
Maintain the exact physical lighting environment. Do not: brighten the property, introduce fill light, add flash, add sunlight, change shadow direction, soften shadows, invent illumination. Improve only the camera's recording of the existing light.

Dynamic Range
Recover: shadow detail, highlight detail, midtone separation. Only where supported by captured data. Never create HDR appearance. Avoid: halos, glow, bloom, exaggerated local contrast, artificial clarity.

Surface Material Fidelity
Recover the authentic photographic appearance of each material (painted stucco, brick, concrete, asphalt, roof tile, wood, metal, glass, foliage, grass). Increase only the visibility of genuine material texture already present in the captured image. Do not synthesize texture. Do not sharpen beyond captured detail. Do not create surface detail unsupported by the original photograph.

Surface Continuity
Treat continuous surfaces independently (sky, stucco, garage door, driveway, asphalt, lawn, roof, sidewalks). Recover smooth, continuous tone while preserving authentic surface texture.

Color
Produce neutral professional color. Correct: camera white balance, color cast, exposure bias.
Do not: enhance color, beautify landscaping, enrich blue sky, deepen greens, increase curb appeal.

Critical Rule
If a correction would make the property appear to be in better physical condition than supported by the captured photograph, do not perform the correction.

Final Objective
The finished image should appear indistinguishable from a professionally photographed and professionally edited architectural real estate photograph captured under the exact same physical conditions. Improve only the quality of the photograph. Never improve the property."""


def _encode_multipart(fields: dict, files: dict) -> tuple:
    """
    Hand-built multipart/form-data body, stdlib only. fields: str->str.
    files: str-> (filename, bytes, content_type). Returns (body_bytes,
    content_type_header_value).
    """
    boundary = f"----OracleGen{random.randint(10**15, 10**16 - 1)}"
    parts = []

    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())

    for name, (filename, data, content_type) in files.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        parts.append(data)
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def _call_openai_edit(image_bytes: bytes, prompt: str) -> dict:
    """
    Single attempt, no retry logic here (see generate_oracle_scene for
    retry/backoff). Raises on any failure -- caller handles it.
    """
    fields = {
        "model": ORACLE_MODEL,
        "prompt": prompt,
        "size": IMAGE_SIZE,
        "quality": IMAGE_QUALITY,
        # moderation left at OpenAI's default ("auto") deliberately -- this
        # pipeline has no reason to request the more permissive "low" tier,
        # every input here is a real estate listing photo.
    }
    files = {
        "image": ("original.png", image_bytes, "image/png"),
    }
    body, content_type = _encode_multipart(fields, files)

    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload.get("data")
    if not data or not isinstance(data, list) or "b64_json" not in data[0]:
        raise ValueError(f"unexpected_response_shape: {str(payload)[:300]!r}")

    image_b64 = data[0]["b64_json"]
    return {"image_bytes": base64.b64decode(image_b64)}


def generate_oracle_scene(orig_image_bytes: bytes, is_exterior: bool) -> tuple:
    """
    Calls GPT Image 2 (images.edit) to produce an Oracle Scene Render from
    a given Original photo's raw bytes (PNG or JPEG). Selects the interior
    or exterior prompt based on is_exterior -- MUST be the same value this
    pipeline run already got from level0_scene_classifier.resolve_scene_type(),
    not re-derived here.

    Retries with exponential backoff + jitter (OpenAI's own stated
    guidance for this endpoint) on transient failures -- timeouts, 429/5xx.
    Does NOT retry on 4xx errors other than 429 (bad request, auth, org
    verification) -- those won't succeed on retry and burning MAX_RETRIES
    attempts on a guaranteed-repeat failure just adds latency for nothing.

    Returns (oracle_image_bytes, report). oracle_image_bytes is None on
    any failure after retries exhausted -- callers MUST treat None as
    "Oracle generation unavailable for this photo" and fall back to the
    existing pipeline's output, never as "use a blank/default Oracle."
    Never raises.
    """
    report = {
        "enabled": ORACLE_GENERATION_ENABLED,
        "called": False,
        "model": ORACLE_MODEL,
        "is_exterior": is_exterior,
        "attempts": 0,
        "error": None,
    }

    if not ORACLE_GENERATION_ENABLED:
        report["error"] = "disabled_via_env"
        return None, report
    if not OPENAI_API_KEY:
        logger.warning("oracleGeneration: missing OPENAI_API_KEY, degrading to None")
        report["error"] = "missing_OPENAI_API_KEY"
        return None, report

    prompt = EXTERIOR_ORACLE_PROMPT if is_exterior else INTERIOR_ORACLE_PROMPT

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        report["attempts"] = attempt
        report["called"] = True
        try:
            result = _call_openai_edit(orig_image_bytes, prompt)
            report["error"] = None
            return result["image_bytes"], report

        except urllib.error.HTTPError as e:
            body_snippet = ""
            try:
                body_snippet = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            last_error = f"HTTPError {e.code}: {body_snippet}"

            # Only retry on 429 (rate limit) or 5xx (transient server error).
            # A 400/401/403 (bad request, bad key, org not verified) will
            # fail identically every time -- stop immediately rather than
            # burn the full retry budget on a guaranteed repeat.
            retryable = e.code == 429 or 500 <= e.code < 600
            if not retryable or attempt == MAX_RETRIES:
                break

        except Exception as e:  # noqa: BLE001 -- timeouts, connection errors, bad response shape
            last_error = f"{type(e).__name__}: {e}"
            if attempt == MAX_RETRIES:
                break

        # Exponential backoff with jitter, per OpenAI's own guidance for this
        # endpoint -- jitter matters here specifically because multiple
        # Railway workers retrying a shared rate-limit hit in lockstep would
        # otherwise all land on the limiter at the same instant.
        sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1.0)
        logger.warning(f"oracleGeneration: attempt {attempt}/{MAX_RETRIES} failed ({last_error}), "
                        f"retrying in {sleep_s:.1f}s")
        time.sleep(sleep_s)

    logger.warning(f"oracleGeneration: all attempts failed, degrading to None. Last error: {last_error}")
    report["error"] = last_error
    return None, report
