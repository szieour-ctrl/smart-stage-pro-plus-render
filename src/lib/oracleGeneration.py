"""
oracleGeneration.py

Produces the Oracle Scene Render -- a live, in-pipeline generative render
of a given Original photo, used purely as a per-pixel delta source by
oracleCorrection.py's alignment + LAB correction step. Never shipped to
the client directly.

ENDPOINT: fal.ai's Flux Kontext [pro] image-editing model
(fal-ai/flux-pro/kontext), NOT OpenAI's GPT Image 2. This module was
originally built against OpenAI's images.edit endpoint; that version is
now retired. The switch was made on real measured evidence, not
preference: head-to-head testing found Flux Kontext's alignment quality
at 2847/2857 inliers (0.65px mean residual) vs GPT Image 2's 32/68
inliers on the same photo, plus 3-8s latency vs up to ~2 minutes for
GPT Image 2 -- which also removes this module from the OpenAI
organization-wide 50 IPM limit that PRO staging (gpt-image-2) already
consumes.

AUTH: FAL_KEY, NOT OPENAI_API_KEY. This is fal.ai's own conventional env
var name. If generate_oracle_scene() reports missing_FAL_KEY despite the
key being present on Railway, check the Railway variable is spelled
exactly "FAL_KEY" before assuming anything else is wrong -- this project
already burned a session on the equivalent OpenAI-key-name confusion.

TRANSPORT: plain REST via fal.ai's queue endpoints (submit / status /
result), stdlib only -- no `requests` package, no `@fal-ai/client` SDK.
Mirrors the same explicit queue.submit() / queue.status() / queue.result()
pattern already established in klingMotion.js and ltxMotion.js, just
over urllib since this module is Python. Submitting a fire-and-poll job
rather than a single blocking call also gives this module a request_id
that's recoverable from the fal.ai dashboard even if the Railway process
dies mid-poll -- the same reasoning that shaped the video pipeline's own
polling design.

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
condition than the captured photo supports, do not perform it").

INTERIOR now uses v5b (Smart_Correct_Oracle_Interior_v3.docx naming lags
three versions behind the actual prompt text below -- rename the docx to
v5b next time it's regenerated). History:
v1 -> v2: v1's language was restrained-by-default with no explicit
permission to recover aggressively -- confirmed on a real photo (Flux
Kontext, IMG_8310): the interior result was a near-total no-op across
the ENTIRE room. v2 added an explicit "Dynamic Range: recover
shadow/highlight/midtone" instruction plus a "Furniture and Materials
in Shadow -- Recover, Do Not Protect" section.
v2 -> v3: v2's abstract "fully recover" language still weakly adopted --
confirmed on the SAME photo, chair L stayed at 42.8 (vs raw Original's
42.0, a near no-op). v3 named "MLS Bright" explicitly, 11 times -- chair
L jumped to 73.6 with recovered carving detail (std 74.7, exceeding the
GPT-based Oracle's own 69.9). This was the single highest-leverage fix
of the whole prompt-iteration history.
v3 -> v4: v3's aggressive brightness push was found (real photo,
side-by-side against a separately-generated "Strobe" render of the same
room) to carry a consistent, whole-frame warm color-temperature bias:
+3.3 mean LAB-b shift vs the Strobe version's +1.1. v4 adds a dedicated
"Color Temperature -- Independent of Brightness" section stating this
directly, and adds warm color drift as a named third failure mode in
Critical Rule, alongside under- and over-correction.
v4 -> v5a: v4 gave ceiling and walls the same "hold captured color
exactly" rule under Color Temperature -- correct for walls, wrong for
ceiling, since a ceiling's captured color under warm practical light is
often already cast-contaminated and "hold the captured color" prevented
Oracle from ever correcting that cast toward the ceiling's true neutral
paint. v5a rewrites the ceiling treatment as adaptive (neutralize cast
if the true material is white/off-white, preserve hue if the true
material is colored) and gives ceiling its own dedicated white/trim
separation section, while walls keep the v4 hold-true-color rule
unchanged. Validated in the fal.ai playground against IMG_8310 (strong
gray-blue walls -- ceiling separated cleanly, wall hue held) and
IMG_8291 (walls and ceiling naturally close in tone -- brightness lift
landed but ceiling/wall separation stayed weak, confirming this is a
scene-contrast ceiling on what the prompt alone can do, not a prompt
defect; further pop on low-contrast scenes is expected to come from the
recoverability-driven correction pipeline, not further prompt tuning).
v5a -> v5b: v5a's Global Exposure Directive included "Target histogram
center at approximately 70% luminance" -- a fixed value, not a floor.
Confirmed on a real already-bright photo (2089 Thornecroft Ln, new-
construction neutral gray room, all lights on): Original frame mean
luminance measured ~73% -- already above the 70% target -- and the
resulting Oracle came back measurably DARKER than the Original across
every region (walls, ceiling, carpet, hallway all down 5-13 RGB points,
whole-frame mean down ~7%), the opposite of the intended "MLS Bright,
never darker" behavior. Root cause: a histogram-center target is a
bidirectional normalization instruction by nature -- it pulls frames
above the target down just as readily as it lifts frames below it up.
v5b removes the fixed percentage entirely and replaces it with a
one-directional floor (lift underexposed frames, never darken an
already well-exposed one) plus a highlight-detail-preservation ceiling
(stop lifting a region once texture/grain would clip to flat white),
since detail preservation -- not a target number -- is the actual
reason any upper bound belongs in this directive at all.

LABELING: every image this module produces is, by definition, the
Oracle Scene Render -- a digitally altered image. That's not a caveat
for this module to enforce (see oracleCorrection.py's own note on this),
it's a fact about anything this function returns, and it must be labeled
wherever shown, per the settled position this project already reached.
"""

import os
import json
import time
import random
import base64
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

ORACLE_GENERATION_ENABLED = os.environ.get("ORACLE_GENERATION_ENABLED", "true").lower() not in ("false", "0", "")

# fal.ai's standard auth env var -- NOT OPENAI_API_KEY. See module
# docstring's "AUTH" section: this is flagged as previously confused
# with the OpenAI key name, not just a code-level rename.
FAL_API_KEY = os.environ.get("FAL_KEY", "")

# Endpoint ID for fal.ai's Flux Kontext [pro] image-editing model, per
# fal.ai's own API docs (fal.ai/models/fal-ai/flux-pro/kontext/api).
# Overridable via env in case a cheaper/faster Kontext variant (e.g.
# fal-ai/flux-kontext/dev) is ever worth testing against a real batch --
# do not change the default without re-running the same head-to-head
# validation this session ran for gpt-image-2 vs Flux Kontext pro.
ORACLE_MODEL = os.environ.get("ORACLE_GENERATION_MODEL", "fal-ai/flux-pro/kontext")

FAL_QUEUE_BASE = "https://queue.fal.run"

# Timeout for each individual HTTP call to fal.ai's queue endpoints
# (submit / poll / fetch-result) -- these are quick control-plane calls,
# NOT the generation itself, which is tracked separately via polling
# below.
HTTP_TIMEOUT_SECONDS = int(os.environ.get("ORACLE_GENERATION_HTTP_TIMEOUT_SECONDS", "30"))

# How often to poll fal.ai for job completion, and how many times to poll
# before giving up on a single submit/poll/result cycle. Sized around
# this session's measured Flux Kontext latency (3-8s) with generous
# headroom, NOT copied from the video pipeline's 10s/90-attempt (15 min)
# settings -- those are sized for Kling/LTX video jobs, a completely
# different latency class. 60 * 2s = 120s ceiling, ~15-40x the measured
# render time -- same margin-over-observed-time ratio the video pipeline
# uses, just scaled to a much faster job. Revisit if production Flux
# Kontext calls are routinely slower than this session's measurements.
POLL_INTERVAL_SECONDS = float(os.environ.get("ORACLE_GENERATION_POLL_INTERVAL_SECONDS", "2.0"))
MAX_POLL_ATTEMPTS = int(os.environ.get("ORACLE_GENERATION_MAX_POLL_ATTEMPTS", "60"))

# Retries the ENTIRE submit->poll->result cycle on transient failure.
# Fewer attempts and a shorter base backoff than the old OpenAI version
# (3 retries / 2.0s base carried over unchanged from OpenAI's ~2-minute
# guidance) -- Flux Kontext's fast, consistent latency doesn't call for
# the same posture. Still jittered: the reason for jitter (a fleet of
# Railway workers retrying a shared rate-limit hit in lockstep) doesn't
# depend on which API is on the other end.
MAX_RETRIES = int(os.environ.get("ORACLE_GENERATION_MAX_RETRIES", "3"))
BASE_BACKOFF_SECONDS = float(os.environ.get("ORACLE_GENERATION_BASE_BACKOFF_SECONDS", "1.0"))

# png, not fal's own default (jpeg) -- oracleCorrection.py computes
# per-pixel LAB deltas between this render and the Original. JPEG
# compression artifacts in the Oracle render would leak directly into
# that delta as false correction signal. Do not change this default
# without checking oracleCorrection.py's alignment/delta code can
# tolerate the change.
OUTPUT_FORMAT = os.environ.get("ORACLE_GENERATION_OUTPUT_FORMAT", "png")


# ---- Prompt text, embedded directly (not read from the .docx at runtime) ----
# Sourced verbatim from Smart_Correct_Oracle_v1-Prompt.docx (interior) and
# Smart_Correct_Exterior_Oracle_v2.docx (exterior, v2 -- see module
# docstring for why v2 and not v1). Embedded as constants so this module
# has no runtime dependency on the docx files or a document-reading step.
# IF EITHER PROMPT DOCX IS REVISED AGAIN, THIS MUST BE UPDATED TO MATCH --
# there is no automated sync between the docx and this constant.

INTERIOR_ORACLE_PROMPT = """Smart Correct Oracle v5b

Primary Role
Generate an idealized MLS Bright lighting map of the same room, preserving all geometry, materials, and decor exactly.
Lighting may be synthetic, natural, or time-shifted, but must remain physically plausible and fixture-consistent.
This Oracle defines how bright each pixel should be if fully resolved, not what new pixels to invent.

Global Exposure Directive
This is a strictly one-directional brightening operation, never a normalization toward a
fixed value. Lift midtones and shadows until no area reads dim or hidden. There is no
target brightness level to converge on -- only a floor to lift up to and a highlight
ceiling to avoid crossing.

If the captured frame is already well-exposed, do not reduce its brightness in any
region. An already-bright frame must never come out of this step darker, in any area,
than the frame that went in.

The only ceiling on brightness is highlight-detail preservation: stop lifting a region
once material texture, grain, or surface detail in that region would clip to flat white
or become illegible. Recoverable detail must remain recoverable -- this is a correction
reference for a downstream pixel-level engine, and a blown-out region has nothing left
in it to correct against.

Ceiling brightness anchors the frame -- equal to or slightly above wall luminance.

Maintain believable daylight direction and fixture glow.

Artificial light may simulate daylight bounce, ceiling fill, or wall wash to achieve even luminance.

Do not add or invent fixtures, bulbs, or lamps.

Ceiling -- Adaptive Neutrality
Treat the ceiling as a diffuse reflector whose true material color must be fully and evenly resolved.

If white/off-white: neutralize warm or mixed-light casts to clean white.

If colored (beige, gray, wood, coffered): preserve hue, remove lighting cast, brighten evenly.

Never bleach or repaint; only neutralize lighting artifacts.

Ceiling should read luminous and open, not shadowed or dull.

Maintain fixture glow and natural falloff.

Ceiling brightness normalization must harmonize with trim and molding luminance separation. Apply local white-balance correction and highlight refinement to achieve clean, luminous whites without loss of texture.

Ceiling & Trim White Separation
Where ceilings, crown molding, baseboards, doors, window trim, or other painted light-neutral surfaces are source-visible as white or near-white, render them as clean photographic whites consistent with their actual material color.
Neutralize ambient yellow, beige, or mixed-light contamination without changing the underlying paint color.

Increase luminance separation between light ceilings/trim and adjacent walls only where supported by the source image. Preserve surface texture, shadow gradients, recessed-light falloff, and natural room illumination.
Do not bleach, clip, flatten, or convert beige/cream materials into white.

Professional ceiling rendering: light-colored ceilings should read clean, luminous, and visually separated from surrounding walls, as they would in professionally balanced real-estate photography.
Do not accomplish this through global exposure increase. Use localized white-balance correction, highlight/midtone refinement, and controlled luminance separation.

Walls
Preserve true paint color at full brightness.

Neutralize mixed lighting casts.

Maintain texture and even tone across the surface.

Do not warm or shift color temperature as exposure increases.

Brightness and color temperature are independent; a wall's true color must remain accurate even at full MLS Bright exposure.

Windows
Recover visible exterior detail only.

Maintain believable glass reflections and exposure balance.

Do not invent scenery or create impossible exterior exposure.

Furniture & Flooring
Recover full texture, grain, and tone at MLS Bright luminance.

Brighten confidently -- dark furniture is usually underexposed, not truly black.

Avoid synthetic gloss or HDR artifacts.

Reveal real material detail wherever sensor data supports recovery.

Color Fidelity
Preserve true material color and texture.

Neutralize mixed lighting casts.

Ceiling color correction is adaptive: neutralize cast, preserve genuine hue.

Walls retain true paint color at full brightness.

Maintain balanced daylight warmth and neutral whites.

Contrast & Clarity
Produce professional MLS photo quality -- bright, crisp, evenly lit, spacious.
Avoid halos, glow, excessive clarity, or oversharpening.

CV Integration Directive
The Oracle is a synthetic luminance reference, not a photo edit.

Every pixel corresponds to a real surface in the original HDR capture.

Artificial light is permitted only to reveal existing sensor data.

No invented geometry, reflections, or materials.

The Oracle defines ideal exposure per pixel for CV correction.

Critical Rule
If any real captured detail remains dim or crushed when sensor data supports recovery, correction is incomplete.
Under-correction, over-correction, or color drift are all failures.
The final image must read as a professionally photographed MLS Bright listing photo of the same room.

Final Objective
Deliver a synthetic MLS Bright lighting reference of the identical room --
identical geometry, materials, and decor, but fully resolved luminance and color balance.
Lighting may be artificial or natural, but must remain physically plausible and fixture-consistent.
The result should serve as a CV-ready ideal map for pixel correction, indistinguishable from a professionally lit MLS photo in exposure and tone."""


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


def _detect_mime_type(image_bytes: bytes) -> str:
    """
    Magic-byte sniff, stdlib only -- no python-magic dependency. Only the
    two formats this pipeline actually produces (PNG from screenshots/
    Cloudinary derivatives, JPEG from camera originals) need to be
    distinguished; anything else falls back to PNG rather than guessing
    wrong and mislabeling the data URI.
    """
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def _build_data_uri(image_bytes: bytes) -> str:
    """
    fal.ai's file-input fields (image_url here) accept either a hosted
    URL or a base64 data URI -- per fal.ai's own docs. This pipeline has
    no existing step that uploads the Original photo to a public URL
    before Oracle generation runs, so a data URI is the simpler, more
    self-contained choice (no new Cloudinary round-trip dependency added
    to this module just to satisfy fal.ai's input format). Revisit if a
    real batch shows the data-URI path meaningfully slower than a hosted
    URL would be -- fal.ai's own docs note large base64 payloads can
    affect request performance, but this pipeline's photo sizes haven't
    been checked against that threshold yet.
    """
    mime_type = _detect_mime_type(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _http_json(method: str, url: str, headers: dict, body: Optional[dict] = None) -> dict:
    """
    Single stdlib HTTP call to a fal.ai queue endpoint, JSON in and out.
    Raises urllib.error.HTTPError on 4xx/5xx (caller handles retry
    logic) or ValueError on an unparseable/non-JSON response body.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        raw = resp.read().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"non_json_response: {raw[:300]!r}") from e


def _call_flux_kontext_edit(image_bytes: bytes, prompt: str) -> dict:
    """
    Single submit->poll->result cycle against fal.ai's Flux Kontext [pro]
    queue endpoint. No retry logic here (see generate_oracle_scene for
    retry/backoff across whole cycles) -- raises on any failure, caller
    handles it. Mirrors the explicit queue.submit() / queue.status() /
    queue.result() pattern already established in klingMotion.js and
    ltxMotion.js, just via plain REST since this module is Python stdlib
    only rather than the @fal-ai/client JS SDK those files use.
    """
    if not FAL_API_KEY:
        raise ValueError("missing_FAL_KEY")

    auth_headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

    submit_url = f"{FAL_QUEUE_BASE}/{ORACLE_MODEL}"
    submit_body = {
        "prompt": prompt,
        "image_url": _build_data_uri(image_bytes),
        "output_format": OUTPUT_FORMAT,
        # aspect_ratio deliberately omitted -- Kontext preserves the
        # input image's own aspect ratio when it isn't specified, which
        # is what a correction task needs (same framing in, same framing
        # out). Setting an explicit aspect_ratio here would risk Flux
        # reframing/cropping the photo relative to the Original, which
        # would break oracleCorrection.py's alignment step downstream.
    }
    submit_payload = _http_json("POST", submit_url, auth_headers, submit_body)

    request_id = submit_payload.get("request_id")
    if not request_id:
        raise ValueError(f"submit_missing_request_id: {str(submit_payload)[:300]!r}")

    # Prefer the URLs fal.ai's own response hands back (per fal.ai's docs,
    # the REST API's response includes URLs for each operation) -- fall
    # back to constructing the standard queue paths only if those keys
    # are absent, so a future response-shape change degrades gracefully
    # instead of breaking outright.
    status_url = submit_payload.get("status_url") or f"{submit_url}/requests/{request_id}/status"
    response_url = submit_payload.get("response_url") or f"{submit_url}/requests/{request_id}"

    logger.info(f"oracleGeneration: fal.ai request queued -- request_id={request_id} "
                f"(recoverable via fal.ai dashboard even if this process dies)")

    final_status = None
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL_SECONDS)
        status_payload = _http_json("GET", status_url, auth_headers)
        status = status_payload.get("status")

        if status == "COMPLETED":
            final_status = status_payload
            break
        if status == "FAILED" or status == "ERROR":
            raise ValueError(f"fal_request_failed: {str(status_payload)[:300]!r}")
        # IN_QUEUE / IN_PROGRESS -- keep polling.

    if final_status is None:
        raise TimeoutError(
            f"fal.ai request {request_id} did not complete within "
            f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS:.0f}s of polling. "
            f"Check this request_id directly on the fal.ai dashboard -- "
            f"generation may have finished even if polling here gave up."
        )

    result_payload = _http_json("GET", response_url, auth_headers)
    images = result_payload.get("images")
    if not images or not isinstance(images, list) or "url" not in images[0]:
        raise ValueError(f"unexpected_response_shape: {str(result_payload)[:300]!r}")

    image_url = images[0]["url"]

    # Download the actual image bytes from fal.ai's CDN. This is a plain
    # public media URL, not an authenticated fal.ai API endpoint -- no
    # Authorization header needed or sent here.
    download_req = urllib.request.Request(image_url, method="GET")
    with urllib.request.urlopen(download_req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        image_bytes_out = resp.read()

    return {
        "image_bytes": image_bytes_out,
        "fal_request_id": request_id,
        "seed": result_payload.get("seed"),
    }


def generate_oracle_scene(orig_image_bytes: bytes, is_exterior: bool) -> tuple:
    """
    Calls fal.ai's Flux Kontext [pro] endpoint to produce an Oracle Scene
    Render from a given Original photo's raw bytes (PNG or JPEG). Selects
    the interior or exterior prompt based on is_exterior -- MUST be the
    same value this pipeline run already got from
    level0_scene_classifier.resolve_scene_type(), not re-derived here.

    Retries the full submit->poll->result cycle with exponential backoff
    + jitter on transient failures -- HTTP timeouts, 429/5xx on submit,
    fal.ai queue FAILED status, or a poll timeout. Does NOT retry on a
    non-429/5xx HTTPError on the submit call itself (bad request, bad
    key) -- those won't succeed on retry and burning MAX_RETRIES attempts
    on a guaranteed-repeat failure just adds latency for nothing.

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
        "fal_request_id": None,
        "error": None,
    }

    if not ORACLE_GENERATION_ENABLED:
        report["error"] = "disabled_via_env"
        return None, report
    if not FAL_API_KEY:
        logger.warning("oracleGeneration: missing FAL_KEY, degrading to None")
        report["error"] = "missing_FAL_KEY"
        return None, report

    prompt = EXTERIOR_ORACLE_PROMPT if is_exterior else INTERIOR_ORACLE_PROMPT

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        report["attempts"] = attempt
        report["called"] = True
        try:
            result = _call_flux_kontext_edit(orig_image_bytes, prompt)
            report["error"] = None
            report["fal_request_id"] = result.get("fal_request_id")
            return result["image_bytes"], report

        except urllib.error.HTTPError as e:
            body_snippet = ""
            try:
                body_snippet = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            last_error = f"HTTPError {e.code}: {body_snippet}"

            # Only retry on 429 (rate limit) or 5xx (transient server error).
            # A 400/401/403 (bad request, bad key) will fail identically
            # every time -- stop immediately rather than burn the full
            # retry budget on a guaranteed repeat.
            retryable = e.code == 429 or 500 <= e.code < 600
            if not retryable or attempt == MAX_RETRIES:
                break

        except Exception as e:  # noqa: BLE001 -- timeouts, connection errors, fal FAILED status, poll timeout, bad response shape
            last_error = f"{type(e).__name__}: {e}"
            if attempt == MAX_RETRIES:
                break

        # Exponential backoff with jitter -- jitter matters here specifically
        # because multiple Railway workers retrying a shared rate-limit hit
        # in lockstep would otherwise all land on the limiter at the same
        # instant.
        sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1.0)
        logger.warning(f"oracleGeneration: attempt {attempt}/{MAX_RETRIES} failed ({last_error}), "
                        f"retrying in {sleep_s:.1f}s")
        time.sleep(sleep_s)

    logger.warning(f"oracleGeneration: all attempts failed, degrading to None. Last error: {last_error}")
    report["error"] = last_error
    return None, report
