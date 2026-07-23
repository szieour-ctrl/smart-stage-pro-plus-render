// lib/endFrame.js — Railway
//
// "End Frame": the address/CTA closing card appended after the main
// video finishes.
//
// REVISED (this session, real architecture correction): this file used
// to own the whole feature — replacing the last room frame's image with
// an AI-edited or composited version. That was wrong on the merits (Sam:
// "the last clip has always been the most important clip in the entire
// video" — it must never be replaced or motion-downgraded) and kept
// producing real bugs tied to that approach across this session (Flux
// spelling/duplication errors, zoom-cropping the baked-in text, Kling
// silently never seeing the edit at all).
//
// The actual feature now lives in assemble.js's revived closing-card
// system (renderClosingCardClip + appendClosingCardWithAudio, wired
// in via the closingCard param on assembleVideo) — a genuinely separate,
// short clip appended AFTER the main video (narration and all) finishes,
// with its own isolated music-only audio stinger, built from a still copy
// of the actual last room photo. The last room clip itself is completely
// untouched now — same motion, same narration, same spoken CTA it would
// have with this feature disabled entirely.
//
// This file's only remaining job is the feature flag — kept separate so
// the kill switch is one clearly-named, easy-to-find place regardless of
// which module the actual rendering logic lives in.
//
// Set END_FRAME_ENABLED=false (or leave unset) in Railway's environment
// to disable the closing card entirely — no code change or redeploy
// needed, just an env var flip + restart.
//
// Related env vars (read directly in renderPipeline.js / assemble.js,
// not here — this file only owns the on/off switch):
// - END_FRAME_CTA_TEXT — the CTA line (default: "Schedule Your Private
//   Showing")
// - END_FRAME_TEXT_COLOR — card text color (default: "white")

function isEndFrameEnabled() {
  return String(process.env.END_FRAME_ENABLED || "").toLowerCase() === "true";
}

module.exports = { isEndFrameEnabled };
