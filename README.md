# Smart Stage PRO Plus — Render Service

Standalone Node.js/Express service deployed on Railway. Handles AI-motion
video generation for Smart Stage PRO Plus. Lives in its own repo, deployed
independently from the main `smart-stage-pro` Netlify site — see the
architecture isolation notes from the Phase 2 planning session.

## What This Service Does

Receives a video render job from `video-job.js` (Netlify function in the
main repo), downloads the staged/original images from Cloudinary, applies
Ken Burns-style motion to each, generates background music, assembles
everything into 16:9 and 9:16 MP4s, uploads the results to Cloudinary, and
calls back to a Netlify webhook with the finished URLs.

This service never writes to Supabase directly. All Supabase writes for
shared tables (`listings`, `staged_images`, `credit_ledger`) happen in
Netlify functions, which trust this service only to report job status via
webhook — not to make its own database decisions.

## Setup

1. `npm install`
2. Copy `.env.example` to `.env` and fill in real values for local testing
3. `npm start` — runs on port 3000 by default
4. Test health check: `curl http://localhost:3000/health`

## Deploying to Railway

1. Push this repo to its own GitHub repo (e.g. `szieour-ctrl/smart-stage-pro-plus-render`)
2. Create a new Railway project, link the GitHub repo
3. `nixpacks.toml` ensures FFmpeg is installed automatically — no manual
   Dockerfile needed
4. Set all variables from `.env.example` as Railway environment variables
5. Railway auto-deploys on push to main

## Current Status of Each Dependency

| Dependency  | Status                                            |
|-------------|----------------------------------------------------|
| FFmpeg      | Fully functional — Railway installs via nixpacks    |
| Cloudinary  | Fully functional — uses existing PRO account creds  |
| Mubert      | **Stubbed with silent-track fallback** — pipeline runs end-to-end without it, but no real music until `MUBERT_API_KEY` is set |
| Webhook     | Fully functional — needs `video-notify.js` built on the Netlify side (see main repo) |

## Testing Without Mubert or Railway Yet

You can test the entire pipeline locally right now:

```bash
node -e "
const { processRenderJob } = require('./src/renderPipeline');
processRenderJob({
  jobId: 'test-123',
  projectId: 'test-project',
  formats: ['16x9'],
  musicStyle: 'Organic Modern',
  frames: [
    { imageUrl: 'https://res.cloudinary.com/.../staged1.jpg', roomType: 'living', motionPreset: 'auto', durationSeconds: 4.5, sequenceOrder: 0 },
    { imageUrl: 'https://res.cloudinary.com/.../staged2.jpg', roomType: 'kitchen', motionPreset: 'auto', durationSeconds: 4.5, sequenceOrder: 1 }
  ]
}).then(() => console.log('Pipeline test complete')).catch(console.error);
"
```

Replace the imageUrl values with two real Cloudinary URLs from an existing
staged listing to test against real assets. Without `MUBERT_API_KEY` set,
this will use a silent audio track and still produce a watchable test video
— useful for validating motion presets and assembly before the Mubert
account exists.

## Known Approximations to Refine During Testing

- `concatenateClips()` in `assemble.js` uses an approximate clip duration
  (4.5s minus crossfade) to calculate xfade offsets across the whole
  sequence. Once real clip durations vary per room, this offset math should
  be made exact by tracking actual durations from `motionPresets.js` output
  rather than assuming a fixed length.
- 9:16 reframe uses a simple center crop. Smart subject-aware cropping
  (keeping a fireplace or chandelier centered rather than literal frame
  center) is a Phase 2B refinement.
