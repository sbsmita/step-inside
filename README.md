# Step Inside

Photograph one page of a story and step inside it. In about 7–9 seconds the page is read, split into moments,
painted all around you and voiced, and you watch it happen line by line: each line read aloud (a different voice
for each character, every word lit up as it is spoken) while that moment is painted in 3D in front of you, inside
the story's own 360° world. It works in a desktop or phone browser, and in a Meta Quest over WebXR.

Then the world comes alive while you watch: the character the line is about moves (cut out of the painting by
OpenCV), the 360° world changes with every moment, its sky drifts with the wind OpenCV reads from the painting,
the sun glows, lamps flicker at night. With video enabled, every moment is filmed as a short clip the moment it
is painted, each one checked by OpenCV, and the story starts once its first scene is filmed.

> **Video status (Oct 26):** clips come from Luma Ray2 on Bedrock (Amazon Nova Reel is retired on this account).
> Luma is sold through AWS Marketplace: the account owner's first call subscribes the account. Clips take about
> 75 s each, one at a time, and OpenCV rejects clips that cut, zoom or don't show the action, so some moments stay
> living paintings. Characters act in the clips and every line is voiced (Polly), but mouths are not lip-synced.

Built for the OpenCV AI Competition 2026 with **OpenCV 5.0** and **AWS** (Bedrock: Nova Lite, Stability AI Stable
Image Core and Inpaint, Luma AI Ray2; Polly; Fargate on Graviton, Step Functions, S3, DynamoDB, CloudFront).

- Technical report: [docs/report.md](docs/report.md)
- Architecture: [docs/architecture.svg](docs/architecture.svg)

## How a page becomes a story in seconds

Agents work in parallel and everything is streamed to the browser as soon as it exists:

1. **OpenCV reads the page** (~0.1 s to flatten it and find its text lines; OCR runs alongside).
2. **The story agent** (Amazon Nova Lite) looks at the page and streams a plan: the page's own words split into
   spoken lines, who speaks each one, and what each moment should show. Painting starts as each moment arrives.
3. **Painters** (Stable Image Core) paint every moment and the world at once, ~3–4 s each.
4. **Voices** (Amazon Polly) read every line, with word timings for the live captions.
5. **OpenCV checks the work**: it puts back lines the story agent skipped, removes words it repeated, rejects
   pictures with lettering in them, erases people from the empty world, and gives every moment its depth.

Playback starts when the world, the first moment and its lines are ready, while later moments are still painting.

**Or just look at a page.** Tap **Look at a page with the camera** (phone, laptop, or a Quest whose browser allows
its cameras). OpenCV checks every frame and coaches you ("move closer", "hold still", "tilt it away from the
light"), captures the page by itself once it's readable, and recognizes pages it has seen before (ORB features +
homography), so a known page starts its story in about 2 seconds. A story made on a phone opens by itself on any
other screen showing the app, such as a headset.

**Living paintings.** OpenCV finds the sky, water and leaves in every moment (depth, colour and texture), and the
browser animates only those: the sky drifts, water ripples, leaves sway.

**A world to stand in.** The story starts on a quick 360° world; meanwhile a world agent paints the full
high-resolution 360° world outward from the first moment itself (5 overlapping Stability AI tiles checked by
OpenCV), so the scenery around you continues the picture in front of you, and it fades in when ready (~2 min).
OpenCV gives the world depth, so gliding with the thumbstick shows real parallax, and makes its sky and leaves move.

**The world comes alive, for any page.** The story agent writes what moves in each moment (a fall, a walk, a jump,
a sway) from the page's own words. Then, while the story plays:
- **Moving characters (~1 s):** OpenCV finds the character the line is about (YOLOX, depth, the story's own
  description), cuts it out with GrabCut, and Stability AI paints the scene behind it; the browser makes it move.
- **A world per moment:** each moment gets its own 360° world, repainted in full by the world agent (~2 min each).
- **World life:** OpenCV reads the wind from the sky's brush strokes, finds the sun or moon, lamps and the
  horizon; the browser animates them, plus rain, snow, stars or storm flashes from the story.
- **Video of every moment (when enabled):** each painted moment (16:9) is queued for filming as soon as it exists;
  if the page has its own picture, the first moment starts from it. OpenCV checks every clip (kept the painting,
  no lettering, no new figures, steady camera, moved the right way, no jump cuts, where to loop), asks once for a
  second take, else keeps the living painting. The viewer shows "Filming your story · n of N scenes" and starts on
  the moving picture.

**In VR:** the left thumbstick glides, the right one turns; Replay, Pause, Next and Exit float in the world (point
and pull the trigger, or pinch).

**Three ways in:** scan with the camera, upload a photo or screenshot, or upload a video of the page (OpenCV picks
its clearest frame). A page that already has a story is recognized and replayed instead of rebuilt.

## Quick start (local)

```bash
uv sync                                   # Python 3.12, pinned dependencies
uv run python -m vision.models.fetch      # OpenCV Zoo + Depth Anything V2 ONNX models
uv run uvicorn server.app:app --port 8000
```

Open http://localhost:8000 and scan a page with the camera, upload a photo or screenshot of a page, or upload a
video of a page.

- **AI features** use your AWS credentials (`aws login`). Without them, the app still runs: the page's lines as
  OpenCV read them, the page itself on the 3D screen, and browser speech.
- **Spending guard:** at most 3 stories are made at once; image calls follow the account's per-minute quota and
  `STABILITY_DAILY_LIMIT` (default 400 a day). Video needs nothing to be set: the app uses Luma Ray2 and finds the
  clips bucket `step-inside-clips-<account>-us-west-2` by itself (the deploy creates it; without it, video is off).
  Every setting can still be overridden: `VIDEO_BUCKET` (an S3 bucket in the model's region; empty turns video off) and
  `VIDEO_MODEL` (`luma.ray-v2:0` by default, us-west-2, one clip at a time, ~$3.75 a 5 s clip; or
  `amazon.nova-reel-v1:0`/`v1:1` where still served); `ANIMATE_MAX` (0 = every moment), `VIDEO_DAILY_LIMIT`,
  `PLAY_WHEN` (`all`, the default with video: the story opens once completely made; `clip`: once the first moment is filmed; `still`: at ~8 s). Without `VIDEO_BUCKET` there are no clips;
  everything else still runs.
- **One clip by hand:** `VIDEO_BUCKET=<bucket> python -m agents.animator still.jpg "what moves" out.mp4`, then
  `python -m vision.clip still.jpg out.mp4 --path fall` prints every score the critic uses.
- **Meta Quest** on the same Wi-Fi: `scripts/lan-https.sh`, then open the printed `https://…:8443` address in the
  Quest Browser and tap **Enter VR**. A = next line, B = pause, right thumbstick turns you.
- **Tests:** `uv run pytest -q`. **Evaluations:** `python -m eval.sfm_accuracy <frames>`; for reading accuracy,
  download the three public-domain illustrations named in `eval/make_pages.py`, then
  `python -m eval.make_pages <illustrations> <pages>` and `python -m eval.ocr_accuracy <pages>`.

## Deploy to AWS

Step-by-step guide for first-timers: [docs/DEPLOY.md](docs/DEPLOY.md). In short: create the judges' password in
Secrets Manager (`step-inside/app-password`), then `cd infra && npm install && npx cdk bootstrap && CDK_DOCKER=finch
npx cdk deploy --all -c video=luma`. The stack prints the CloudFront URL.

**Using it:** anyone can watch stories; creating one needs the shared password (**🔒 Log in**). A finished story
waits behind a magic door: **✨ Step inside** opens it and carries you through into the story.

## Layout

| Path | What |
|---|---|
| `vision/` | OpenCV 5: page reading, checks on painted moments, depth, 360° worlds, moving characters (`actors.py`), world life (`life.py`), the clip critic (`clip.py`), photo → 2.5D, video → 3D (SfM), Capture Coach |
| `agents/` | Bedrock and Polly agents: story agent (streamed), painters, the animator (Luma Ray2), voices, narrator, verifier |
| `server/` | FastAPI app; `live.py` runs the parallel agents; storage on local disk or S3 |
| `web/` | three.js WebXR viewer and the story theatre |
| `infra/` | AWS CDK stack |
| `eval/` | Reproducible evaluations and test-data generators |
# step-inside
