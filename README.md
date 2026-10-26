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

## Running the app

### Prerequisites

| Tool | Install | Notes |
|---|---|---|
| Python 3.12 | `brew install python@3.12` or [python.org](https://www.python.org/downloads/) | |
| uv | `brew install uv` or `pip install uv` | Fast Python package manager |
| Node.js 22 | `brew install node@22` | Only needed to minify JS; optional for local dev |
| AWS CLI | `brew install awscli` | Needed for AI features (painting, voices, video) |

### 1. Clone and install

```bash
git clone <your-repo-url>
cd opencv-hackathon
uv sync                                   # installs Python 3.12 + all dependencies
```

### 2. Download the vision models

```bash
uv run python -m vision.models.fetch
```

This downloads OpenCV Zoo models (PP-OCRv3, YuNet, YOLOX, Depth Anything V2) into `vision/models/`. It takes
a minute; they are ~250 MB total and only needed once.

### 3. Set up AWS (for AI features)

```bash
aws login
aws sts get-caller-identity          # confirm credentials work
```

Without AWS credentials the app still runs: OpenCV reads the page, the text is shown on a 3D screen, and the
browser reads it aloud with browser speech. The painted moments, voices, and video require AWS.

### 4. Start the server

```bash
uv run uvicorn server.app:app --port 8000
```

Open **http://localhost:8000** in a browser. You should see the Step Inside home page.

To create a story you need to log in. In local development no password is required: creating a story works
immediately. (On the deployed app, the password set in AWS Secrets Manager is required.)

### 5. Make your first story

Three ways to give the app a page:

- **Upload a photo or screenshot** — click **Bring your own page**, then drag or click to upload any JPG or PNG of a
  printed or digital page. Best with a clear, well-lit photo.
- **Upload a video** — same button; OpenCV picks the sharpest frame from the video.
- **Use the camera live** — click **Look at a page with the camera** (works on phones, laptops, Quest). OpenCV
  coaches you ("move closer", "hold still") and captures the page when it is readable.

The story builds in about 7–9 seconds without video; with video each moment is filmed as it is painted (~75 s per
clip). A progress bar shows each step. When the first scene is ready, a magic door appears: click **✨ Step inside**
to enter.

### 6. Meta Quest (VR)

Your Mac and Quest must be on the same Wi-Fi.

```bash
scripts/lan-https.sh
```

This prints a `https://…:8443` address. Open that address in the Quest Browser (accept the self-signed certificate
warning), log in, open any story, then tap **Enter VR**. Controls in VR:

| Button | Action |
|---|---|
| Left thumbstick | Glide through the world |
| Right thumbstick | Turn |
| A | Next line |
| B | Pause / resume |
| Point + trigger / pinch | Buttons (Replay, Next, Exit) |

### Environment variables (optional overrides)

| Variable | Default | What it does |
|---|---|---|
| `VIDEO_BUCKET` | auto-discovered | S3 bucket for video clips; empty string turns video off |
| `VIDEO_MODEL` | `luma.ray-v2:0` | `luma.ray-v2:0` (us-west-2) or `amazon.nova-reel-v1:0` (us-east-1) |
| `VIDEO_DAILY_LIMIT` | 24 | Max clips per day |
| `ANIMATE_MAX` | 0 (every moment) | Max moments filmed per story |
| `PLAY_WHEN` | `all` | `all`: wait for all clips; `clip`: start after first clip; `still`: start at ~8 s |
| `STABILITY_DAILY_LIMIT` | 400 | Max image calls per day |
| `APP_PASSWORD` | unset | Password to log in; unset = no login required (local dev) |
| `PORT` | 8000 | Server port |

Example — run locally with video enabled:

```bash
VIDEO_BUCKET=step-inside-clips-931628335592 \
AWS_REGION=us-east-1 \
PLAY_WHEN=all \
uv run uvicorn server.app:app --port 8000
```

### Tests and evaluations

```bash
uv run pytest -q                          # 99 unit + integration tests

# Clip critic
python -m vision.clip still.jpg out.mp4 --path fall    # score a single clip

# SfM (video → 3D) accuracy
python -m eval.sfm_accuracy <frames>

# OCR accuracy (needs public-domain illustrations; see eval/make_pages.py)
python -m eval.make_pages <illustrations> <pages>
python -m eval.ocr_accuracy <pages>
```

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
