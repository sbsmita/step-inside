# Step Inside: technical report

*OpenCV AI Competition 2026 entry. OpenCV 5.0 + AWS.*

## 1. Problem and users

A story on a page is something you look at. **Step Inside** lets anyone photograph one page — a picture-book page,
a poem, a poster, a screenshot — and, within seconds, stand inside it and watch it happen line by line:

- Each line is read aloud, in a narrator's voice or the voice of the character who says it, with every word lit
  up as it is spoken.
- While a line is read, the moment it describes is painted in front of you on a giant curved screen that has real
  depth, inside a 360° world of the story's place. When the story moves on, the picture changes.
- The scanned page floats beside you with the line being read highlighted, so you can follow along.

It runs in any desktop or phone browser and, with no install, in a Meta Quest over WebXR.

Users: children and parents at bedtime; teachers (literacy, comprehension, poetry); people who find reading hard or
tiring (the text is read aloud with word-by-word captions); anyone who wants to see a story they love come alive.

Speed is the point: a story you have to wait minutes for is not one you step into. Measured on the deployed
pipeline's code, the first line plays **6–9 s** after the photo is sent.

## 2. What happens to a page (parallel agents, streamed)

```
reader (OpenCV) ─┬─ flattened page ─> story agent (Nova Lite, streamed) ─┬─ header ──────> world painter ─> OpenCV: erase figures, wrap 360°
                 │                                                        ├─ each moment ─> moment painter ─> OpenCV: lettering check, depth
                 │                                                        └─ each line ───> voice (Polly + word timings)
                 └─ OCR (OpenCV, parallel) ─> checks the story agent: restores skipped lines, drops repeats, finds each line on the page
```

| Step | What runs | Where |
|---|---|---|
| 1. Flatten the page | Contour + 4-corner fit, accepted only if the detected text lies inside it; `getPerspectiveTransform` / `warpPerspective`. Small images (a 700 px poster) are enlarged so letters are ~20 px tall | `vision/page.py`, `vision/clean.py` |
| 2. Find and read the text | PP-OCRv3 DB text detection and CRNN recognition, OpenCV Model Zoo ONNX models through `cv.dnn`. Each line's polarity is found by Otsu (white-on-blue posters are inverted), words are split by projection profile, and words are recognized by 8 model copies in parallel (8.2 s → 1.9 s on a 160-word poster) | `vision/page.py`, `vision/netpool.py` |
| 3. Plan the story | Amazon Nova Lite (Bedrock `ConverseStream`) looks at the flattened page and streams JSON Lines: a header (title, painting style, the place, the cast with fixed looks), then each moment (the page's own words as spoken lines, the speaker of each, what to show). Each object is handed over the moment it closes. Code then fixes what the model gets wrong: quoted speech is split from narration and given to the character the narration names ("said old Mrs. Rabbit"); prose copied with printed line breaks is rejoined into sentences; at most 6 moments | `agents/planner.py` |
| 4. Paint | Stability AI Stable Image Core (Bedrock, us-west-2) paints each moment as a 2304×960 picture in ~3–4 s, all moments in parallel; the world is painted from the header while the moments are still being planned. A token bucket per model keeps calls inside the account's quota (9/min for Core), and SD3.5 Large takes the overflow | `agents/imagery.py` |
| 5. OpenCV checks every picture | **Lettering:** PP-OCRv3 DB detection on the painting; word-like text covering >0.4% of it → painted again (clean paintings measure 0.0000, lettered ones 0.013). **Depth:** Depth Anything V2 Small through `cv.dnn` + `ximgproc.guidedFilter` → an 8-bit depth map; the browser bends the curved screen so near things come toward you. **World:** YOLOX (`cv.dnn`) on overlapping tiles finds people the painter added anyway, and `xphoto` shift-map inpainting rebuilds the scenery behind them (Telea and FSR smear; shift-map, done per figure at half size, takes 0.5–1 s); the wide painting is wrapped into a seamless 2:1 equirectangular world (ends cross-faded behind you, calm poles) | `vision/shot.py` |
| 6. Voices | Amazon Polly (neural) reads every line, plus word speech marks (byte offsets converted to characters) for the captions; narrator and one voice per character; adaptive retries pace calls under Polly's rate limit | `agents/voice.py` |
| 7. OpenCV checks the story agent | The agent's lines are aligned with OpenCV's OCR character by character (`difflib`). OCR lines no planned line covers are put back where they belong (after the planned line covering the line above), spelled with the words the agent read elsewhere ("ittle" → "little", compared by letter shape so "iight" → "light"); words the agent wrote twice that the page shows once are removed; every spoken line gets its box on the page for highlighting | `server/live.py` |
| 0. Look (optional, live camera) | Every frame (~0.1 s, `POST /api/look`): PP-OCRv3 text lines, median text height, Laplacian variance over the text (sharpness), clipped white brighter than the paper (glare), mean brightness, text touching the frame edge. The first problem becomes a spoken instruction ("Move closer: the words are too small to read"); two good frames in a row → the full-size frame is captured. Known pages: ORB (1500 features) + ratio test + RANSAC homography against every finished story's page, ≥30 inliers and clearly better than the runner-up, starts that story without any AI call. Active perception: the system looks, decides and directs the user | `vision/look.py`, `web/app.js` |
| 5b. Living paintings | For every moment, from the Depth Anything map, HSV colour and local texture (standard deviation): **sky** = far, smooth, not foliage, connected to the top edge; **leaves** = green and textured; **water** = blue, low, smooth and wide, searched only when the story agent's description mentions water (by colour alone a blue coat or night hills would ripple). Soft masks → a motion PNG; the browser's shader moves only those pixels (seamless two-phase flow for sky and water, ripple, sway) | `vision/shot.py`, `web/theatre.js` |
| 0b. A video of the page | Active perception on a video: OpenCV judges up to 40 evenly spaced frames with the photo check (text found, size, sharpness, glare) and keeps the one it reads best | `server/app.py` (`best_frame`) |
| 0c. Seen it before | Every upload is first matched against finished stories' pages (ORB + RANSAC homography, ≥60 inliers); a match replays that story instead of rebuilding it | `server/app.py`, `vision/look.py` |
| 4b. The full 360° world (world agent, second pass) | Once the first moment is painted, Stability AI Inpaint paints a 4096×2048 world outward from it: the moment is placed at the exact angles the curved screen covers (150° × its cylindrical height), then 5 overlapping tiles, each continuing its neighbours, OpenCV's flat-field check on every tile, a two-scale YOLOX scan for stray figures, colour transfer in Lab. ~2 minutes while the story already plays; the browser cross-fades it in over the quick world | `server/live.py` (`_world_full`), `vision/world.py`, `server/stories.py` |
| 5c. The world has depth and life | Depth Anything V2 (`cv.dnn`) on the equirectangular world → the sphere's radius follows depth (near scenery up to half as far as the sky, none at the poles), so thumbstick gliding (±3 m) shows parallax; the same sky/leaves/water masks animate the world | `vision/shot.py`, `web/theatre.js` |
| 8. Watch | three.js WebXR: the 360° world sphere; the moment on a 150° curved screen displaced by its depth (192×80 vertices), edges feathered into the world, slow push-in, 0.9 s crossfades; a caption panel and the scanned page in 3D (HTML is invisible in a headset); ambient particles per moment (stars, snow, rain, fireflies, leaves, sparkles). Playback starts as soon as the world, the first moment and its lines are ready; later moments arrive while it plays. left thumbstick glides within 3 m, right stick snap-turns; Replay / Pause / Next / Exit buttons float in the world (controller ray + trigger, or hand pinch); the 3D caption shows in VR, the page caption on screens | `web/theatre.js`, `web/viewer.js`, `web/app.js` |

Orchestration (`server/live.py`): one thread per story, a shared pool for the agents; the state is published in
memory the moment anything is ready and the browser polls it every 0.4 s. Without Bedrock everything still runs:
the moments use OpenCV's reading of the page, the page itself goes on the 3D screen, and the browser speaks.

## 2b. The world comes alive (upgrades while the story plays)

The story starts on painted moments at ~8 s. From then on each moment keeps getting more alive, and the browser
swaps each improvement in the moment it arrives (it keeps polling, more slowly, after "ready"):

```
each painted moment ─┬─> actor agent (OpenCV, ~1 s) ── Stability Inpaint paints the plate behind the character
                     ├─> world composer (OpenCV) ───── its own 360° world + wind, sun/moon, lamps, haze
                     ├─> world agent (Stability Inpaint, ~2 min, one at a time) ── that world repainted around it
                     └─> animator (every moment, as soon as it is painted) ─> clip critic (OpenCV) ─> keep / retry / reject

When video is enabled the story **waits for the film**: it starts once its first moment is filmed, on the moving
picture, with a "Filming your story · n of N scenes" panel meanwhile.
```

| Step | What runs | Where |
|---|---|---|
| 9. What moves | The story agent adds, per moment, an `action`: who or what moves, the movement in words, and one of seven paths (fall, walk-left, walk-right, rise, jump, sway, still), derived from the page's own lines. Synonyms ("tumbles", "flies", "dancing") are folded into the paths in code; a movement without a description becomes "still". Nothing is specific to one story | `agents/planner.py` |
| 10. Actor agent | YOLOX (`cv.dnn`) on the moment and on half-overlapping square tiles (merged by NMS), Depth Anything V2 pop-outs for drawn characters, refined by depth and edges (`scene25d.foreground_layers`). The story's own words choose among them: a subject it describes as a person or animal is matched to that detector class; a subject it doesn't ("a large egg") is the largest standalone object or depth shape, people last. A judge refuses cut-outs that are too small or large, ragged (solidity < 0.35), mostly on the border, or the ground. GrabCut, seeded by the mask and the detector box (with room for hats and crowns, kept only where depth says it is as near as the character), gives a soft alpha matte. The plate (the scene without the character) is painted by Stability AI Inpaint; OpenCV's shift-map inpainting is the fallback | `vision/actors.py`, `server/live.py` (`_actor`) |
| 11. The character moves | The browser puts the cut-out back exactly where it was painted, 0.3 m in front of the curved screen, over the plate, and moves it along its path in time with the spoken lines: a fall sits, wobbles, topples about its feet and drops if there is room; walks cross the screen with a bob; jumps squash on landing. Pause freezes it, replay restarts it. OpenCV's YuNet face detector reads which way the cut-out looks (the nose's offset from between the eyes), so a character walking the other way is mirrored instead of walking backwards (no face found: no guess). The screen's depth relief is flattened where the character was painted, so no bump of its shape is left once it moves | `web/theatre.js` (`pose`, `#attachActor`) |
| 12. A world per moment | Each moment is set into the story's world at the exact angles the curved screen covers (`world.place` + `composite`), with its own depth and motion maps, so the 360° world around the viewer changes with the story; the world agent (step 4b) then repaints each moment's world in turn | `server/live.py` (`_moment_world`) |
| 13. World life | From every world: **wind** = orientation of the sky's brush strokes (structure tensor of Sobel gradients, coherence-weighted, doubled-angle mean) and the story's wind → the sky drifts along its own strokes; **sun/moon** = brightest compact blob in the sky (≤0.6% of the image) → a glow and slow rays; **lamps** at night = small warm bright blobs below the sky (HSV + connected components) → flickering glows; **haze** = the sky's colour at the horizon → a mist band. The story agent's time of day and weather add stars, rain, snow or storm flashes. **Sound:** wind, rain and running water synthesized in the browser from noise (Web Audio, no files), the wind's strength from the analysis, rain from the weather, water when OpenCV found water; quiet under the voices | `vision/life.py`, `web/theatre.js` (`worldLife`) |
| 14. Animator | Every painted moment (now painted at 16:9) is queued for filming the moment its painting exists, from that painting (and, when the page has its own picture, the first moment is that picture extended to 16:9 by Stability Inpaint). Backends: **Luma AI Ray2** (us-west-2, 5 s, aspect matched to the painting, one job at a time on this account, 2–5 min, ~$3.75; sold through AWS Marketplace) and **Amazon Nova Reel** (6 s at exactly 1280×720 from a start frame OpenCV fits without stretching; up to 10 jobs at once; **retired in us-east-1 in 2026**, see Evaluation). One priority queue, `VIDEO_CONCURRENCY` workers that never exceed the account's quota, a daily cap; a retired model or a missing permission switches video off once instead of failing every moment | `agents/animator.py`, `server/live.py` (`_shot`, `_clip`) |
| 15. Clip critic | OpenCV watches every clip before anyone sees it: **fidelity** (ORB + RANSAC inliers between the painting and frame 0, against the painting's own re-encoded copy; SSIM only for featureless paintings, since SSIM calls flat pictures alike), **lettering** (PP-OCRv3), **new figures** (YOLOX counts), **motion** (Farneback flow of the moving pixels around the subject: net direction must match the path; a moving background means the camera moved), **cuts** (frame SSIM and HSV histogram Bhattacharyya), **loop point** (late frame closest to frame 0). A failed clip is retried once with a stronger prompt, then rejected; the moving character stays | `vision/clip.py`, `server/live.py` (`_clip`) |
| 16. The clip in the world | The clip plays on the same curved, depth-displaced screen (a video texture on a mesh sharing the picture's geometry and feathered edges), crossfading over the character; it loops at the critic's loop point or holds its last frame. Its frames are uploaded from the render loop, since a headset's immersive session can throttle the browser's video-frame callbacks | `web/theatre.js` (`#attachClip`, `#syncClip`) |

### How much OpenCV, agent orchestration and active inspection

OpenCV 5 modules in use: `dnn` (4 networks: PP-OCRv3 DB detector, CRNN recognizer, Depth Anything V2, YOLOX),
`features2d` (ORB), `calib3d` (RANSAC homography), `imgproc` (GrabCut, morphology, connected components, distance
transform, Sobel, structure tensor, histograms), `video` (Farneback optical flow), `quality` (SSIM), `photo` and
`xphoto` (inpainting), `ximgproc` (guided filter), `videoio`.

Agents per story: reader, story agent, painters, quick-world agent, actor agent, world composer, world agent,
voices, animator, plus the conductor that streams each result to the viewer.

Active inspection (OpenCV looks, and the system acts on what it sees), 11 loops: camera coaching and auto-capture;
page recognition; best video frame; restoring the story agent's skipped lines; repainting lettered moments;
erasing world figures; repainting flat world tiles; repainting stray figures; the actor judge; the clip critic
(retry or reject); the loop-point decision.

## 3. Other OpenCV pipelines in the app

**Photo → 2.5D scene** (`vision/scene25d.py`): Depth Anything V2 through `cv.dnn`, field of view from vanishing
points (`HoughLinesP` + RANSAC), metric scale from detected objects of known height (YOLOX), foreground layers
by depth pop-out refined with a GrabCut band, `inpaint` behind each layer, meshes → GLB. A 2:1 panorama becomes
a photo sphere.

**Video → real 3D** (`vision/sfm/`): keyframes by Laplacian sharpness and LK optical-flow parallax; SIFT matching
verified by `findEssentialMat(USAC_MAGSAC)`; `recoverPose` + `triangulatePoints`; incremental `solvePnPRansac` +
`solvePnPRefineLM`; bundle adjustment; dense fusion of DNN depth scaled to the triangulated points; floor by RANSAC.
The **Capture Coach** watches each frame (blur, darkness, tracking loss, rotation-only motion via homography
residuals: "slide sideways instead"); the **Reconstruction Verifier** decides done / record more / fall back.

An earlier picture-book mode (2.5D illustration in an AI-painted 360° world, ~2 minutes per page) is still in
the API (`/api/stories`, `vision/world.py`) but no longer in the app: it was too slow to step into.

## 4. AWS

| Service | Role |
|---|---|
| Amazon Bedrock: Nova Lite | The story agent, streamed (page image in, JSON Lines out) |
| Amazon Bedrock: Stability AI Stable Image Core (+ SD3.5 Large) | Paints every moment and the world, in parallel |
| Amazon Polly (neural) | Every line, narrator and character voices, word timings |
| Amazon Bedrock: Stability AI Inpaint | Each moment's full 360° world (5 tiles) and the plate behind each moving character |
| Amazon Bedrock: Luma AI Ray2 (or Nova Reel where served) | A video clip of every moment (async, image-to-video); off unless enabled at deploy time (`cdk deploy -c video=luma`) |
| Amazon Bedrock: Nova Pro | Narration for photo scenes and 3D captures |
| ECS Fargate on Graviton (arm64) | API + web app (8 vCPU: parallel OCR and depth), and the reconstruction worker |
| Step Functions | Video pipeline: reconstruct → narrate, with retries, catch states, logs, X-Ray |
| S3 | Pages, pictures, depth maps, worlds, cut-outs, clips, audio, scenes; raw capture frames expire after 7 days; a us-west-2 bucket for Luma's output (1-day expiry) |
| DynamoDB | Scene index |
| CloudFront | HTTPS (WebXR and camera access require it); edge caching of story and scene files |
| CloudWatch | Container and pipeline logs |
| AWS CDK | The whole stack as code (`infra/`) |

## 5. Evaluation

**Speed** (seconds from upload, real Bedrock and Polly calls, laptop CPU; `server/live.py` timings):

| Page | Lines read aloud | Story planned (first moment) | World painted | First moment painted = **playable** | Everything |
|---|---|---|---|---|---|
| "Twinkle Twinkle Little Star" poster (27 lines) | 27 | 2.7–3.5 | 6.5–6.8 | **6.7–8.2** | 9.9–12.5 |
| Peter Rabbit p.1 | 2 | 2.3–4.6 | 6.0–7.2 | **7.0–8.7** | 7.0–8.7 |
| Peter Rabbit p.2 (dialogue) | 3 | 3.0–3.4 | 6.4–7.2 | **7.2–7.6** | 7.6–9.8 |
| Peter Rabbit p.3 | 1 | 2.3–2.7 | 5.8–6.4 | **6.3–7.1** | 6.4–7.6 |

In the browser (headless Chrome, the real app), the first word of the poster was spoken 9.0 s after the click.

**Reading the page** (`python -m eval.ocr_accuracy`): 1.5% character error, 8.3% word error on three photographed
Peter Rabbit pages (unchanged by this round's changes). On the white-on-blue "Twinkle" poster the reader produced
garbage before the polarity fix ("tndebuindelid.tar How.monderwiatyuance") and now reads all 30 lines with small
slips ("lttle", "travel ler"), in 1.9 s instead of 8.2 s.

**OpenCV checking the story agent** (same runs): on the poster the agent skipped a verse or chorus in every run;
the check restored 3–4 lines each time, in the right place, so all 27 lines were read in order. OCR/agent
agreement 0.83–0.98. Code-level speaker attribution was needed on every dialogue run (the model shifted speakers by
one line); after it, Mrs. Rabbit's words were in her voice and the speech tag in the narrator's.

**Picture checks:** the lettering check measured 0.0000 on 15 clean paintings and 0.013–0.39 on images with text;
figure erasing removed 6 of 6 figures in two world paintings in 0.55–0.95 s.

**Camera (simulated in headless Chrome with a fake camera: a page on a table, blurred for the first frames):** a
page that already had a story was recognized and speaking 2.5 s after pressing the button; a new page was captured
after 2.1 s and its story was speaking at 10.8 s. The photo coach on the Peter Rabbit photos and degraded copies:
blurred, dark and cut-off frames flagged, clean ones passed (glare 0.0 on all real pages). Recognition: 191–549
inliers on pages seen in perspective; none of 7 unrelated paintings matched; two pages with identical words give
no match rather than a wrong one.

**Living paintings:** in a night-sky moment, 13% of the screen moved over 2.5 s while hills, houses and captions
stayed still; daytime scenes: grass, flowers and bushes sway, houses and characters don't.

**The world comes alive** (a Humpty Dumpty page, real Bedrock calls, Oct 25):
- The story was playable in 7.9–9.7 s. The agent marked "Humpty Dumpty had a great fall" as `fall` and the other
  lines as `still`, so exactly one moment got a moving character.
- **Failure caught and fixed:** in the first run the actor agent cut out the king, the largest detected person,
  because YOLOX has no "egg" class, and OpenCV's shift-map plate filled the hole with a collage of nearby faces.
  Using the story's own description ("a large egg-shaped creature with a crown" names no person or animal) made
  it choose the largest standalone object instead: the egg, which YOLOX had labelled "clock". In the second run
  (a different painting) it cut out Humpty with his crown, labelled "keyboard". Stability Inpaint now paints the
  plate: the archway behind Humpty, open to the countryside. Still imperfect: a crown can be clipped, and his
  shoes stayed on the plate.
- In the browser, the world changed with every moment and the character toppled about its feet on the "fall" line.
  It didn't drop, because its feet were already at the bottom of the painting. Exit left no 3D objects or videos
  behind.
- **Clip critic:** 6 synthetic clips (a fall passes and fails as "rise"; a still clip fails "nothing happens";
  lettering, a foreign first frame and a jump cut are rejected; a sway gets a loop point). On a test clip made
  from the real painting it measured 605 matching features against the painting (1,124 for its own copy) and
  verified the fall.
- **The video model (Oct 26), failure found:** the plan moved video to Amazon Nova Reel because the account reported
  it AUTHORIZED with quotas of 10 and 3 parallel jobs. The first real call to both versions answered
  `ResourceNotFoundException: This model version has reached the end of its life`. The only active video model is
  Luma Ray2, sold through AWS Marketplace (the account owner's first call subscribes the account). The animator now
  detects a retired model and switches video off once (tested), and the story then starts on the paintings.
- **Real Luma Ray2 clips (Oct 26; small sample, honest numbers):**
  - *Humpty Dumpty, the fall:* made in 75 s (AWS quotes 2–5 min). The egg crumples, topples, lands and shouts: the
    character does act the line. OpenCV rejected it: at ~1.5 s Luma cut to a different place. Its other scores were
    good (860 matching features with the painting against 1,143 for the painting's own copy; subject moving 0.31
    frame heights per second; the fall seen by optical flow; steady background). Fix: the prompt now asks for one
    continuous shot with a locked-off camera.
  - *Cinderella (5 moments, made on the Quest):* 4 of 5 prompts were refused by Luma's filter ("ip prompt moderation
    failed"): they named a famous character. Fix: a refused prompt is retried once with names replaced by the
    character's description. The 5th moment, a crowd on a stage, came back twice as a camera zoom with nobody
    acting; OpenCV rejected both (a cut, and more figures counted as the zoom enlarged them). Fixes: the story agent
    now frames each moment close on one or two characters, and extra figures reject a clip only when the camera
    held still (a zoom is reported as a camera move). The same story showed the agent labelling "rushes out of the
    ball" and "waves her wand" as still, and the video prompt then dropping the described action; both fixed.
  - *Timmy and Bella, The Golden Key (Quest):* the story agent's reply was thrown away and the app fell back to
    OpenCV's raw reading (lines cut at printed line breaks, no speakers, no movement). Cause: in 3 of 4 calls Nova
    Lite nested every moment inside its header. Fixed; 4 of 4 real calls then gave 5 moments with whole sentences,
    and quotes now go to the right character even without quote marks.
  - *The Hare and the Tortoise (Quest, after the fixes above): the first story with every moment filmed.* 6 of 6
    moments became video (7 clips made, 1 retry); first clip after 76 s, all clips after 523 s (about 9 minutes, one
    at a time). Every accepted clip kept the painting (825–912 matching features), with no cuts and a steady camera
    (background motion ≤ 0.002 frame heights per second). Three clips show clear acting: the hare wakes in the
    moonlight and raises a paw; the tortoise walks in beside the surprised hare; the tortoise strides toward the
    windmill. Three show only subtle motion. Failures we found in it: the opening picture showed three tortoises and
    no hare, and the moral's picture two hares, because characters named only in the action were never described to
    the painter (fixed: everyone a moment names is painted from their fixed description); the hare fades out of the
    sleeping scene mid-clip, a drift the critic did not catch; and one take was rejected only for walking left
    instead of right (fixed: a walk in either direction is the action).
  - *Hansel and Gretel (a page from a published book, Quest, built-in settings):* the story agent is asked to copy
    the page's words, and Nova Lite's own content filter cut its reply ("The generated text has been blocked by our
    content filters") on 2 of 3 calls. The app fell back to OpenCV's raw reading: credits read aloud, misread words
    ("Hanse and G retel", "ived"), no cast, no voices, no movement. The first moment started from the page's own
    picture, but the crop held the printed paragraph (lettering 0.072 against 0.004 allowed) and both takes were
    rejected; one accepted clip had no motion at all; the reader waited 14.5 minutes. Fixes: (1) a reply cut by the
    filter is detected and the moments that arrived complete are kept; (2) if nothing usable arrived, a second
    "lines mode" shows the model OpenCV's numbered lines and asks only for line numbers, speakers, what to show,
    lines to skip (credits, titles) and short fixes for misread words, so the page is never copied (on the real page
    it gave the title, Hansel and Gretel as the cast, credits skipped, "Hanse" → "Hansel", "ived" → "lived"; it was
    itself blocked on 1 of 3 calls, so it gets a second try; fixes that would change a common word everywhere are
    refused, after the model proposed "and" → "land"); (3) the raw fallback leaves out titles and credits; (4) a
    page picture containing printing is never used; (5) moments with nothing described are not filmed (a living
    painting instead), which roughly halves the wait on pages like this; (6) a clip in which nothing moves is
    rejected.
  - Accepted clips: **0 of 3** in the first three stories, then **6 of 7** in the tortoise story. The thresholds were
    not loosened; the fixes target the causes. They are still calibrated on few real clips.
- **The viewer waiting for the film** (headless Chrome, stand-in clips made from the story's own paintings, their
  arrival simulated 8–14 s after opening): the story stayed in its "Filming your story · n of 4 scenes" panel
  until the first clip existed, then started directly on video (first playing frame: a video texture, playing);
  every one of the 4 moments played as video with its time advancing; the 360° world changed with each moment;
  replay restarted moment 1's clip within 0.4 s; pause froze the clip and the moment clock; exit left no 3D objects
  or video sources and no console errors. The same run exposed an older bug: the VR caption panel also showed on
  ordinary screens whenever a word was highlighted (fixed).
- **Sun detector:** at first it accepted a bright overcast patch about 40° wide as the sun; the blob is now limited
  to 0.6% of the image.

**Review of the last three stories (Oct 26, evening), and what changed.**
- **The paintings never matched the page.** In 10 of 11 stories the style was the prompt's own example text,
  copied by the model. A cartoon boy became a realistic bearded man. The prompts now ask the agent to describe
  what it sees in the page's picture, with no example to copy, and to draw each character as the page does.
  Re-run on three real pages, each got its own description, and the king and minister got different looks.
- **A worksheet played OpenCV's raw reading**, including shreds such as "a a a a the han t". The story agent had
  failed with something other than the content filter, and only filter blocks used to get lines mode. Any failure
  now gets lines mode (2 tries), and lines that are mostly one- and two-letter fragments are no longer read aloud.
- **A clip that turned into another scene was accepted.** Fidelity compared the painting with frame 0 only, which
  the video model copies from the painting, so it always passed (815 matches). The critic now also counts ORB +
  RANSAC matches between the painting and the clip's middle and last frames. Calibration on all 57 real clip files on
  disk: clips that morphed into another scene or were filled by something kept 4–14 matches; the weakest clip that
  still looked right kept 58. Threshold: 40, the same minimum the first-frame check uses.
- **Two frozen takes were paid for** on one quiet moment. A frozen take of a "still" moment is no longer retried,
  because the second take gets the same prompt.

**3D reconstruction** (`python -m eval.sfm_accuracy`, rendered room with exact ground truth): 24/24 keyframes
placed, 0.145 px median reprojection error, camera path error 0.05 cm mean, 354k dense points (99.2% agreed).

**Tests** (`uv run pytest`, 108): the streamed-object splitter (chunks, nesting, malformed objects); the reading
check (skipped lines found, slips fixed, logos/links ignored); the quota bucket; the world seam; the lettering
check; light-on-dark posters; the live API end to end without models; the story queue cap; actions folded into
paths; the actor agent and its judge; world life (wind, sun, lamps, haze); the clip critic; the animator's
queue (one at a time for Luma, never above `VIDEO_CONCURRENCY` in parallel); the Nova Reel request (1280×720 start
frame, 6 s, ≤ 512-character prompt naming the subject and the motion); a retired model switching video off; every
painted moment of a 4-moment story queued for filming at paint time and ending ready; the first moment starting
from the page's own picture; the conductor with a fake animator (accept, retry then reject); Range requests; every
older pipeline.

## 6. Limitations (observed, not hypothetical)

- **Video is slow, serial and often rejected:** Luma Ray2 is the only video model this account can call (Nova Reel
  is retired); it films one clip at a time (75 s measured, ~$3.75 each), so a 4-moment story takes several minutes
  to film, and in our small sample Luma often cut or zoomed instead of animating, which OpenCV rejects. A rejected
  moment stays a living painting with an OpenCV cut-out character.
- **Characters act, but there is no lip sync:** in Luma's clips the characters move, gesture and open their mouths
  ("acts this out while speaking"), and each line is voiced by Amazon Polly with a voice per character, but the
  mouths are not matched to the words. Lip-sync models exist (audio-driven talking-head models and hosted video
  products), but none is on Amazon Bedrock or callable from this account, and they are trained on photographed
  faces, not painted storybook characters. An OpenCV mouth puppet driven by Polly's viseme timings was considered
  and left out for time.
- **Famous characters:** Luma refuses prompts naming well-known characters; the retry describes them instead, which
  may still be refused.
- **Published books:** Bedrock's content filter often refuses to reproduce a published page. Lines mode avoids
  copying it, but its words then come from OpenCV's reading, so misreadings the model did not list remain, and the
  filter can still block (then the cleaned OpenCV reading is used).
- **Lesson:** two plans in a row treated a video model as available from the account's listings; only a real call
  showed the truth. Video was first an optional upgrade (the result looked like stills in 360°), then the product,
  but the product depends on a model the account can actually call.
- **One mover per moment:** the actor agent moves the one character the line is about; everyone else stays painted.
- **No 360° video:** the world around the viewer is a painted world made alive by OpenCV, not generated video.
- **Choosing the character can fail:** the detector names only 80 kinds of thing, so subjects it can't name are
  chosen by size and depth; when several things stand out, the wrong one can move.
- **Characters change between moments:** each moment is painted from a fixed text description of the cast, so
  the same rabbit can look a little different from picture to picture.
- **Image quota:** the account allows 9 Stable Image Core pictures a minute, so a page gets 1 world + at most 6
  moments, and a second story started within the same minute waits for quota (or uses SD3.5 Large, 2 a minute).
- **The story agent's JSON is sometimes malformed** (about 1 run in 5 on quote-heavy pages): that moment is lost
  and OpenCV's check restores its lines from the OCR, with OCR slips and printed line breaks.
- **Moments can repeat** on poems with many similar verses (several "child looking at the star" pictures).
- **World figures:** the painter adds small figures to "empty" landscapes; YOLOX + shift-map erases the ones it
  detects, leaving faint patch edges; undetected silhouettes stay.
- **2.5D only:** the curved screen has relief from a single picture's depth, so it holds up near the viewing spot.
- **Headset cameras:** the Quest browser's camera access is experimental and not on every headset; where it isn't
  available, a phone is the camera and the story opens on the headset by itself.
- **Recognition leans on text:** ORB features sit mostly on the lettering, so pages with the same words can't be
  told apart (the app then makes a new story instead of guessing).
- **Water by colour** needs the story to mention water; still water that is green or brown isn't found.

## 7. Responsible use

- **Faithful reading:** what is read aloud is the page's own words; the story agent may fix misreadings but not add
  or reword, OpenCV restores skipped lines and removes repeats, and the page panel shows exactly which printed line
  is being read.
- **Honest labelling:** the app shows which model read, planned, painted and voiced each story, and timings for
  every step. Without AI, it says so and shows the page itself.
- **Spending guard on a public endpoint:** at most 3 stories built at once (HTTP 429 otherwise), pages downscaled on
  arrival, per-minute quota respected, and a daily cap on paid image calls (`STABILITY_DAILY_LIMIT`, default 400);
  a story left "running" by a restart is reported as failed rather than polled forever. Video is off unless chosen
  at deploy time, and then capped per day (`VIDEO_DAILY_LIMIT`: 16 clips for Luma).
- **Content rights:** the pages used for the evaluation are from a public-domain book (1902) and are not shipped
  with the app. Users must have the right to the pages they upload; nothing is shared publicly by default.
- **Data:** uploads stay in the deployer's AWS account (S3, encrypted, public access blocked).

## 8. Run it

```bash
uv sync && uv run python -m vision.models.fetch
uv run uvicorn server.app:app --port 8000      # http://localhost:8000
scripts/lan-https.sh                            # Meta Quest on the same Wi-Fi (self-signed HTTPS)
uv run pytest -q
cd infra && npm install && CDK_DOCKER=finch npx cdk deploy --all   # add -c video=luma for video (see README)
```
