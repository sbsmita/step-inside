# Demo video: shot list (≤ 5 minutes)

Record the browser at 1440×900 (the deployed URL), plus Quest footage for the VR part. The app's own clock is the
proof of speed, so keep it in frame.

| # | Time | Shot | What to say |
|---|---|---|---|
| 1 | 0:00–0:15 | A phone photographs a page (the "Twinkle Twinkle" poster, or a Peter Rabbit page) | "Any page of a story. One photo." |
| 2 | 0:15–0:40 | The app: choose the photo, **Bring it to life**. The progress list ticks: reading (OpenCV) ✓, story ✓, world ✓, moments 1 of 6…, voices ✓; the clock reads "Ready in ~8 s" | "OpenCV reads it while a story agent plans it; painters and voices work in parallel." |
| 3 | 0:40–1:40 | It plays: "When the blazing sun is gone" over a sunset, "Then the traveller in the dark" over a night road; the caption lights each word; the page panel highlights the line | "You hear each line and see what it says, word by word." |
| 4 | 1:40–2:10 | "How it was made": the page with OpenCV's line boxes, the moments list, models, timings | "Every step is timed. OpenCV also checks the AI: it put back the verse the story agent skipped." |
| 5 | 2:10–2:40 | A Peter Rabbit dialogue page: Mrs. Rabbit's words in her voice, the narrator's in his | "Quotes go to the character who says them." |
| 6 | 2:40–3:30 | Quest: **Enter VR**, look around the 360° world, the curved picture with depth in front, captions and the page in 3D; A = next line, B = pause, snap turn | "In a headset you're standing in it." |
| 7 | 3:30–4:05 | `docs/architecture.png`; AWS console: Fargate service, Bedrock, CloudWatch logs | "Nova Lite plans, Stable Image Core paints, Polly speaks, all on AWS; the stack is CDK." |
| 8 | 4:05–4:40 | Evaluation and failure cases from the report: the poster before/after the light-on-dark fix; restored lines; malformed JSON caught; image quota | "We measure it and show where it breaks." |
| 9 | 4:40–5:00 | Title card: repo, live URL, "OpenCV 5 + AWS" | — |

Tips: make one story just before recording so the image quota (9 pictures a minute) is fresh; keep the
"Stories" list visible to replay a story without regenerating it.
