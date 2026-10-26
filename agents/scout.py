"""Scout: active perception. Decides whether the frame is good enough, or what the user should do to get a better one."""
from vision.quality import MOTION_MAX

ADVICE = {
    "moving_fast": "Slow down a little, I'm still looking.",
    "too_dark": "It's too dark for me to see. Could you turn on a light?",
    "blurry": "Hold the phone still for a moment.",
    "glare": "There's glare. Tilt the phone down a little.",
    "lost": "I don't recognize this view. Turn slowly toward a part of the room you scanned before.",
    "partial": "Turn a little to show me more of the room.",
}


def advise(scan: dict, motion: float | None = None) -> dict:
    issues = scan["quality"]["issues"]
    if motion is not None and motion > MOTION_MAX:
        reason = "moving_fast"
    elif "too_dark" in issues:
        reason = "too_dark"
    elif "blurry" in issues:
        reason = "blurry"
    elif scan["status"] == "lost":
        reason = "glare" if "glare" in issues else "lost"
    elif scan.get("coverage", 1.0) < 0.35:
        reason = "partial"
    else:
        return {"action": "analyze", "reason": "frame_ok"}
    return {"action": "rescan", "reason": reason, "say": ADVICE[reason]}
