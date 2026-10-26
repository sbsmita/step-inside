"""Reconstruction Verifier: judges a 3D result from its OpenCV statistics and decides the next action."""


def verify(meta: dict) -> dict:
    s, d = meta["sfm"], meta["dense"]
    reasons, advice = [], []
    lo, hi = meta["bounds"]["min"], meta["bounds"]["max"]
    width = max(hi[0] - lo[0], hi[2] - lo[2], 1e-6)
    path = meta["camera_path"]
    path_len = sum(((b[0] - a[0]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 for a, b in zip(path, path[1:]))

    if s["registered_frac"] < 0.6:
        reasons.append(f"only {round(100 * s['registered_frac'])}% of views could be placed")
        advice.append("Move more slowly and keep the same area in view.")
    if s["median_reproj_px"] > 2.0:
        reasons.append(f"geometry is loose (median error {s['median_reproj_px']} px)")
    if d.get("fallback") == "sparse" or d.get("consistent_frac", 1.0) < 0.5:
        reasons.append("dense depth didn't agree between views")
        advice.append("Avoid mirrors, windows and plain walls; include furniture and textured surfaces.")
    if path_len < 0.12 * width:
        reasons.append("the camera barely moved relative to the room size")
        advice.append("Sweep further sideways, about a metre or more, for stronger 3D.")
    if not meta["orient"]["floor_found"]:
        advice.append("Tilt down a little so the floor is in view; it fixes level and scale.")

    if s["registered_frac"] < 0.4:
        verdict, action = "failed", "fallback_2_5d"
    elif reasons:
        verdict, action = "partial", "record_more"
    else:
        verdict, action = "good", "done"
    return {"verdict": verdict, "next_action": action, "reasons": reasons, "advice": advice}
