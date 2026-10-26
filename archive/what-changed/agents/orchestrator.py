"""Scout -> Judge -> Narrator -> Escalator, with a trace of every agent decision."""
import time

from . import escalator, judge, narrator, scout


def analyze(home: str, scan: dict, motion: float | None = None, household_notes: list[str] | None = None) -> dict:
    trace = []
    t0 = time.perf_counter()

    advice = scout.advise(scan, motion)
    trace.append({"agent": "scout", **advice, "quality": scan["quality"], "status": scan["status"]})
    if advice["action"] == "rescan":
        return {"action": "rescan", "say": advice["say"], "items": [], "trace": trace}

    verdicts, source = judge.judge(scan, household_notes)
    trace.append({"agent": "judge", "source": source, "verdicts": verdicts})

    say, spoken = narrator.narrate(scan, verdicts)
    trace.append({"agent": "narrator", "say": say})

    by_id = {c["id"]: c for c in scan["changes"]}
    for v in verdicts:
        if v["escalate"] or (v["verdict"] == "hazard" and not v["safe_for_user"]):
            c = by_id[v["id"]]
            result = escalator.notify(home, f"{v['label']} ({v['reason']}) near {scan.get('keyframe', 'home')}, "
                                            f"{c['direction']}.")
            trace.append({"agent": "escalator", "change": v["id"], **result})

    items = [{**by_id[v["id"]], "verdict": v} for v in verdicts if v["verdict"] != "ignore"]
    return {"action": "report", "say": say, "items": items, "spoken": [v["id"] for v in spoken],
            "keyframe": scan.get("keyframe"), "trace": trace,
            "agents_ms": round((time.perf_counter() - t0) * 1000)}
