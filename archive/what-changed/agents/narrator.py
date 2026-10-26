"""Narrator: turns judged changes into short speech.

The judge supplies only a noun phrase ("the mug", "something"); sentence shape comes from the change type and
every spatial word comes from OpenCV geometry, so the language model can never misplace an object."""

ORDER = {"hazard": 0, "useful": 1}


def _where(c: dict) -> str:
    return f"{c['direction']}, {c['height']}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def sentence(c: dict, v: dict) -> str:
    label, kind = v["label"], c["kind"]
    if v["verdict"] == "hazard":
        on_floor = " on the floor" if c["height"].startswith("low") else ""
        return f"Careful: {label}{on_floor}, {_where(c)}."
    if kind == "moved":
        return f"{_cap(label)} has moved. It's now {_where(c)}; it used to be {c['from_direction']}."
    if kind == "disappeared":
        return f"{_cap(label)} is missing. It was {_where(c)}."
    if kind == "appeared":
        return f"{_cap(label)} is new here, {_where(c)}."
    return f"{_cap(label)} looks different, {_where(c)}."


def narrate(scan: dict, verdicts: list[dict], max_items: int = 3) -> tuple[str, list[dict]]:
    by_id = {c["id"]: c for c in scan["changes"]}
    items = sorted((v for v in verdicts if v["verdict"] != "ignore"), key=lambda v: ORDER[v["verdict"]])
    if not items:
        return "Everything looks the same as usual.", []

    spoken = items[:max_items]
    n = len(items)
    parts = ["I noticed one change." if n == 1 else f"I noticed {n} changes."]
    parts += [sentence(by_id[v["id"]], v) for v in spoken]
    if n > max_items:
        parts.append(f"And {n - max_items} more. Say 'more' to hear them.")
    return " ".join(parts), spoken
