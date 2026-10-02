"""Natural-language copilot over a digital-twin scene.

Queries are parsed into structured intents and answered from the twin's
measurements — no external LLM is required (an optional remote endpoint can
be wired via env config, but every intent resolves deterministically here):

* "measure …" / "height|area|volume of <class>" — objects + measurements
* "where is/are / show / highlight <class>" — matching objects
* "objects with confidence below X%" — low-confidence objects
* "summary / overview" — scene statistics
* anything else — a structured list of what this copilot can answer

Answers are plain data (objects, matched counts, suggested overlay layers)
so any UI can render them directly.
"""

from __future__ import annotations

import re

MEASURE_WORDS = ("measure", "measurements", "dimensions", "how tall", "how big",
                 "how large", "how much", "height of", "area of", "volume of",
                 "estimate", "size of", "footprint")


def answer(scene: dict, query: str) -> dict:
    """Resolve a natural-language question against a scene index / twin dict."""
    q = (query or "").strip().lower()
    if not q:
        return _clarify(scene)
    objects = scene.get("objects", [])
    classes = sorted({o.get("class") for o in objects})
    cls = _match_class(q, classes)

    if any(word in q for word in ("confidence below", "low confidence", "confidence <", "below")):
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        threshold = (float(m.group(1)) / 100.0) if m else 0.8
        hits = [o for o in objects if o.get("confidence", 1.0) < threshold]
        return {
            "intent": "low_confidence",
            "answer": f"{len(hits)} object(s) below {threshold * 100:.0f}% confidence",
            "objects": hits,
            "suggestions": ["confidence_overlay"],
        }

    if any(w in q for w in ("summary", "overview", "what is there", "report")):
        return {
            "intent": "summary",
            "answer": (f"{len(objects)} semantic objects — " +
                       ", ".join(f"{c}: {sum(1 for o in objects if o['class'] == c)}"
                                 for c in classes)),
            "objects": objects,
            "suggestions": ["semantic_overlay"],
        }

    if cls is not None and ("measure" in q or any(w in q for w in MEASURE_WORDS)):
        hits = [o for o in objects if o["class"] == cls]
        want_volume = "volume" in q
        return {
            "intent": "measure",
            "class": cls,
            "answer": _measure_answer(hits, cls, want_volume),
            "objects": hits,
            "suggestions": ["measurements_panel", "semantic_overlay"],
        }

    if cls is not None and any(w in q for w in ("where", "show", "find", "highlight",
                                                "list", "locate", "display")):
        hits = [o for o in objects if o["class"] == cls]
        return {
            "intent": "locate",
            "class": cls,
            "answer": f"{len(hits)} {cls} object(s) in the scene",
            "objects": hits,
            "suggestions": [f"highlight:{cls}", "semantic_overlay"],
        }

    return _clarify(scene, query=query)


def _match_class(q: str, classes: list[str]) -> str | None:
    if not classes:
        return None
    for cls in classes:
        if re.search(rf"\b{re.escape(cls)}", q):
            return cls
    return None


def _measure_answer(objects: list[dict], cls: str, want_volume: bool) -> str:
    if not objects:
        return f"no {cls} objects found in this scene"
    if want_volume:
        vol = [o for o in objects if o.get("volume_m3") is not None]
        if not vol:
            closed = [o for o in objects if o.get("volume_closed") is True]
            return (f"{len(objects)} {cls} object(s); none has a closed volume "
                    "envelope, so volume cannot be measured from this mesh "
                    "(open surfaces)") if not closed else \
                f"{len(objects)} {cls} object(s); volume requires closed geometry"
        total = sum(o["volume_m3"] for o in vol)
        return (f"{cls}: {len(vol)} object(s) with volume; total ≈ "
                f"{total:.1f} m³ (largest {max(o['volume_m3'] for o in vol):.1f} m³)")
    total_area = sum(o.get("surface_area_m2", 0) for o in objects)
    return (f"{cls}: {len(objects)} object(s), {total_area:.1f} m² surface, "
            f"max height {max(o.get('height_m', 0) for o in objects):.1f} m")


def _clarify(scene: dict, query: str = "") -> dict:
    return {
        "intent": "clarify",
        "answer": (f"'{query or ''}' is not answerable from this scene. I can "
                   "answer: measure <class>, volume of <class>, where is <class>, "
                   "objects with confidence below X%, and summary."),
        "objects": [],
        "suggestions": [],
    }
