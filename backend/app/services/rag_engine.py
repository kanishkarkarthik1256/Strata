"""Local RAG engine over one mission's intelligence artifacts.

Builds a grounded index (``intel/rag_index.json``) from every Phase 7/8
artifact in the workspace — twin objects, damage findings, risk indicators,
infrastructure measurements, change clusters, environment report and mission
recommendations. Each *record* keeps its numeric fields and a human-readable
text; queries resolve either to a structured intent (count / measure /
locate / damage / risk / change / recommend / summary) over the records or,
for anything else, to ranked retrieval of the exact record texts.

Groundedness is enforced structurally: answers quote record ids and are built
only from indexed numbers — there is no generative step, so the engine cannot
invent facts absent from the mission data. When a question's vocabulary does
not match any record, the answer says so explicitly.

Backends:

* ``lexical`` (default, always available) — token-overlap retrieval with
  phrase boost; deterministic and dependency-free.
* ``transformers`` (guarded) — sentence-transformer embeddings stored in
  ``intel/rag_vectors.npy`` (requires the model installed and
  ``INTEL_RAG_BACKEND=transformers``); falls back to lexical with a logged
  reason when the model is unavailable.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register
from app.services.scene_intelligence import _SYNONYMS, CONCEPT_TO_CLASSES

log = get_logger("drone_recon.services.rag_engine")

_RECORD_LIMIT = 200  # objects beyond this are summarized, not individually indexed
_OBJECT_FIELDS = ("uuid", "class", "centroid", "bbox_min", "bbox_max", "height_m",
                  "surface_area_m2", "volume_m3", "confidence", "footprint_area_m2")


def _tokens(text: str) -> Counter:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(words)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def _add(records: list[dict], kind: str, doc_key: str, text: str, fields: dict) -> None:
    tokens = _tokens(text)
    if not tokens:
        return
    records.append({
        "id": f"{doc_key}:{len(records)}",
        "kind": kind, "doc": doc_key, "text": text,
        "fields": {k: v for k, v in fields.items() if v is not None},
        "tokens": dict(tokens),
    })


def build_index(workspace: Path) -> dict:
    """Collect grounded records from the workspace's intel artifacts."""
    records: list[dict] = []
    tw = _read(workspace / "twin" / "twin.json")
    if tw:
        objects = tw.get("objects", [])
        for o in objects[:_RECORD_LIMIT]:
            cls = o.get("class", "unknown")
            centroid = o.get("centroid") or o.get("bbox_min") or []
            text = (f"{cls} object" + (f" id {o.get('uuid')}" if o.get("uuid") else "")
                    + f" at {centroid}" + f", height {o['height_m']:.1f} m"
                    if o.get("height_m") is not None else "")
            _add(records, "object", "twin", text, {f: o.get(f) for f in _OBJECT_FIELDS})
        if len(objects) > _RECORD_LIMIT:
            extra = len(objects) - _RECORD_LIMIT
            _add(records, "object", "twin",
                 f"{extra} additional semantic objects beyond the per-record limit",
                 {"count": extra})

    def _import_json(name: str, doc: str, kind: str, text_fn) -> None:
        data = _read(workspace / "intel" / name)
        if not data:
            return
        text_fn(records, data, doc, kind)

    _import_json("damage_report.json", "damage", "finding",
                 lambda r, d, doc, kind: [_add(r, kind, doc,
                                               f"{f.get('type')} ({f.get('severity')}) — {f.get('rationale', '')}",
                                               f) for f in d.get("findings", [])])
    _import_json("risk_report.json", "risk", "risk", _risk_records)
    _import_json("infrastructure_report.json", "infrastructure", "measurement",
                 _infra_records)
    _import_json("change_report.json", "change", "change", _change_records)
    _import_json("environment_report.json", "environment", "environment",
                 lambda r, d, doc, kind: [
                     _add(r, kind, doc, f"environmental quality {d.get('quality_score')} "
                          f"({d.get('grade')}), dominant condition {d.get('dominant_condition')}",
                          {"quality_score": d.get("quality_score"), "grade": d.get("grade"),
                           "dominant_condition": d.get("dominant_condition")}),
                     _add(r, kind, doc, "visibility " + str(d.get("visibility", {}).get(
                         "estimated_visibility_m")), {"visibility_m": d.get("visibility", {}).get(
                             "estimated_visibility_m")})])
    _import_json("mission_recommendations.json", "recommendations", "recommendation",
                 lambda r, d, doc, kind: [_add(r, kind, doc, rec.get("reason", ""),
                                               {"type": rec.get("type"),
                                                "priority": rec.get("priority")})
                                          for rec in d.get("recommendations", [])])

    gps = _read(workspace / "georef" / "gps_report.json")
    if gps:
        q = (gps.get("gps_quality") or {}).get("gps_score")
        _add(records, "gps", "gps_report",
             f"GPS quality score {q}" if q is not None else "GPS report present, no score",
             {"gps_score": q})
    return {
        "method": "grounded_record_index",
        "backend": _backend(),
        "record_count": len(records),
        "records": records,
    }


def _risk_records(records: list[dict], data: dict, doc: str, kind: str) -> None:
    ind = data.get("overall_risk", {}).get("indicators", [])
    for i in ind:
        _add(records, kind, doc, f"{i.get('name')} risk {i.get('level')} — {i.get('reasoning', '')}",
             {"name": i.get("name"), "level": i.get("level")})
    for z in data.get("risk_zones", []):
        _add(records, kind, doc, f"risk zone ({z.get('cause')}) of {z.get('area_m2')} m² "
             f"at {z.get('centroid')}", z)
    for b in data.get("accessibility", {}).get("blocked_routes", []):
        _add(records, kind, doc, f"blocked access to {b.get('class')} {b.get('uuid')} — "
             f"{b.get('cause')}", b)


def _infra_records(records: list[dict], data: dict, doc: str, kind: str) -> None:
    for section in ("buildings", "trees", "roads", "terrain"):
        if section not in data:
            continue
        _add(records, kind, doc, f"{section}: {data[section]}", {"section": section})


def _change_records(records: list[dict], data: dict, doc: str, kind: str) -> None:
    for c in data.get("added_clusters", []):
        _add(records, kind, doc, f"structure ADDED near baseline — {c.get('volume_m3')} m³ "
             f"at {c.get('centroid')}", {"type": "added", **{k: c[k] for k in ("centroid", "volume_m3") if k in c}})
    for c in data.get("removed_clusters", []):
        _add(records, kind, doc, f"structure REMOVED vs baseline — {c.get('volume_m3')} m³ "
             f"at {c.get('centroid')}", {"type": "removed", **{k: c[k] for k in ("centroid", "volume_m3") if k in c}})


def _backend() -> str:
    if settings.intel.rag_backend == "lexical":
        return "lexical"
    if settings.intel.rag_backend == "transformers":
        try:
            import sentence_transformers  # noqa: F401
            return "transformers"
        except ImportError:
            return "lexical"
    # auto
    try:
        import sentence_transformers  # noqa: F401
        return "transformers"
    except ImportError:
        return "lexical"


def _embed(records: list[dict]) -> tuple[np.ndarray | None, str]:
    """Embed record texts with sentence-transformers; None when unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, "lexical"
    model = SentenceTransformer(settings.intel.rag_model_name)
    vec = model.encode([r["text"] for r in records], normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32), "transformers"


def _lexical_rank(records: list[dict], q_tokens: Counter, top_k: int) -> list[tuple[float, dict]]:
    scored = []
    qnorm = np.sqrt(sum(v * v for v in q_tokens.values())) or 1.0
    for r in records:
        r_tokens = r.get("tokens", {})
        dot = sum(q_tokens.get(t, 0) * c for t, c in r_tokens.items())
        rnorm = np.sqrt(sum(c * c for c in r_tokens.values())) or 1.0
        scored.append((float(dot / (qnorm * rnorm)), r))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def _vector_rank(records: list[dict], vectors: np.ndarray, query: str, top_k: int) -> list[tuple[float, dict]]:
    """Cosine rank over precomputed record embeddings (transformer backend)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return _lexical_rank(records, _tokens(query), top_k)
    qv = SentenceTransformer(settings.intel.rag_model_name).encode([query], normalize_embeddings=True)
    sims = vectors @ qv[0]
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), records[i]) for i in order]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _classes_for(concept: str) -> list[str]:
    return CONCEPT_TO_CLASSES.get(concept, [])


def _class_objects(records: list[dict], classes: list[str]) -> list[dict]:
    return [r for r in records if r["kind"] == "object"
            and r.get("fields", {}).get("class") in classes]


def answer_question(index: dict, query: str, vectors: np.ndarray | None = None) -> dict:
    """Resolve a natural-language question against the indexed mission data."""
    records = index.get("records", [])
    q = (query or "").lower().strip()
    out: dict = {"query": query, "intent": "", "answer": "", "results": [],
                 "sources": [], "grounded": True,
                 "note": "answers are built only from indexed mission records"}
    if not q:
        out.update(intent="clarify", answer="ask about objects, damage, risk, change or recommendations")
        return out

    concept = _concept_in(q)
    classes = _classes_for(concept) if concept else []
    matches = _class_objects(records, classes) if classes else []

    # --- intent dispatch (numbers are taken from records, never invented) ---
    if re.search(r"\b(how many|count|number of|total)\b", q):
        if re.search(r"\b(damaged|destroyed|collapsed|flooded|affected)\b", q):
            hits = [r for r in records if r["kind"] == "finding"]
            n = len(hits)
            out.update(intent="count_damage",
                       answer=f"{n} damage finding(s) match — see results",
                       results=[{"kind": "damage", **r["fields"]} for r in hits[:20]],
                       sources=[r["id"] for r in hits])
        elif concept:
            out.update(intent="count",
                       answer=f"{len(matches)} {concept} object(s) indexed",
                       results=[{"count": len(matches), "class": concept}],
                       sources=[r["id"] for r in matches[:50]])
        else:
            by_kind = Counter(r["kind"] for r in records)
            out.update(intent="count",
                       answer="index contains " + ", ".join(f"{k}: {c}" for k, c in by_kind.items()),
                       results=[{"by_kind": dict(by_kind)}], sources=[])
        return out

    if concept and re.search(r"\b(measure|height|area|volume|size|how tall|how big|dimensions)\b", q):
        metric = ("volume_m3" if "volume" in q else "height_m"
                  if "height" in q or "tall" in q else "surface_area_m2" if "area" in q or "size" in q else "height_m")
        rows = [{"uuid": r["fields"].get("uuid"), "class": r["fields"].get("class"),
                 metric: r["fields"].get(metric)} for r in matches if r["fields"].get(metric) is not None]
        total = sum(float(r[metric]) for r in rows if r[metric] is not None)
        out.update(intent="measure",
                   answer=f"{len(rows)} {concept} object(s) with {metric}; total ≈ {total:.1f}",
                   results=rows[:20], sources=[r["id"] for r in matches])
        return out

    if re.search(r"\b(damage|destroyed|collapse|collapsed|flood|flooded|landslide|debris|rubble|blocked|blockage)\b", q):
        hits = [r for r in records if r["kind"] in ("finding", "risk")]
        if not hits:
            out.update(intent="damage", answer="no damage or risk records exist for this mission",
                       results=[])
        else:
            out.update(intent="damage",
                       answer=f"{len(hits)} damage/risk record(s) — see results",
                       results=[{"kind": r["kind"], "doc": r["doc"], "text": r["text"]}
                                for r in hits[:20]], sources=[r["id"] for r in hits])
        return out

    if concept and re.search(r"\b(where|show|find|list|locate|highlight|which)\b", q):
        out.update(intent="locate",
                   answer=f"{len(matches)} {concept} object(s) located",
                   results=[{k: r["fields"].get(k) for k in ("uuid", "class", "centroid",
                                                             "bbox_min", "bbox_max", "confidence")}
                            for r in matches[:20]], sources=[r["id"] for r in matches])
        return out

    if re.search(r"\b(risk|danger|unsafe|emergency|access|evacuat)\b", q):
        risk = [r for r in records if r["kind"] == "risk"]
        out.update(intent="risk",
                   answer=f"{len(risk)} risk indicator(s) indexed" if risk else "no risk indicators indexed",
                   results=[{"text": r["text"]} for r in risk[:20]], sources=[r["id"] for r in risk])
        return out

    if re.search(r"\b(change|compare|added|removed|new|difference|different|progress|vs)\b", q):
        ch = [r for r in records if r["kind"] == "change"]
        out.update(intent="change",
                   answer=f"{len(ch)} change record(s)" if ch else "no multi-mission change records (no baseline)",
                   results=[{"text": r["text"]} for r in ch[:20]], sources=[r["id"] for r in ch])
        return out

    if re.search(r"\b(recommend|re-fly|refly|improve|better|rescan|optimiz)\b", q):
        rec = [r for r in records if r["kind"] == "recommendation"]
        out.update(intent="recommendation",
                   answer=f"{len(rec)} recommendation(s)" if rec else "no recommendations indexed",
                   results=[{"text": r["text"], **r["fields"]} for r in rec[:20]],
                   sources=[r["id"] for r in rec])
        return out

    if re.search(r"\b(summary|overview|what is there|report)\b", q) or (concept and not matches and not classes):
        counts = Counter(r["kind"] for r in records)
        out.update(intent="summary",
                   answer="index summary: " + (", ".join(f"{k}: {c}" for k, c in counts.items())
                                               or "empty index"), results=[dict(counts)], sources=[])
        return out

    # --- free-form: ranked retrieval of the exact record texts ---
    if records:
        if vectors is not None:
            top = _vector_rank(records, vectors, q, settings.intel.rag_top_k)
        else:
            top = _lexical_rank(records, _tokens(q), settings.intel.rag_top_k)
    else:
        top = []
    if not top or top[0][0] <= 0.0:
        out.update(intent="no_match",
                   answer="nothing in the indexed mission data answers this question — "
                          "rephrase around objects, damage, risk, change or recommendations",
                   results=[], grounded=True)
        return out
    out.update(intent="retrieval",
               answer=f"closest {len(top)} indexed record(s) below — read them for the answer",
               results=[{"doc": r["doc"], "kind": r["kind"], "text": r["text"],
                         "score": round(float(s), 3)} for s, r in top],
               sources=[r["id"] for _, r in top])
    return out


def _concept_in(q: str) -> str | None:
    """Canonical concept from the question, or None when none matches."""
    for word in q.split():
        concept = _SYNONYMS.get(word)
        if concept:
            return concept
    for concept, syns in (("building", ("building", "buildings", "house", "homes", "structure", "structures")),
                          ("water", ("water", "flood", "pond", "lake", "river")),
                          ("road", ("road", "roads", "street", "path"))):
        if any(s in q for s in syns):
            return concept
    return None


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class RagIndexStage(PipelineStage):
    name = "rag_index"
    description = "Local grounded retrieval index over mission intelligence artifacts"
    artifact_rel = "intel/rag_index.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "intel").is_dir():
            raise StageNotApplicable("no intel artifacts — run the intelligence chain first")

    def execute(self) -> None:
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        index = build_index(self.workspace)
        (intel_dir / "rag_index.json").write_text(json.dumps(index, indent=2))
        self._outputs = [{"kind": "data", "name": "rag_index", "path": str(intel_dir / "rag_index.json")}]
        vectors = None
        if index["backend"] == "transformers":
            vectors, _ = _embed(index["records"])
            if vectors is not None:
                np.save(intel_dir / "rag_vectors.npy", vectors)
                self._outputs.append({"kind": "data", "name": "rag_vectors",
                                      "path": str(intel_dir / "rag_vectors.npy")})
        self._count = index["record_count"]
        self._detail = {"backend": index["backend"], "records": self._count}
        self.progress(1.0, {"records": self._count})


def query_workspace(workspace: Path, query: str) -> dict:
    """Load this mission's index (+ vectors) and answer one question."""
    index = _read(workspace / "intel" / "rag_index.json")
    if index is None:
        return {"query": query, "intent": "no_index", "answer": "no rag index — "
                "run rag_index on this mission first", "results": [], "sources": []}
    vectors = None
    vpath = workspace / "intel" / "rag_vectors.npy"
    if vpath.exists():
        try:
            vectors = np.load(vpath)
        except (OSError, ValueError):
            vectors = None
    return answer_question(index, query, vectors=vectors)


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
