"""The sparse stage must publish its substage milestones.

`_stage_sparse` is the longest stage in the pipeline, and for a long time it
reported nothing until it finished: its progress callback referenced a
`stage_publish` that was never a parameter of the function, so every milestone
raised `NameError`. `sparse_reconstruction` deliberately swallows a failing
progress callback ("progress must never break reconstruction"), so the run
completed and the only symptom was that the longest stage sat at 0% — the
hardest possible symptom to attribute, and exactly the one a user reported.

This pins the contract at the seam that was broken: whatever milestones the
sparse pipeline reports, the stage publisher must receive them.
"""

from __future__ import annotations

from pathlib import Path

from app.services import pipeline_orchestrator as orch


def test_sparse_forwards_its_substage_milestones(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "run"
    workspace.mkdir()
    (workspace / "frames").mkdir()

    published: list[tuple] = []

    def fake_sparse(selected_dir, ws, **kwargs):
        # The real stage reports milestones through this callback; a callback
        # that raises is the defect this test exists for.
        kwargs["progress"]("features", 0.25)
        kwargs["progress"]("mapping", 0.75)
        (Path(ws) / "poses.json").write_text("{}")
        return {"reconstruction": {"num_cameras": 7}, "localization": {}}

    monkeypatch.setattr(orch, "run_sparse_reconstruction", fake_sparse)

    state = orch.StageState("sparse")
    orch._stage_sparse(
        "job1",
        workspace,
        state,
        lambda name, frac, detail=None: published.append((name, frac, detail)),
    )

    assert published == [
        ("sparse", 0.25, {"substage": "features"}),
        ("sparse", 0.75, {"substage": "mapping"}),
    ], "sparse milestones never reached the stage publisher"
    assert state.count == 7
