from app.routes import pipeline


def test_live_status_becomes_completed_after_last_stage(monkeypatch):
    events = [
        {"type": "stage:frames", "timestamp": 1, "payload": {"status": "completed", "progress": 1}},
        {"type": "stage:sparse", "timestamp": 2, "payload": {"status": "completed", "progress": 1}},
    ]
    monkeypatch.setattr(pipeline.engine, "replay", lambda _job_id: events)

    status = pipeline._live_status("run")

    assert status is not None
    assert status["status"] == "completed"


def test_live_status_preserves_running_stage(monkeypatch):
    events = [
        {"type": "stage:frames", "timestamp": 1, "payload": {"status": "completed", "progress": 1}},
        {"type": "stage:sparse", "timestamp": 2, "payload": {"status": "running", "progress": 0.4}},
    ]
    monkeypatch.setattr(pipeline.engine, "replay", lambda _job_id: events)

    assert pipeline._live_status("run")["status"] == "running"
