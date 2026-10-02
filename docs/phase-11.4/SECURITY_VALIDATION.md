# STRATA Phase 11.4 — Security & Path Traversal Validation

## Overview
This document evaluates security protections against malicious upload filenames and path traversal attempts.

## Security Protection Matrix

| Vulnerability Vector | Test Vector | Protection Mechanism | Result |
|----------------------|-------------|----------------------|--------|
| **Filename Traversal** | `../../../test.txt` | `upload_service.safe_filename()` extracts `Path(raw).name` and strips path separators | **BLOCKED** |
| **Windows Traversal** | `C:\Windows\System32\test.mov` | `safe_filename()` normalizes `\` to `/` before extracting stem | **BLOCKED** |
| **Unallowed Extension** | `malicious.exe`, `script.py` | Extension allowlist check against `.mp4`, `.mov`, `.avi`, `.mkv` | **REJECTED (400)** |
| **Artifact Path Traversal** | `/api/runs/r1/artifacts/../../../../etc/passwd` | `run_service.resolve_artifact()` verifies `path.is_relative_to(run_dir)` | **BLOCKED (400)** |
| **Oversized Upload** | > 10 GB file | Streamed chunk size check `written > max_bytes` unlinks partial file | **REJECTED (413)** |

## Automated Test Coverage
Security assertions are enforced automatically in `backend/tests/test_phase11_3.py` and `backend/tests/test_runs.py`.
