# STRATA Phase 11.5 — Desktop Logging & Diagnostics Specification

## Overview
This document defines application log locations, structured log formatting, and user system diagnostics in STRATA.

## Log Locations & Partitioning

- **Application Shell Logs**: Saved to `logs/app.log` in user data directory.
- **Backend Engine Logs**: Managed by `logging_config.py` writing structured JSON/text logs to `logs/backend.log`.
- **Pipeline Execution Reports**: Saved to project workspace `data/storage/<PROJECT_ID>/reports/summary_report.json`.

## System Diagnostics UI
Available in Settings under **STRATA Desktop Application & Engine**:
- Version: `1.0.0`
- Engine Status: `READY (Connected)`
- Storage Path: `data/storage`
- Backend Endpoint: `http://127.0.0.1:8000`
- Open Logs Button: Triggers `Open Logs Location` CTA for rapid log discovery.
