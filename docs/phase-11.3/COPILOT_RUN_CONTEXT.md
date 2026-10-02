# STRATA Phase 11.3 — Copilot Run Context & Grounding

## Overview
This document specifies how the AI Copilot queries real run artifacts to deliver zero-hallucination, grounded answers.

## Grounded Q&A Architecture

```
[ User Query ] (e.g. "How many points were reconstructed?")
       |
       v
[ Copilot UI Panel ]
       |
       | POST /api/runs/<RUN_ID>/ask {"query": "How many points were reconstructed?"}
       v
[ answer_run_question() service ]
       |
       +--> 1. Read manifest.json from workspace
       +--> 2. Parse binary PLY header for exact vertex count (zero guessing)
       +--> 3. Read poses.json for camera frame & GPS count
       +--> 4. Format grounded response with metric sources
       v
[ JSON Answer Response ]
       |
       v
[ Rendered in Copilot Panel with Grounded Badge ]
```

## Contextual Prompts for Phase 11.3
The Copilot panel automatically offers run-grounded quick prompts when inspecting a live or demo mission:
- `How many points were reconstructed in this mission?`
- `What is the reprojection error of this sparse model?`
- `How many GPS fixes were extracted from the drone video?`
- `Which pipeline stages passed and did any fallback engines execute?`
- `Is metric scale accuracy validated for this site model?`

## Zero-Hallucination Enforcements
- Point counts come strictly from header parsing of `dense/dense_model.ply` and `sparse/sparse_model.ply`.
- Camera counts come from `poses.json` frame array size.
- If a metric is absent (e.g. no GPS in EXIF), the Copilot explicitly states: `"No GPS data is recorded for this run"`, rather than inventing coordinates.
