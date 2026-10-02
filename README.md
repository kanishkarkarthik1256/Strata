# DroneRecon

DroneRecon turns drone video and associated telemetry into inspectable 3D
reconstructions. The web interface submits and tracks jobs, while the FastAPI
backend extracts useful frames, estimates camera poses, builds sparse and dense
geometry, and prepares reconstruction artifacts for the 3D viewer.

The frontend is React, TypeScript, Vite, and Three.js. The backend is Python
3.11+, FastAPI, PyTorch, Open3D, and COLMAP. Runtime state is stored locally;
there is no external database service required for development.

## Requirements

- Git
- Python 3.11 or newer
- Node.js 20 or newer and npm
- COLMAP and FFmpeg on the `PATH` for local backend runs
- Docker Compose for the containerized setup
- An NVIDIA GPU and NVIDIA Container Toolkit for GPU-enabled Docker Compose

On macOS, install the system tools with Homebrew:

```bash
brew install colmap ffmpeg
```

The backend can run on CPU by setting `AI_DEVICE=cpu`. GPU inference requires a
compatible PyTorch/CUDA installation and model checkpoints. Model weights and
input datasets are not included in this repository.

## Run With Docker Compose

From the repository root, create the environment file and start both services:

```bash
cp .env.example .env
docker compose up --build
```

Open the application at <http://localhost:5173>. The backend API and interactive
API documentation are at <http://localhost:8000> and
<http://localhost:8000/api/docs>; readiness is reported at
<http://localhost:8000/api/ready>.

The Compose configuration requests an NVIDIA GPU. On a machine without the
NVIDIA Container Toolkit, use the local development setup below or adjust the
Compose GPU reservation for that host. On macOS, the local CPU setup is the
portable option. Stop the services with `Ctrl+C`, or run `docker compose down`
in another terminal. Compose's named data volumes persist after stopping.

## Run Locally

### Backend

Install COLMAP and FFmpeg first. Then, from the repository root:

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Start the API from the `backend/` directory:

```bash
AI_DEVICE=cpu uvicorn app.main:create_app --factory --reload --host 127.0.0.1 --port 8000
```

Use `AI_DEVICE=cuda` only when CUDA and the corresponding PyTorch build are
available. Backend settings can be customized with environment variables or a
`.env` file in the backend working directory. The complete settings template is
at `../.env.example`.

### Frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Vite serves the UI at <http://localhost:5173> and proxies `/api` requests to the
backend at `http://127.0.0.1:8000`.

## Use the Application

Start both backend and frontend, open the UI, and provide a drone video through
the available mission workflow. The backend processes the uploaded media,
persists job state, and makes generated artifacts available to the frontend's
viewer and API. Processing time and output quality depend on input coverage,
available compute, and configured model checkpoints.

The project does not ship sample videos, generated reconstructions, precomputed
demo runs, or model weights. Add your own permitted input media; a fresh checkout
will not have precomputed fast-demo runs available.

## Data and Configuration

- `data/storage/<job_id>/` contains uploaded inputs and per-job intermediate
  artifacts when running from the repository root.
- `output/<run_id>/` contains generated run artifacts.
- `models/weights/` is the local destination for model checkpoints.
- `.env.example` documents server, storage, database, AI, COLMAP, and pipeline
  settings. Keep real credentials in an untracked `.env` file.

The default local database is SQLite. Runtime data, model weights, and generated
outputs are intentionally excluded from Git; they can be recreated or supplied
locally as needed.

## Tests and Build

Backend tests (from `backend/`):

```bash
pytest tests -q
```

Frontend checks (from `frontend/`):

```bash
npm test
npm run build
```

## Repository Layout

- `backend/` — FastAPI API, processing pipeline, and backend tests
- `frontend/` — React application and 3D viewer
- `docker/` and `docker-compose.yml` — container build and runtime configuration
- `docs/` — architecture, validation, and operations notes
- `scripts/` — benchmark and diagnostic utilities