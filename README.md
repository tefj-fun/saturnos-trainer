# saturnos-trainer

GitHub: https://github.com/tefj-fun/saturnos-trainer

This service runs YOLOv8 detection/segmentation training jobs on a Linux GPU server.
It polls Supabase `training_runs` rows, launches training with Ultralytics
YOLO, and uploads artifacts to Supabase Storage.

## Requirements

- Python 3.11+
- CUDA-capable GPU (optional, but recommended)
- Access to local datasets on the training host

## Setup

1) Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment variables (see `.env.example`):

```bash
export SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
export SUPABASE_STORAGE_BUCKET="sops"
export SUPABASE_DATASETS_BUCKET="datasets"
export SUPABASE_ARTIFACTS_BUCKET="training-artifacts"
export DATASET_ROOT="/mnt/d/datasets"
export RUNS_DIR="/mnt/d/datasets/runs"
export POLL_INTERVAL="10"
export HEARTBEAT_INTERVAL="10"
export PROGRESS_INTERVAL="10"
export CANCEL_CHECK_INTERVAL="5"
export WORKER_ID="trainer-1"
export TIMESTAMP_STREAMS="1"
```

3) Run the service:

```bash
python trainer_service.py
```

## Training Configuration

The service reads `training_runs.configuration` (JSON) plus fields on the row.
Supported keys and defaults:

- `data_yaml` (or `dataYaml` in config): dataset YAML path or URL
- `base_model`: `YOLOv8n`, `YOLOv8s`, `YOLOv8m`, or a `.pt` path (default `YOLOv8s`)
- `task`/`modelTask`/`trainingTask`: `detect` or `segment` (auto-inferred if omitted)
- `epochs` (default `100`)
- `batchSize` (default `16`)
- `imgSize` (default `640`)
- `learningRate` (default `0.001`)
- `optimizer` (default `Adam`)
- `device` (default `0`)
- `augmentation`: dict of YOLO augmentation values (e.g., `fliplr`, `mosaic`, `mixup`)

If `data_yaml` is a relative path, it is resolved under `DATASET_ROOT`.

## Dataset YAML

When creating a training run, set **Dataset YAML Path (Server)** to the
absolute path on the training server, for example:

```
/mnt/d/datasets/my_project/data.yaml
```

You can also point to:
- `storage:` URIs (e.g., `storage:datasets/my_project/data.yaml`)
- Supabase Storage URLs (public or signed)
- Direct HTTP(S) URLs

When using `storage:` or Supabase Storage URLs, the service downloads the YAML
and dataset files under `images/` and `labels/` into `DATASET_ROOT/datasets/run_<id>`.

## Logging

The service logs timing checkpoints around dataset downloads, model loading, and
training. Set `TIMESTAMP_STREAMS=0` to disable timestamped stdout/stderr.

## Outputs

The service uploads artifacts to Supabase Storage under:

```
training_runs/<run_id>/
```

It updates `training_runs` with:
- `status`, `started_at`, `completed_at`, `error_message`, `worker_id`
- `trained_model_url`
- `results` (metrics + artifact URLs + run directory)
