import csv
import json
import os
import socket
import time
from datetime import datetime, timezone

from supabase import create_client
from ultralytics import YOLO

RUNS_TABLE = "training_runs"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "sops")

DATASET_ROOT = os.getenv("DATASET_ROOT", "/mnt/d/datasets")
RUNS_DIR = os.getenv("RUNS_DIR", "/mnt/d/datasets/runs")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))
WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())

MODEL_MAP = {
    "YOLOv8n": "yolov8n-seg.pt",
    "YOLOv8s": "yolov8s-seg.pt",
    "YOLOv8m": "yolov8m-seg.pt",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_data_yaml(value):
    if not value:
        return None
    if os.path.isabs(value):
        return value
    return os.path.join(DATASET_ROOT, value)


def map_base_model(value):
    if not value:
        return MODEL_MAP["YOLOv8s"]
    if value.endswith(".pt"):
        return value
    return MODEL_MAP.get(value, value)


def parse_config(raw_config):
    if not raw_config:
        return {}
    if isinstance(raw_config, dict):
        return raw_config
    try:
        return json.loads(raw_config)
    except json.JSONDecodeError:
        return {}


def parse_results_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[-1]
    metrics = {}

    def pick(*keys):
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except ValueError:
                    return value
        return None

    metrics["precision"] = pick("metrics/precision(M)", "metrics/precision(B)")
    metrics["recall"] = pick("metrics/recall(M)", "metrics/recall(B)")
    metrics["mAP50"] = pick("metrics/mAP50(M)", "metrics/mAP50(B)")
    metrics["mAP50_95"] = pick("metrics/mAP50-95(M)", "metrics/mAP50-95(B)")
    metrics["raw"] = row
    return metrics


def upload_file(supabase, local_path, remote_path):
    with open(local_path, "rb") as handle:
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            remote_path,
            handle,
            {"content-type": "application/octet-stream", "x-upsert": "true"},
        )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{remote_path}"


def upload_artifacts(supabase, run_id, save_dir):
    artifacts = []
    files = [
        os.path.join(save_dir, "weights", "best.pt"),
        os.path.join(save_dir, "weights", "last.pt"),
        os.path.join(save_dir, "results.csv"),
        os.path.join(save_dir, "labels.jpg"),
    ]

    for local_path in files:
        if not os.path.exists(local_path):
            continue
        name = os.path.basename(local_path)
        remote_path = f"training_runs/{run_id}/{name}"
        url = upload_file(supabase, local_path, remote_path)
        artifacts.append({"name": name, "path": remote_path, "url": url})

    return artifacts


def get_next_run(supabase):
    response = (
        supabase.table(RUNS_TABLE)
        .select("*")
        .in_("status", ["queued", "running"])
        .is_("started_at", "null")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )
    data = response.data or []
    return data[0] if data else None


def update_run(supabase, run_id, updates):
    return (
        supabase.table(RUNS_TABLE)
        .update(updates)
        .eq("id", run_id)
        .execute()
    )


def run_training_job(supabase, run):
    run_id = run["id"]
    config = parse_config(run.get("configuration"))
    data_yaml = resolve_data_yaml(run.get("data_yaml") or config.get("dataYaml"))
    if not data_yaml or not os.path.exists(data_yaml):
        update_run(
            supabase,
            run_id,
            {
                "status": "failed",
                "error_message": f"Dataset YAML not found: {data_yaml}",
                "completed_at": utc_now(),
            },
        )
        return

    model_path = map_base_model(run.get("base_model"))
    epochs = int(config.get("epochs", 100))
    batch = int(config.get("batchSize", 16))
    imgsz = int(config.get("imgSize", 640))
    lr0 = float(config.get("learningRate", 0.001))
    optimizer = config.get("optimizer", "Adam")
    device = config.get("device", 0)

    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        optimizer=optimizer,
        device=device,
        project=RUNS_DIR,
        name=f"train_{run_id}",
        exist_ok=True,
    )

    save_dir = str(getattr(results, "save_dir", os.path.join(RUNS_DIR, f"train_{run_id}")))
    metrics = parse_results_csv(os.path.join(save_dir, "results.csv"))
    artifacts = upload_artifacts(supabase, run_id, save_dir)

    trained_model_url = None
    for artifact in artifacts:
        if artifact["name"] == "best.pt":
            trained_model_url = artifact["url"]
            break

    update_run(
        supabase,
        run_id,
        {
            "status": "completed",
            "results": {
                "mAP": metrics.get("mAP50_95") or metrics.get("mAP50"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "artifacts": artifacts,
                "run_dir": save_dir,
                "metrics": metrics.get("raw"),
            },
            "trained_model_url": trained_model_url,
            "completed_at": utc_now(),
            "error_message": None,
        },
    )


def main():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    os.makedirs(RUNS_DIR, exist_ok=True)

    while True:
        run = get_next_run(supabase)
        if not run:
            time.sleep(POLL_INTERVAL)
            continue

        update_run(
            supabase,
            run["id"],
            {
                "status": "running",
                "started_at": utc_now(),
                "worker_id": WORKER_ID,
                "error_message": None,
            },
        )

        try:
            run_training_job(supabase, run)
        except Exception as exc:
            update_run(
                supabase,
                run["id"],
                {
                    "status": "failed",
                    "completed_at": utc_now(),
                    "error_message": str(exc),
                },
            )


if __name__ == "__main__":
    main()
