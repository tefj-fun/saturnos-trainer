import csv
import json
import os
import socket
import time
import urllib.request
from urllib.parse import urlparse, unquote
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client
from ultralytics import YOLO

RUNS_TABLE = "training_runs"

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "sops")
SUPABASE_DATASETS_BUCKET = os.getenv("SUPABASE_DATASETS_BUCKET", "datasets")

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


def extract_storage_path(value):
    if not value:
        return None, None
    if value.startswith("storage:"):
        raw = value.split("storage:", 1)[1].lstrip("/")
        if "/" in raw:
            bucket, path = raw.split("/", 1)
            return bucket, path
        return SUPABASE_DATASETS_BUCKET, raw
    if value.startswith("http"):
        parsed = urlparse(value)
        prefixes = [
            "/storage/v1/object/public/",
            "/storage/v1/object/sign/",
            "/storage/v1/object/",
        ]
        for prefix in prefixes:
            if prefix in parsed.path:
                remainder = parsed.path.split(prefix, 1)[1]
                parts = remainder.split("/", 1)
                if len(parts) == 2:
                    return parts[0], unquote(parts[1])
        return None, None
    return None, None


def download_storage_file(supabase, bucket, remote_path, local_path):
    result = supabase.storage.from_(bucket).download(remote_path)
    content = getattr(result, "data", result)
    if isinstance(content, dict) and "data" in content:
        content = content["data"]
    if not isinstance(content, (bytes, bytearray)):
        raise RuntimeError("Unexpected storage download response type.")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(content)
    return local_path


def download_dataset_prefix(supabase, bucket, storage_prefix, local_dir):
    if not storage_prefix:
        return
    storage = supabase.storage.from_(bucket)
    subdirs = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]
    for subdir in subdirs:
        remote_dir = f"{storage_prefix}/{subdir}"
        try:
            items = storage.list(remote_dir) or []
        except Exception:
            items = []
        for item in items:
            name = item.get("name")
            is_dir = item.get("metadata", {}).get("is_dir")
            if not name or is_dir:
                continue
            remote_path = f"{remote_dir}/{name}"
            local_path = os.path.join(local_dir, subdir, name)
            download_storage_file(supabase, bucket, remote_path, local_path)


def resolve_data_yaml(value, run_id, supabase):
    if not value:
        return None
    if os.path.isabs(value):
        return value
    storage_bucket, storage_path = extract_storage_path(value)
    if storage_path:
        bucket = storage_bucket or SUPABASE_DATASETS_BUCKET
        local_dir = os.path.join(DATASET_ROOT, "datasets", f"run_{run_id}")
        filename = os.path.basename(storage_path) or "data.yaml"
        local_path = os.path.join(local_dir, filename)
        yaml_path = download_storage_file(supabase, bucket, storage_path, local_path)
        storage_prefix = os.path.dirname(storage_path)
        download_dataset_prefix(supabase, bucket, storage_prefix, local_dir)
        return yaml_path
    if value.startswith("http"):
        local_dir = os.path.join(DATASET_ROOT, "datasets", f"run_{run_id}")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, "data.yaml")
        urllib.request.urlretrieve(value, local_path)
        return local_path
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
    data_yaml = resolve_data_yaml(run.get("data_yaml") or config.get("dataYaml"), run_id, supabase)
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
    print(f"[{utc_now()}] Trainer service started (worker_id={WORKER_ID})")
    print(f"[{utc_now()}] Polling {RUNS_TABLE} every {POLL_INTERVAL}s")

    while True:
        run = get_next_run(supabase)
        if not run:
            print(f"[{utc_now()}] No queued runs. Sleeping {POLL_INTERVAL}s.")
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
            print(f"[{utc_now()}] Starting run {run['id']}")
            run_training_job(supabase, run)
            print(f"[{utc_now()}] Completed run {run['id']}")
        except Exception as exc:
            print(f"[{utc_now()}] Run {run['id']} failed: {exc}")
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
