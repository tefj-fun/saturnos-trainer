import csv
import json
import os
import socket
import mimetypes
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse, unquote
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

try:
    from supabase import ClientOptions
except Exception:
    ClientOptions = None
from ultralytics import YOLO

RUNS_TABLE = "training_runs"
WORKERS_TABLE = "trainer_workers"

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
if SUPABASE_URL and not SUPABASE_URL.endswith("/"):
    SUPABASE_URL = f"{SUPABASE_URL}/"
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "sops")
SUPABASE_DATASETS_BUCKET = os.getenv("SUPABASE_DATASETS_BUCKET", "datasets")
SUPABASE_ARTIFACTS_BUCKET = os.getenv("SUPABASE_ARTIFACTS_BUCKET", "training-artifacts")

DATASET_ROOT = os.getenv("DATASET_ROOT", "/mnt/d/datasets")
RUNS_DIR = os.getenv("RUNS_DIR", "/mnt/d/datasets/runs")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", "10"))
CANCEL_CHECK_INTERVAL = int(os.getenv("CANCEL_CHECK_INTERVAL", "5"))
WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())

MODEL_MAP_DETECT = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
}

MODEL_MAP_SEGMENT = {
    "YOLOv8n": "yolov8n-seg.pt",
    "YOLOv8s": "yolov8s-seg.pt",
    "YOLOv8m": "yolov8m-seg.pt",
}

RAW_STDOUT = sys.stdout
RAW_STDERR = sys.stderr


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    RAW_STDOUT.write(f"[{utc_now()}] {message}\n")
    RAW_STDOUT.flush()

def log_duration(message, start_time):
    elapsed = time.time() - start_time
    log(f"{message} ({elapsed:.1f}s)")


class TimestampedStream:
    def __init__(self, stream, timestamp_fn):
        self._stream = stream
        self._timestamp_fn = timestamp_fn
        self._buffer = ""
        self._lock = threading.Lock()

    def _write_line(self, line):
        if not line:
            return
        self._stream.write(f"[{self._timestamp_fn()}] {line}")

    def write(self, data):
        if not data:
            return 0
        with self._lock:
            self._buffer += data
            while True:
                newline_index = self._buffer.find("\n")
                carriage_index = self._buffer.find("\r")
                if newline_index == -1 and carriage_index == -1:
                    break
                if newline_index == -1:
                    split_index = carriage_index
                elif carriage_index == -1:
                    split_index = newline_index
                else:
                    split_index = min(newline_index, carriage_index)
                line = self._buffer[:split_index]
                separator = self._buffer[split_index]
                self._buffer = self._buffer[split_index + 1 :]
                self._write_line(line)
                self._stream.write(separator)
        return len(data)

    def flush(self):
        with self._lock:
            if self._buffer:
                self._write_line(self._buffer)
                self._buffer = ""
            self._stream.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self._stream, "errors", None)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_timestamped_streams():
    enabled = os.getenv("TIMESTAMP_STREAMS", "1").lower() not in ("0", "false", "no")
    if not enabled:
        return
    sys.stdout = TimestampedStream(RAW_STDOUT, utc_now)
    sys.stderr = TimestampedStream(RAW_STDERR, utc_now)


install_timestamped_streams()


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
        return 0
    storage = supabase.storage.from_(bucket)
    subdirs = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]
    total_downloaded = 0
    for subdir in subdirs:
        remote_dir = f"{storage_prefix}/{subdir}"
        log(f"Listing storage files: {bucket}/{remote_dir}")
        try:
            items = storage.list(remote_dir) or []
        except Exception as exc:
            log(f"Failed to list {bucket}/{remote_dir}: {exc}")
            items = []
        files = []
        for item in items:
            name = item.get("name") if isinstance(item, dict) else None
            metadata = item.get("metadata") if isinstance(item, dict) else None
            is_dir = (metadata or {}).get("is_dir")
            if not name or is_dir:
                continue
            files.append(name)
        if not files:
            continue
        log(f"Downloading {len(files)} files from {bucket}/{remote_dir}")
        for index, name in enumerate(files, start=1):
            remote_path = f"{remote_dir}/{name}"
            local_path = os.path.join(local_dir, subdir, name)
            download_storage_file(supabase, bucket, remote_path, local_path)
            if index % 100 == 0 or index == len(files):
                log(f"Downloaded {index}/{len(files)} from {bucket}/{remote_dir}")
        total_downloaded += len(files)
    if total_downloaded:
        log(f"Downloaded {total_downloaded} files from storage prefix {bucket}/{storage_prefix}")
    return total_downloaded


def resolve_data_yaml(value, run_id, supabase):
    if not value:
        return None
    if os.path.isabs(value):
        log(f"Using local dataset YAML: {value}")
        yaml_path = value
        local_dir = os.path.dirname(yaml_path)
        return normalize_data_yaml(yaml_path, local_dir)
    storage_bucket, storage_path = extract_storage_path(value)
    if storage_path:
        bucket = storage_bucket or SUPABASE_DATASETS_BUCKET
        local_dir = os.path.join(DATASET_ROOT, "datasets", f"run_{run_id}")
        filename = os.path.basename(storage_path) or "data.yaml"
        local_path = os.path.join(local_dir, filename)
        log(f"Downloading dataset YAML from storage: {bucket}/{storage_path}")
        start_time = time.time()
        yaml_path = download_storage_file(supabase, bucket, storage_path, local_path)
        log_duration(f"Downloaded dataset YAML to {yaml_path}", start_time)
        storage_prefix = os.path.dirname(storage_path)
        if storage_prefix:
            log(f"Downloading dataset files from storage prefix: {bucket}/{storage_prefix}")
        start_time = time.time()
        downloaded = download_dataset_prefix(supabase, bucket, storage_prefix, local_dir)
        if downloaded:
            log_duration(f"Downloaded dataset files ({downloaded})", start_time)
        return normalize_data_yaml(yaml_path, local_dir)
    if value.startswith("http"):
        local_dir = os.path.join(DATASET_ROOT, "datasets", f"run_{run_id}")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, "data.yaml")
        log(f"Downloading dataset YAML from URL: {value}")
        start_time = time.time()
        urllib.request.urlretrieve(value, local_path)
        log_duration(f"Downloaded dataset YAML to {local_path}", start_time)
        return normalize_data_yaml(local_path, local_dir)
    resolved_path = os.path.join(DATASET_ROOT, value)
    log(f"Resolved dataset YAML path: {resolved_path}")
    return resolved_path


def normalize_data_yaml(yaml_path, local_dir):
    if not yaml_path or not os.path.exists(yaml_path):
        return yaml_path
    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except Exception:
        return yaml_path

    updated = []
    path_written = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("path:"):
            updated.append(f"path: {local_dir}")
            path_written = True
            continue
        if stripped.startswith("train:"):
            updated.append("train: images/train")
            continue
        if stripped.startswith("val:"):
            updated.append("val: images/val")
            continue
        if stripped.startswith("test:"):
            updated.append("test: images/test")
            continue
        updated.append(line)

    if not path_written:
        updated.insert(0, f"path: {local_dir}")

    try:
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(updated) + "\n")
    except Exception:
        return yaml_path
    return yaml_path


def normalize_task(raw_task):
    if not raw_task:
        return None
    task = str(raw_task).strip().lower()
    if task.startswith("seg"):
        return "segment"
    if task.startswith("det"):
        return "detect"
    return None


def infer_task_from_labels(dataset_root):
    if not dataset_root or not os.path.isdir(dataset_root):
        return "detect"
    saw_segment = False
    saw_bbox = False
    label_roots = [
        os.path.join(dataset_root, "labels", "train"),
        os.path.join(dataset_root, "labels", "val"),
        os.path.join(dataset_root, "labels", "test"),
    ]
    for label_root in label_roots:
        if not os.path.isdir(label_root):
            continue
        for name in os.listdir(label_root):
            if not name.endswith(".txt"):
                continue
            label_path = os.path.join(label_root, name)
            try:
                with open(label_path, "r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) == 5:
                            saw_bbox = True
                            continue
                        if len(parts) > 5:
                            saw_segment = True
                            continue
            except Exception:
                continue
    if saw_segment and not saw_bbox:
        return "segment"
    return "detect"


def format_label_value(value):
    formatted = f"{value:.6f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def convert_bbox_to_polygon(parts):
    if len(parts) != 5:
        return None
    try:
        x_center, y_center, width, height = map(float, parts[1:5])
    except ValueError:
        return None
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    coords = [x1, y1, x2, y1, x2, y2, x1, y2]
    return [parts[0]] + coords


def convert_polygon_to_bbox(parts):
    if len(parts) <= 5 or len(parts) % 2 != 1:
        return None
    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [parts[0], x_center, y_center, width, height]


def normalize_labels_for_task(task, dataset_root):
    if task not in ("detect", "segment"):
        return {"converted": 0, "invalid": 0, "files": 0}
    if not dataset_root or not os.path.isdir(dataset_root):
        return {"converted": 0, "invalid": 0, "files": 0}

    converted = 0
    invalid = 0
    updated_files = 0
    label_roots = [
        os.path.join(dataset_root, "labels", "train"),
        os.path.join(dataset_root, "labels", "val"),
        os.path.join(dataset_root, "labels", "test"),
    ]
    for label_root in label_roots:
        if not os.path.isdir(label_root):
            continue
        for name in os.listdir(label_root):
            if not name.endswith(".txt"):
                continue
            label_path = os.path.join(label_root, name)
            try:
                with open(label_path, "r", encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
            except Exception:
                continue
            updated_lines = []
            changed = False
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    if task == "segment":
                        converted_line = convert_bbox_to_polygon(parts)
                        if converted_line:
                            converted += 1
                            changed = True
                            formatted = [converted_line[0]] + [
                                format_label_value(value) for value in converted_line[1:]
                            ]
                            updated_lines.append(" ".join(formatted))
                        else:
                            invalid += 1
                            changed = True
                    else:
                        updated_lines.append(line)
                    continue
                if len(parts) > 5 and len(parts) % 2 == 1:
                    if task == "detect":
                        converted_line = convert_polygon_to_bbox(parts)
                        if converted_line:
                            converted += 1
                            changed = True
                            formatted = [converted_line[0]] + [
                                format_label_value(value) for value in converted_line[1:]
                            ]
                            updated_lines.append(" ".join(formatted))
                        else:
                            invalid += 1
                            changed = True
                    else:
                        updated_lines.append(line)
                    continue
                invalid += 1
                changed = True
            if changed:
                try:
                    with open(label_path, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(updated_lines))
                        if updated_lines:
                            handle.write("\n")
                    updated_files += 1
                except Exception:
                    continue
    return {"converted": converted, "invalid": invalid, "files": updated_files}


def scan_label_stats(dataset_root):
    stats = {
        "total_labels": 0,
        "label_files": 0,
        "empty_files": 0,
        "class_ids": set(),
    }
    if not dataset_root or not os.path.isdir(dataset_root):
        return stats
    label_roots = [
        os.path.join(dataset_root, "labels", "train"),
        os.path.join(dataset_root, "labels", "val"),
        os.path.join(dataset_root, "labels", "test"),
    ]
    for label_root in label_roots:
        if not os.path.isdir(label_root):
            continue
        for name in os.listdir(label_root):
            if not name.endswith(".txt"):
                continue
            stats["label_files"] += 1
            label_path = os.path.join(label_root, name)
            try:
                with open(label_path, "r", encoding="utf-8") as handle:
                    saw_label = False
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        saw_label = True
                        stats["total_labels"] += 1
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        stats["class_ids"].add(class_id)
                    if not saw_label:
                        stats["empty_files"] += 1
            except Exception:
                continue
    return stats


def resolve_task(config, data_yaml):
    task = normalize_task(config.get("task"))
    if task:
        return task
    task = normalize_task(config.get("modelTask"))
    if task:
        return task
    task = normalize_task(config.get("trainingTask"))
    if task:
        return task
    dataset_root = os.path.dirname(data_yaml) if data_yaml else None
    return infer_task_from_labels(dataset_root)


def map_base_model(value, task):
    if not value:
        return MODEL_MAP_DETECT["YOLOv8s"] if task == "detect" else MODEL_MAP_SEGMENT["YOLOv8s"]
    if value.endswith(".pt"):
        return value
    model_map = MODEL_MAP_SEGMENT if task == "segment" else MODEL_MAP_DETECT
    return model_map.get(value, value)


def parse_config(raw_config):
    if not raw_config:
        return {}
    if isinstance(raw_config, dict):
        return raw_config
    try:
        return json.loads(raw_config)
    except json.JSONDecodeError:
        return {}


AUGMENTATION_KEYS = {
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
}


def normalize_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_augmentation_kwargs(config):
    if not isinstance(config, dict):
        return {}
    augmentation = config.get("augmentation") or {}
    if isinstance(augmentation, str):
        try:
            augmentation = json.loads(augmentation)
        except json.JSONDecodeError:
            augmentation = {}
    values = {}
    for key in AUGMENTATION_KEYS:
        raw = augmentation.get(key) if isinstance(augmentation, dict) else None
        if raw is None and isinstance(config, dict):
            raw = config.get(key)
        value = normalize_float(raw)
        if value is not None:
            values[key] = value
    return values


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


def guess_content_type(filename):
    if not filename:
        return "application/octet-stream"
    extension = os.path.splitext(filename.lower())[1]
    if extension in (".pt", ".pth", ".bin"):
        return "application/octet-stream"
    if extension == ".csv":
        return "text/csv"
    if extension == ".jpg" or extension == ".jpeg":
        return "image/jpeg"
    if extension == ".png":
        return "image/png"
    if extension == ".webp":
        return "image/webp"
    if extension == ".json":
        return "application/json"
    if extension in (".txt", ".log"):
        return "text/plain"
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def format_eta(seconds):
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def read_latest_metrics(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None, None
    row = rows[-1]
    epoch_value = None
    for key in ("epoch", "Epoch"):
        raw = row.get(key)
        if raw not in (None, ""):
            try:
                epoch_value = int(float(raw))
            except ValueError:
                epoch_value = None
            break
    if epoch_value is None:
        epoch = len(rows)
    else:
        epoch = epoch_value + 1
    row["epoch"] = epoch
    return row, epoch


def update_training_progress(supabase, run_id, total_epochs, results_csv_path, started_at, last_epoch):
    row, epoch = read_latest_metrics(results_csv_path)
    if epoch is None or epoch == last_epoch:
        return last_epoch
    percent = None
    if total_epochs:
        percent = min(100.0, (epoch / total_epochs) * 100.0)
    elapsed = time.time() - started_at
    eta_seconds = None
    if total_epochs and epoch > 0:
        seconds_per_epoch = elapsed / epoch
        eta_seconds = int(seconds_per_epoch * max(0, total_epochs - epoch))
    updates = {
        "results": {
            "metrics": row,
            "progress": {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "percent": percent,
                "eta_seconds": eta_seconds,
                "eta": format_eta(eta_seconds),
            },
        }
    }
    update_run(supabase, run_id, updates)
    return epoch


def upload_file(supabase, local_path, remote_path):
    content_type = guess_content_type(local_path)
    with open(local_path, "rb") as handle:
        supabase.storage.from_(SUPABASE_ARTIFACTS_BUCKET).upload(
            remote_path,
            handle,
            {"content-type": content_type, "x-upsert": "true"},
        )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_ARTIFACTS_BUCKET}/{remote_path}"


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
        .eq("cancel_requested", False)
        .is_("started_at", "null")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )
    error = getattr(response, "error", None)
    if error and "cancel_requested" in str(error):
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


def is_cancel_requested(supabase, run_id):
    try:
        response = (
            supabase.table(RUNS_TABLE)
            .select("cancel_requested")
            .eq("id", run_id)
            .single()
            .execute()
        )
    except Exception:
        return False
    error = getattr(response, "error", None)
    if error and "cancel_requested" in str(error):
        return False
    data = response.data or {}
    return bool(data.get("cancel_requested"))


def upsert_worker_heartbeat(supabase, status="online"):
    payload = {
        "worker_id": WORKER_ID,
        "status": status,
        "last_seen": utc_now(),
    }
    try:
        supabase.table(WORKERS_TABLE).upsert(payload, on_conflict="worker_id").execute()
    except Exception as exc:
        log(f"Failed to update worker heartbeat: {exc}")


def start_heartbeat_thread(supabase, status):
    stop_event = threading.Event()

    def loop():
        while not stop_event.is_set():
            upsert_worker_heartbeat(supabase, status=status)
            stop_event.wait(HEARTBEAT_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop_event, thread


def start_progress_thread(supabase, run_id, total_epochs, results_csv_path, cancel_event):
    stop_event = threading.Event()
    started_at = time.time()
    last_epoch_holder = {"value": 0}

    def loop():
        while not stop_event.is_set() and not cancel_event.is_set():
            last_epoch_holder["value"] = update_training_progress(
                supabase,
                run_id,
                total_epochs,
                results_csv_path,
                started_at,
                last_epoch_holder["value"],
            )
            stop_event.wait(PROGRESS_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop_event, thread


def start_cancel_poll_thread(supabase, run_id, cancel_event):
    stop_event = threading.Event()

    def loop():
        while not stop_event.is_set() and not cancel_event.is_set():
            if is_cancel_requested(supabase, run_id):
                cancel_event.set()
                update_run(supabase, run_id, {"status": "canceling"})
                break
            stop_event.wait(CANCEL_CHECK_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop_event, thread


def build_cancel_callbacks(model, cancel_event):
    def check_cancel(trainer):
        if cancel_event.is_set():
            trainer.stop = True

    if hasattr(model, "add_callback"):
        added = 0
        for name in ("on_fit_epoch_end", "on_train_epoch_end", "on_train_batch_end"):
            try:
                model.add_callback(name, check_cancel)
                added += 1
            except Exception:
                continue
        if added:
            return None

    return {
        "on_fit_epoch_end": check_cancel,
        "on_train_epoch_end": check_cancel,
    }


def run_training_job(supabase, run):
    run_id = run["id"]
    config = parse_config(run.get("configuration"))
    log(f"Run {run_id}: resolving dataset YAML")
    step_start = time.time()
    data_yaml = resolve_data_yaml(run.get("data_yaml") or config.get("dataYaml"), run_id, supabase)
    log_duration(f"Run {run_id}: resolved dataset YAML to {data_yaml}", step_start)
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
    if is_cancel_requested(supabase, run_id):
        update_run(
            supabase,
            run_id,
            {
                "status": "canceled",
                "canceled_at": utc_now(),
                "completed_at": utc_now(),
                "error_message": "Canceled by user.",
            },
        )
        return

    task = resolve_task(config, data_yaml)
    if task not in ("detect", "segment"):
        task = "detect"
    dataset_root = os.path.dirname(data_yaml) if data_yaml else None
    log(f"Run {run_id}: normalizing labels for task={task}")
    step_start = time.time()
    normalization = normalize_labels_for_task(task, dataset_root)
    log_duration(
        (
            f"Run {run_id}: label normalization done "
            f"(converted={normalization['converted']}, "
            f"invalid={normalization['invalid']}, files={normalization['files']})"
        ),
        step_start,
    )
    if normalization["converted"] or normalization["invalid"]:
        log(
            "Normalized labels for task="
            f"{task}: converted={normalization['converted']}, "
            f"invalid={normalization['invalid']}, "
            f"files={normalization['files']}"
        )
    log(f"Run {run_id}: scanning label stats in {dataset_root}")
    step_start = time.time()
    label_stats = scan_label_stats(dataset_root)
    log_duration(
        (
            f"Run {run_id}: label scan found {label_stats['total_labels']} labels "
            f"in {label_stats['label_files']} files ({label_stats['empty_files']} empty)"
        ),
        step_start,
    )
    if label_stats["total_labels"] == 0:
        update_run(
            supabase,
            run_id,
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error_message": (
                    "No labeled annotations found in the dataset. "
                    "Add labels (and ensure class names are assigned) then retry."
                ),
            },
        )
        return
    model_path = map_base_model(run.get("base_model"), task)
    epochs = int(config.get("epochs", 100))
    batch = int(config.get("batchSize", 16))
    imgsz = int(config.get("imgSize", 640))
    lr0 = float(config.get("learningRate", 0.001))
    optimizer = config.get("optimizer", "Adam")
    device = config.get("device", 0)
    log(
        "Run "
        f"{run_id}: task={task}, model={model_path}, epochs={epochs}, "
        f"batch={batch}, imgsz={imgsz}, lr0={lr0}, optimizer={optimizer}, device={device}"
    )

    results_csv_path = os.path.join(RUNS_DIR, f"train_{run_id}", "results.csv")
    update_run(
        supabase,
        run_id,
        {
            "results": {
                "progress": {
                    "epoch": 0,
                    "total_epochs": epochs,
                    "percent": 0.0,
                    "eta_seconds": None,
                    "eta": None,
                }
            }
        },
    )

    cancel_event = threading.Event()
    cancel_stop, cancel_thread = start_cancel_poll_thread(supabase, run_id, cancel_event)
    progress_stop, progress_thread = start_progress_thread(
        supabase,
        run_id,
        epochs,
        results_csv_path,
        cancel_event,
    )

    model_exists = os.path.exists(model_path)
    log(f"Run {run_id}: loading model {model_path} (local={model_exists})")
    step_start = time.time()
    model = YOLO(model_path)
    log_duration(f"Run {run_id}: model loaded", step_start)
    callbacks = build_cancel_callbacks(model, cancel_event)
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "lr0": lr0,
        "optimizer": optimizer,
        "device": device,
        "project": RUNS_DIR,
        "name": f"train_{run_id}",
        "exist_ok": True,
    }
    augmentation_kwargs = extract_augmentation_kwargs(config)
    if augmentation_kwargs:
        train_kwargs.update(augmentation_kwargs)
    if callbacks:
        train_kwargs["callbacks"] = callbacks
    log(f"Run {run_id}: starting training")
    try:
        results = model.train(**train_kwargs)
    except TypeError as exc:
        if "callbacks" in str(exc) and "callbacks" in train_kwargs:
            train_kwargs.pop("callbacks", None)
            results = model.train(**train_kwargs)
        else:
            raise

    log(f"Run {run_id}: training finished, collecting artifacts")
    progress_stop.set()
    progress_thread.join(timeout=2)
    cancel_stop.set()
    cancel_thread.join(timeout=2)

    if cancel_event.is_set():
        update_run(
            supabase,
            run_id,
            {
                "status": "canceled",
                "canceled_at": utc_now(),
                "completed_at": utc_now(),
                "error_message": "Canceled by user.",
            },
        )
        return

    save_dir = str(getattr(results, "save_dir", os.path.join(RUNS_DIR, f"train_{run_id}")))
    metrics = parse_results_csv(os.path.join(save_dir, "results.csv"))
    step_start = time.time()
    artifacts = upload_artifacts(supabase, run_id, save_dir)
    log_duration(f"Run {run_id}: uploaded {len(artifacts)} artifacts", step_start)

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

    storage_url = f"{SUPABASE_URL}storage/v1/"
    os.environ.setdefault("SUPABASE_STORAGE_URL", storage_url)
    if ClientOptions:
        try:
            supabase = create_client(
                SUPABASE_URL,
                SUPABASE_SERVICE_ROLE_KEY,
                options=ClientOptions(storage_url=storage_url),
            )
        except Exception:
            supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    else:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    os.makedirs(RUNS_DIR, exist_ok=True)
    log(f"Trainer service started (worker_id={WORKER_ID})")
    log(f"Polling {RUNS_TABLE} every {POLL_INTERVAL}s")

    while True:
        upsert_worker_heartbeat(supabase, status="online")
        run = get_next_run(supabase)
        if not run:
            log(f"No queued runs. Sleeping {POLL_INTERVAL}s.")
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

        stop_event, heartbeat_thread = start_heartbeat_thread(supabase, status="busy")
        try:
            log(f"Starting run {run['id']}")
            run_training_job(supabase, run)
            log(f"Completed run {run['id']}")
        except Exception as exc:
            log(f"Run {run['id']} failed: {exc}")
            update_run(
                supabase,
                run["id"],
                {
                    "status": "failed",
                    "completed_at": utc_now(),
                    "error_message": str(exc),
                },
            )
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=2)
            upsert_worker_heartbeat(supabase, status="online")


if __name__ == "__main__":
    main()
