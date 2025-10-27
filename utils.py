import json
from pathlib import Path

# Path to your log file
LOG_PATH = Path("data/system_logs.jsonl")
LABELED_PATH = Path("data/system_logs.labeled.jsonl")

def ensure_dir_exists(path):
    """Ensure parent directories exist."""
    d = path.parent
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)


def append_log(entry, path):
    """Append a JSON entry to a log file."""
    ensure_dir_exists(path)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_logs(path):
    """Read all JSONL logs as a list of dictionaries."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
