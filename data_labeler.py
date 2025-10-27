import json
import time
from utils import LOG_PATH, LABELED_PATH

# ---------------------- RULES ----------------------
RULES = [
    (
        lambda d: d.get('cpu_percent', 0) > 85
        and any('chrome' in (p or '').lower() for p in d.get('top_processes', [])),
        'close_heavy_browser_tabs',
    ),
    (lambda d: d.get('virtual_memory_percent', 0) > 85, 'free_up_ram'),
    (lambda d: d.get('disk_percent', 0) > 90, 'clean_disk'),
    (lambda d: d.get('num_processes', 0) > 300, 'close_background_apps'),
]
DEFAULT = 'general_cleanup'


# ---------------------- LABELING FUNCTION ----------------------
def label_entry(entry):
    """Apply labeling rules to a single log entry."""
    for cond, label in RULES:
        try:
            if cond(entry):
                return label
        except Exception:
            continue
    return DEFAULT


# ---------------------- LABELER LOOP ----------------------
def run_labeler(poll_interval=10):
    """Continuously reads logs and appends labels automatically."""
    seen = 0
    while True:
        try:
            with open(LOG_PATH, 'r') as f:
                lines = f.read().strip().splitlines()
        except FileNotFoundError:
            lines = []

        new = lines[seen:]
        if new:
            with open(LABELED_PATH, 'a') as fo:
                for line in new:
                    obj = json.loads(line)
                    obj['label'] = label_entry(obj)
                    fo.write(json.dumps(obj) + '\n')
            seen = len(lines)

        time.sleep(poll_interval)


# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    print(' Starting labeler... labeling new logs automatically...')
    run_labeler()
