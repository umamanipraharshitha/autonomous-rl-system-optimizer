# Continuously collects real system metrics and appends to data/system_logs.jsonl
import json
import time
from datetime import datetime
import psutil
from utils import append_log, LOG_PATH

SLEEP_SECONDS = 5  # time gap between each sample


def sample_once():
    try:
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "virtual_memory_percent": psutil.virtual_memory().percent,
            "swap_percent": psutil.swap_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "num_processes": len(psutil.pids()),
            "top_processes": [p.info.get('name') for p in psutil.process_iter(['name'])][:5]
        }

        # Try reading temperature sensors (optional, may fail on some systems)
        try:
            temps = psutil.sensors_temperatures()
            data['temperature'] = None
            for k, v in temps.items():
                if v:
                    data['temperature'] = v[0].current
                    break
        except Exception:
            data['temperature'] = None

        return data

    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    print("Starting data_collector: writing to", LOG_PATH)
    while True:
        entry = sample_once()
        append_log(entry, LOG_PATH)
        print(f"Wrote: {entry.get('timestamp')} | CPU: {entry.get('cpu_percent')}%")
        time.sleep(SLEEP_SECONDS)
