import subprocess
import time
import os
import signal
from pathlib import Path

# --- Paths to your scripts ---
SCRIPTS = {
    "data_collector": "data_collector.py",
    "data_labeler": "data_labeler.py",
    "trainer": "trainer.py",
    "rl_agent_live": "rl_agent_live.py"
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
processes = {}
restart_counts = {}

def start_process(name, script):
    """Start a script as a background process and log output."""
    log_path = LOG_DIR / f"{name}.log"
    log_file = open(log_path, "a", buffering=1)
    print(f"🚀 Starting {name} (logging to {log_path})")
    p = subprocess.Popen(
        ["python", script],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True
    )
    processes[name] = (p, log_file)
    restart_counts[name] = restart_counts.get(name, 0)
    return p

def stop_process(name):
    """Safely stop a process."""
    if name in processes:
        p, log_file = processes[name]
        if p.poll() is None:
            os.kill(p.pid, signal.SIGTERM)
        log_file.close()
        processes.pop(name, None)
        print(f"🛑 Stopped {name}")

def restart_process(name):
    """Restart a crashed process with cooldown if needed."""
    restart_counts[name] = restart_counts.get(name, 0) + 1
    if restart_counts[name] > 3:
        print(f"⏸ Too many restarts for {name}, waiting 30s before retry...")
        time.sleep(30)
        restart_counts[name] = 0
    print(f"⚠️ Restarting {name}...")
    stop_process(name)
    time.sleep(1)
    start_process(name, SCRIPTS[name])

def check_processes():
    """Monitor processes and restart if needed."""
    for name, (proc, _) in list(processes.items()):
        if proc.poll() is not None:
            print(f"💥 {name} crashed — restarting...")
            restart_process(name)

if __name__ == "__main__":
    print("\n🤖 AutoSupervisor started — managing all modules...\n")

    for name, script in SCRIPTS.items():
        start_process(name, script)
        time.sleep(2)

    try:
        while True:
            check_processes()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all processes...")
        for name in list(processes.keys()):
            stop_process(name)
        print("✅ All modules stopped. Goodbye!")
