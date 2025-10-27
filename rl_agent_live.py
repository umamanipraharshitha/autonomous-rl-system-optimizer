import time
import random
import torch
import torch.nn as nn
from pathlib import Path
import json

# ---------------------- PATHS ----------------------
DATA_PATH = Path("data/system_logs.jsonl")
MODEL_PATH = Path("models/rl_agent_weights.pth")
LIVE_LOG_PATH = Path("logs/rl_live_log.jsonl")  # <-- new log file

# ---------------------- HELPERS ----------------------
def read_latest(n=1):
    """Read the latest n entries from system_logs.jsonl"""
    try:
        with open(DATA_PATH, "r") as f:
            lines = f.readlines()
        if not lines:
            return []
        latest = [json.loads(x) for x in lines[-n:]]
        return latest
    except FileNotFoundError:
        return []


def compute_reward(metrics):
    """Reward function based on system performance"""
    cpu = metrics.get("cpu_percent", 0)
    ram = metrics.get("virtual_memory_percent", 0)
    disk = metrics.get("disk_percent", 0)
    return 100 - (0.6 * cpu + 0.3 * ram + 0.1 * disk)


# ---------------------- Q-NETWORK ----------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ---------------------- LIVE RL AGENT ----------------------
def run_live():
    print(" Starting live RL autotuning...")

    input_dim = 3   # CPU, RAM, gamma
    output_dim = 3  # decrease, maintain, increase gamma

    net = QNetwork(input_dim, output_dim)

    # Try to load trained model
    if MODEL_PATH.exists():
        try:
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
            net.load_state_dict(state_dict)
            net.eval()
            print(" Loaded trained model for live agent.")
        except Exception as e:
            print("Model load failed, using random policy:", e)
    else:
        print(" No trained model found, using random policy.")

    gamma_val = 50
    epsilon = 0.3   # 30% exploration

    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    while True:
        latest = read_latest(1)
        if not latest:
            print(" Waiting for system logs...")
            time.sleep(3)
            continue

        m = latest[0]
        cpu = m.get("cpu_percent", 0)
        ram = m.get("virtual_memory_percent", 0)

        state = torch.FloatTensor([cpu, ram, gamma_val]).unsqueeze(0)

        # Choose action (ε-greedy)
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                out = net(state)
                action = int(out.argmax().item())

        # Update gamma based on action
        if action == 0:
            gamma_val = max(0, gamma_val - 5)
        elif action == 2:
            gamma_val = min(100, gamma_val + 5)

        reward = compute_reward(m)

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_percent": cpu,
            "ram_percent": ram,
            "action": action,
            "gamma": gamma_val,
            "reward": reward
        }

        # Append to JSONL file
        with open(LIVE_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f" CPU:{cpu:.1f}% RAM:{ram:.1f}% → "
            f"Action:{action} → Gamma:{gamma_val} → Reward:{reward:.2f}"
        )

        time.sleep(5)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    run_live()
