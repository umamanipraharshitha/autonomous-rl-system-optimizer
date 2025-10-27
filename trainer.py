import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path

# ---------------------- PATHS ----------------------
MODEL_PATH = Path("models/rl_agent_weights.pth")
LABELED_PATH = Path("data/system_logs.labeled.jsonl")
TRAIN_LOG_PATH = Path("logs/training_log.json")  # ✅ new log file


# ---------------------- HELPERS ----------------------
def ensure_dir_exists(path):
    d = path.parent
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)


def save_model(state_dict):
    ensure_dir_exists(MODEL_PATH)
    torch.save(state_dict, MODEL_PATH)


def log_training(metrics):
    """Append training info (loss, time, sample count) to a JSON log file."""
    ensure_dir_exists(TRAIN_LOG_PATH)
    logs = []
    if TRAIN_LOG_PATH.exists():
        try:
            with open(TRAIN_LOG_PATH, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    logs.append(metrics)
    with open(TRAIN_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


# ---------------------- Q-NETWORK ----------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ---------------------- TRAINER ----------------------
class Trainer:
    def __init__(self):
        self.input_dim = 3  # cpu, ram, gamma
        self.output_dim = 3  # decrease, maintain, increase
        self.net = QNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def build_dataset(self):
        X, Y = [], []
        try:
            with open(LABELED_PATH) as f:
                for line in f:
                    obj = json.loads(line)
                    cpu = obj.get('cpu_percent', 0)
                    ram = obj.get('virtual_memory_percent', 0)
                    gamma = random.uniform(0, 100)

                    label = obj.get('label', 'general_cleanup')
                    if label in ['close_heavy_browser_tabs', 'free_up_ram']:
                        action = 0  # decrease
                    elif label == 'clean_disk':
                        action = 1  # maintain
                    else:
                        action = 2  # increase

                    X.append([cpu, ram, gamma])
                    target = [0.0] * self.output_dim
                    target[action] = 1.0
                    Y.append(target)
        except FileNotFoundError:
            pass

        if not X:
            return None, None

        X = torch.FloatTensor(np.array(X))
        Y = torch.FloatTensor(np.array(Y))
        return X, Y

    def train_once(self, epochs=10):
        X, Y = self.build_dataset()
        if X is None:
            return False

        epoch_losses = []
        for _ in range(epochs):
            preds = self.net(X)
            loss = self.criterion(preds, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        save_model(self.net.state_dict())

        # ✅ Log training stats to JSON
        metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": len(X),
            "epochs": epochs,
            "avg_loss": round(avg_loss, 6)
        }
        log_training(metrics)
        return True


# ---------------------- MAIN LOOP ----------------------
if __name__ == '__main__':
    print("[Trainer] Trainer started — will retrain periodically if labeled data exists...")
    trainer = Trainer()
    while True:
        ok = trainer.train_once()
        if ok:
            print("[Trainer] Model trained and saved. Loss logged to logs/training_log.json")
        else:
            print("[Trainer] No labeled data yet — skipping")
        time.sleep(60)
