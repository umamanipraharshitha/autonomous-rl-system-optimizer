#  Autonomous RL System Optimizer

An **autonomous reinforcement learning pipeline** that **monitors, learns, and optimizes** system performance in real time.
It continuously analyzes CPU, RAM, and disk metrics, dynamically adjusts a tunable parameter (`gamma`), and provides a **React-based dashboard** for live visualization.

---

##  Overview

This project creates an **end-to-end intelligent optimization system** using reinforcement learning:

1. **Data Collector** – samples system metrics.
2. **Data Labeler** – applies rule-based labels.
3. **Trainer** – trains an RL model using labeled data.
4. **Live RL Agent** – runs the trained model and dynamically adjusts system parameters.
5. **Flask API (`app.py`)** – serves live and historical logs to the frontend.
6. **React Dashboard** – visualizes system metrics, rewards, and RL actions in real time.

---
Sure — here’s a clean, concise version with **one line per layer**, perfect for your `README.md`:

---

**Machine Learning:** Python, PyTorch, NumPy
**System Monitoring:** psutil, json, pathlib, rule-based labeling
**Backend:** Flask, Flask-CORS, REST API (JSON)
**Frontend:** React.js, Axios, Chart.js, Tailwind CSS
**Data & Logs:** JSONL, JSON
**Development Tools:** Git, GitHub, pip, Node.js, npm


## The Role of `gamma`

The **`gamma` parameter** is a **tunable control variable** — currently used to adjust system optimization intensity.
In future versions, it can be mapped to **hardware-level controls** such as:

* CPU/GPU frequency scaling
* Cooling fan speed
* Power or thermal throttling
* Process scheduling or priority levels
* Energy efficiency trade-offs

The RL agent continuously learns the **best adjustment policy** for `gamma` based on real-time rewards derived from system performance.

---

##  Module Architecture

| Component           | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `data_collector.py` | Collects live CPU, RAM, Disk, and process metrics.                    |
| `data_labeler.py`   | Labels collected data using heuristic rules.                          |
| `trainer.py`        | Trains a neural Q-network using labeled logs.                         |
| `rl_agent_live.py`  | Runs the RL agent to make real-time optimization decisions.           |
| `autosupervisor.py` | Manages all modules and restarts them if they crash.                  |
| `app.py`            | Flask backend serving API data to the dashboard.                      |
| `frontend`         | React dashboard that visualizes system metrics, actions, and rewards. |

---

##  Flask API Endpoints

| Endpoint        | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `/api/system`   | Returns recent system logs (`data/system_logs.jsonl`)            |
| `/api/live`     | Returns live RL decisions and rewards (`logs/rl_live_log.jsonl`) |
| `/api/training` | Returns training history (`logs/training_log.json`)              |

---

##  React Dashboard

The frontend (in `dashboard/frontend`) connects to Flask’s API and visualizes:

* **CPU / RAM usage** (line charts)
* **Live gamma adjustments & actions** (real-time display)
* **Reward trends** (performance improvement over time)
* **Training progress** (loss curves, epochs, sample counts)


##  Run Locally

### 1️⃣ Backend Setup

```bash
git clone https://github.com/your-username/autonomous-rl-system-optimizer.git
cd autonomous-rl-system-optimizer
pip install -r requirements.txt
python autosupervisor.py  # Starts all agents
python app.py              # Runs Flask API (port 5000)
```

### 2️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start  # React runs on http://localhost:3000
```

## Data & Logs

| File                             | Description                |
| -------------------------------- | -------------------------- |
| `data/system_logs.jsonl`         | Raw system metrics         |
| `data/system_logs.labeled.jsonl` | Labeled data for training  |
| `logs/training_log.json`         | Model training progress    |
| `logs/rl_live_log.jsonl`         | Real-time RL activity logs |

---

##  Reinforcement Learning Model

* **State:** `[CPU%, RAM%, gamma]`
* **Actions:** `[Decrease γ, Maintain γ, Increase γ]`
* **Reward:**

  ```
  reward = 100 - (0.6 * CPU + 0.3 * RAM + 0.1 * Disk)
  ```
* **Goal:** Maintain balanced resource usage for optimal performance.

---

##  Future Work

* Integrate **hardware control APIs** for real power/fan/speed management.
* Add **real-time charts and alerts** in the dashboard.

---

##  License

Released under the **MIT License** — free to use, modify, and extend.

