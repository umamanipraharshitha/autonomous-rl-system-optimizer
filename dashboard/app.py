from flask import Flask, jsonify
from flask_cors import CORS
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow React frontend to call API

DATA_PATH = Path("../data/system_logs.jsonl")
LIVE_PATH = Path("../logs/rl_live_log.jsonl")
TRAIN_PATH = Path("../logs/training_log.json")

def read_jsonl(path, limit=30):
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines if l.strip()]

@app.route("/api/system")
def system_data():
    return jsonify(read_jsonl(DATA_PATH))

@app.route("/api/live")
def live_data():
    return jsonify(read_jsonl(LIVE_PATH))

@app.route("/api/training")
def training_data():
    if not TRAIN_PATH.exists():
        return jsonify([])
    with open(TRAIN_PATH) as f:
        data = json.load(f)
    return jsonify(data[-30:])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
