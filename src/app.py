"""
app.py - Demonstration of:
1) Real-time alerts when predicted CPU usage crosses a threshold.
2) Expanded dashboard with multiple metrics (CPU, memory, GPU).
3) Basic date range filtering for historical metrics.
"""
import os
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import psutil

# -----------------------------
# 1) Model Definition (LSTM)
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# 2) Flask App Setup
# -----------------------------
app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.sys_perf
metrics_collection = db.metrics
alerts_collection = db.alerts  # We'll store alerts here

# Load or create a dummy model
MODEL_PATH = 'model.pth'
model = LSTMModel()
if os.path.exists(MODEL_PATH):
    print("Loading existing model.pth")
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    print("No model found. Using an untrained LSTM for demonstration.")
model.eval()

# -----------------------------
# 3) Routes
# -----------------------------

@app.route('/')
def index():
    """Render the expanded dashboard with multiple metrics and date filtering."""
    return render_template('index.html')


@app.route('/api/metrics')
def get_metrics():
    """
    Returns CPU, memory, and GPU usage data from MongoDB.
    Supports optional query params: ?start=YYYY-MM-DD&end=YYYY-MM-DD
    """
    # Parse optional date range filters
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    
    query = {}
    if start_str:
        try:
            start_date = datetime.fromisoformat(start_str)
            query['timestamp'] = {'$gte': start_date}
        except ValueError:
            pass  # Ignore invalid date
    
    if end_str:
        try:
            end_date = datetime.fromisoformat(end_str)
            if 'timestamp' in query:
                query['timestamp']['$lte'] = end_date
            else:
                query['timestamp'] = {'$lte': end_date}
        except ValueError:
            pass
    
    # Fetch up to 100 docs that match the query, sorted by timestamp ascending
    data = list(metrics_collection.find(query).sort('timestamp', 1).limit(100))
    
    # Convert ObjectId and datetime to strings for JSON
    for d in data:
        d['_id'] = str(d['_id'])
        d['timestamp'] = d['timestamp'].isoformat()
    
    return jsonify(data)


@app.route('/api/prediction')
def get_prediction():
    """
    Predicts the next CPU usage based on live data.
    If the predicted CPU usage exceeds a threshold, logs an alert.
    Returns JSON with the predicted CPU, alert flag, and alert message.
    """
    # Set a low threshold for testing
    THRESHOLD = 0.5  # Temporarily lower threshold for demonstration

    window_size = 10
    live_data = [psutil.cpu_percent(interval=0.3) for _ in range(window_size)]
    X_live = np.array(live_data).reshape((1, window_size, 1)).astype(np.float32)

    with torch.no_grad():
        prediction_tensor = model(torch.from_numpy(X_live))
    predicted_cpu = prediction_tensor.item()

    # Uncomment the next line to force a high prediction for testing:
    # predicted_cpu = 85.0

    alert_triggered = False
    alert_message = ""

    if predicted_cpu > THRESHOLD:
        alert_triggered = True
        alert_message = f"High CPU usage predicted: {predicted_cpu:.2f}%"
        alerts_collection.insert_one({
            "timestamp": datetime.now(),
            "message": alert_message,
            "predicted_cpu": predicted_cpu
        })

    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "predicted_cpu": predicted_cpu,
        "alert_triggered": alert_triggered,
        "alert_message": alert_message
    })


@app.route('/api/alerts')
def get_alerts():
    """
    Returns recent alerts from the 'alerts' collection.
    """
    recent_alerts = list(alerts_collection.find().sort('timestamp', -1).limit(10))
    for a in recent_alerts:
        a['_id'] = str(a['_id'])
        a['timestamp'] = a['timestamp'].isoformat()
    return jsonify(recent_alerts)


def generate_recommendations(metrics):
    """
    Generate performance improvement recommendations based on recent metrics.
    :param metrics: A list of recent metric documents from MongoDB.
    :return: A list of recommendation strings.
    """
    recommendations = []
    
    if not metrics:
        return recommendations

    # Calculate average CPU and Memory usage over recent data
    avg_cpu = sum(doc.get('cpu', 0) for doc in metrics) / len(metrics)
    avg_memory = sum(doc.get('memory', 0) for doc in metrics) / len(metrics)
    
    # Example rules:
    if avg_cpu > 75:
        recommendations.append("Your average CPU usage is high. Consider closing background applications or lowering in-game graphics settings.")
    if avg_memory > 80:
        recommendations.append("Your memory usage is high. Consider closing memory-intensive applications.")
    # You can add more rules based on GPU or other parameters.
    
    return recommendations

@app.route('/api/recommendations')
def get_recommendations():
    """
    Endpoint that returns performance improvement recommendations based on the latest 50 metrics.
    """
    data = list(metrics_collection.find().sort('timestamp', -1).limit(50))
    # Reverse to have oldest first
    data = data[::-1]
    # Generate recommendations based on these metrics
    recs = generate_recommendations(data)
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "recommendations": recs
    })

if __name__ == '__main__':
    app.run(debug=True)
