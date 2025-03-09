import time
import torch
import numpy as np
from datetime import datetime
import psutil  # For live CPU usage monitoring
# Assume model is loaded from a saved state
from train_model import LSTMModel, create_sequences

# Load your trained model (if saved, or use the current instance)
model = LSTMModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Function to collect a short window of live data (simulate using psutil for CPU)
def collect_live_data(window_size):
    data = []
    for _ in range(window_size):
        cpu = psutil.cpu_percent(interval=1)
        data.append(cpu)
    return np.array(data)

window_size = 10

while True:
    # Collect a window of live data
    live_data = collect_live_data(window_size)
    
    # Prepare the data similar to training (reshape etc.)
    X_live = live_data.reshape((1, window_size, 1))
    X_live_tensor = torch.from_numpy(X_live).float()
    
    # Predict the next CPU value
    with torch.no_grad():
        prediction = model(X_live_tensor)
    print(f"{datetime.now()}: Predicted CPU usage: {prediction.item():.2f}%")
    
    # You can add logic here to trigger alerts if the prediction exceeds a threshold
    
    time.sleep(5)  # Wait before next prediction cycle
