import pandas as pd
import numpy as np
from pymongo import MongoClient
import torch
import torch.nn as nn

# Connect to MongoDB (make sure MongoDB is running)
client = MongoClient('mongodb://localhost:27017/')
db = client.sys_perf
collection = db.metrics

# Retrieve data from MongoDB
data = list(collection.find())
if not data:
    print("No data found in MongoDB. Please insert data into the 'metrics' collection first.")
    exit()

df = pd.DataFrame(data)

# Convert 'timestamp' to datetime, sort by time, and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
df.set_index('timestamp', inplace=True)

# We'll predict CPU usage for this example
cpu_series = df['cpu']

# Create sequences using a sliding window approach
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        seq = series.iloc[i:i + window_size].values
        target = series.iloc[i + window_size]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

window_size = 10  # Adjust window size based on your dataset size
print("Total data points:", len(cpu_series))

X, y = create_sequences(cpu_series, window_size)

if X.size == 0:
    print("Not enough data to create sequences. Please collect more data or lower the window size.")
    exit()

# Reshape X for LSTM: [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print("Data prepared:", X.shape, y.shape)

# Define the LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)          # out shape: [batch, time_steps, hidden_size]
        out = self.fc(out[:, -1, :])    # take the output at the last time step
        return out

# Convert numpy arrays to torch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(1)

# Instantiate the model, define loss function and optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 20
batch_size = 32

# Function to generate batches
def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in get_batches(X_tensor, y_tensor, batch_size):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")
print("Training complete!")
