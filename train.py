import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# === configuration ===
csv_path = "./dataset/train_pca.csv" 
input_window = 30  
target_window = 5 
device = torch.device("cpu") # CPU
if torch.cuda.is_available():
    device = torch.device("cuda") # nVidia
elif torch.mps.is_available():
    device = torch.device("mps") # Apple silicon
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("./lossPic", exist_ok=True)

# === read data ===
df = pd.read_csv(csv_path)
features = ['T1', 'T2', 'T3', 'T4', 'PC1', 'PC2', 'PC3']
target_col = 'total_cpu_rate'

X_raw = df[features].values
y_raw = df[[target_col]].values  

# === Normalize ===
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# === window ===
def create_sequences(X, y, input_len, target_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - input_len - target_len):
        X_seq.append(X[i:i+input_len])
        y_avg = np.mean(y[i+input_len:i+input_len+target_len])
        y_seq.append(y_avg)
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, input_window, target_window)

# === put data to GPU device ===
# only support fp32, do not change to torch.float16 and other width.
X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)  

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Loaded {len(X_tensor)} samples to device: {device}")


# === construct LSTM model ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x) 
        last_step = out[:, -1, :]
        return self.fc(last_step)

# === optimizer ===
input_size = X_tensor.shape[2]  
model = LSTMRegressor(input_size=input_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# === Training ===
num_epochs = 250

loss_history = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    loss_history.append(avg_loss)

    if epoch == num_epochs - 1:
        save_path = "./checkpoi" \
        "nt/lstm{inputw}min_checkpoint_epoch{nepoch}.pth".format(
            nepoch=num_epochs,
            inputw = input_window
        )
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, save_path)
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, marker='o')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        pic_name = "./lossPic/loss{inputw}min_curve.png".format(
            inputw = input_window
            )
        plt.savefig(pic_name)
        print("Loss curve saved.")

