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

# === 配置参数 ===
csv_path = "./dataset/train_pca.csv"  # ← 替换为你的文件名
input_window = 15  # 输入历史长度（分钟）
target_window = 5  # 预测未来多长时间（分钟）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("./lossPic", exist_ok=True)

# === Step 1: 读取数据 ===
df = pd.read_csv(csv_path)
features = ['T1', 'T2', 'T3', 'T4', 'PC1', 'PC2', 'PC3']
target_col = 'total_cpu_rate'

X_raw = df[features].values
y_raw = df[[target_col]].values  # 需要保持二维 shape

# === Step 2: 标准化 ===
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# === Step 3: 构造输入窗口 + 未来平均值目标 ===
def create_sequences(X, y, input_len, target_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - input_len - target_len):
        X_seq.append(X[i:i+input_len])
        y_avg = np.mean(y[i+input_len:i+input_len+target_len])
        y_seq.append(y_avg)
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, input_window, target_window)

# === Step 4: 转为 tensor 并放入 GPU ===
X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)  # 加维度变成 (N, 1)

# === Step 5: 创建 DataLoader（可选）===
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Loaded {len(X_tensor)} samples to device: {device}")


# === Step 6: 定义 LSTM 模型 ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出一个标量

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        last_step = out[:, -1, :]  # 取最后一步的输出
        return self.fc(last_step)

# === Step 7: 初始化模型与优化器 ===
input_size = X_tensor.shape[2]  # 即 feature 数量（7）
model = LSTMRegressor(input_size=input_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === Step 8: 训练模型 ===
num_epochs = 150

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
        print("✅ Loss 曲线已保存为 'loss_curve.png'")

