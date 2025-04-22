import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==== configuration ====
model_path = "./checkpoint/lstm15min_checkpoint_epoch150.pth"
test_csv_path = "./dataset/test_pca_fixed.csv"
input_window = 15
target_window = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== model ====
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==== dataloader and preprocessing ====
df = pd.read_csv(test_csv_path)

features = ['T1', 'T2', 'T3', 'T4', 'PC1', 'PC2', 'PC3']
target_col = 'total_cpu_rate'

X_raw = df[features].values
y_raw = df[[target_col]].values 

# normalize
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# window
def create_sequences(X, y, input_len, target_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - input_len - target_len):
        X_seq.append(X[i:i+input_len])
        y_avg = np.mean(y[i+input_len:i+input_len+target_len])
        y_seq.append(y_avg)
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, input_window, target_window)

# to tensor
X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)

# ==== load model to cuda device ====
model = LSTMRegressor(input_size=X_tensor.shape[2]).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ==== predict ====
with torch.no_grad():
    y_pred = model(X_tensor).cpu().numpy()
    y_true = y_tensor.cpu().numpy()

y_pred_inv = y_scaler.inverse_transform(y_pred)
y_true_inv = y_scaler.inverse_transform(y_true)

# ==== calculate MSE RMSE and MAE ====
mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_inv, y_pred_inv)

print(f"Evaluation on test data:")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")

result_df = pd.DataFrame({
    "true": y_true_inv.flatten(),
    "pred": y_pred_inv.flatten()
})
result_df.to_csv("./evaluation/prediction_result.csv", index=False)
print("Prediction results saved to 'prediction_result.csv'")
