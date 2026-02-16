import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# - NASA POWER (метео: T2M, PRECTOTCORR)
# - MODIS/Sentinel (NDVI, EVI)
# - Eurostat (историческая урожайность)

def generate_synthetic_data(num_regions=50, years_per_region=10, seq_len=36):
    """
    Сейчас будем использовать данные, имитирующие сезон вегетации (36 декад), датасет на ПК.
    Features: Temperature, Precipitation, NDVI.
    Target: Yield (ц/га).
    """
    np.random.seed(42)
    
    data = []
    targets = []
    meta_info = [] # Для визуализации по регионам
    
    # Имитация региональных различий (базовая урожайность)
    base_yields = np.random.uniform(20, 60, num_regions)
    
    for region_id in range(num_regions):
        for year in range(years_per_region):
            # 1. Генерация временных рядов
            time_steps = np.arange(seq_len)
            
            # Температура: Сезонный паттерн (холодно -> тепло -> холодно)
            temp_pattern = 15 + 15 * np.sin(np.pi * time_steps / seq_len)
            temp_noise = np.random.normal(0, 2, seq_len)
            temp = temp_pattern + temp_noise
            
            # Осадки: Случайные дожди
            precip_pattern = np.random.exponential(scale=10, size=seq_len)
            
            # NDVI: Рост и спад вегетации (зависит от погоды)
            # NDVI зависит от осадков и температуры (упрощенная модель)
            ndvi_base = 0.2 + 0.6 * np.sin(np.pi * time_steps / seq_len) # Базовый рост
            # Влияние осадков на NDVI (с лагом)
            ndvi = ndvi_base + np.roll(precip_pattern, -2) * 0.01 
            ndvi = np.clip(ndvi, 0, 1) # NDVI от 0 до 1
            
            # Объединяем признаки [Seq_Len, Features]
            features = np.stack([temp, precip_pattern, ndvi], axis=1)
            
            # 2. Генерация целевой переменной (Урожайность)
            # Урожайность = Базовая + Влияние погоды + Шум
            # Засуха (мало осадков) снижает урожай
            precip_effect = (np.sum(precip_pattern) / 1000) * 10 
            # Высокие температуры (теплового стресса) снижают
            temp_stress = -1 * (np.max(temp) - 30) if np.max(temp) > 30 else 0
            
            yield_val = base_yields[region_id] + precip_effect + temp_stress + np.random.normal(0, 2)
            yield_val = max(5, yield_val) # Минимальный порог
            
            data.append(features)
            targets.append(yield_val)
            meta_info.append({'region_id': region_id, 'year': 2000 + year})
            
    return np.array(data), np.array(targets), pd.DataFrame(meta_info)

class YieldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Архитектура модели (CNN-LSTM Hybrid)
# ==========================================

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, cnn_out=64, lstm_hidden=64, num_layers=1):
        super(CNN_LSTM_Model, self).__init__()
        
        # 1. CNN блок: извлечение локальных признаков
        # Conv1d ожидает (Batch, Channels, Seq_Len), поэтому permute в forward
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out),
            nn.MaxPool1d(kernel_size=2) # Уменьшение длины последовательности
        )
        
        # 2. LSTM блок: моделирование временных зависимостей
        self.lstm = nn.LSTM(
            input_size=cnn_out, 
            hidden_size=lstm_hidden, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 3. Полносвязный слой (Regressor)
        self.fc = nn.Linear(lstm_hidden, 1)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features] -> need [Batch, Features, Seq_Len] for Conv1d
        x = x.permute(0, 2, 1) 
        
        # CNN
        x = self.cnn(x) # Out: [Batch, CNN_Out, Seq_Len/2]
        
        # Подготовка к LSTM: [Batch, Seq_Len/2, CNN_Out]
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Берем последнее скрытое состояние
        last_hidden = hn[-1] 
        
        # Regressor
        out = self.fc(last_hidden)
        return out

# ==========================================
# 3. Обучение и оценка
# ==========================================

def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return total_loss / len(loader), mae, rmse, np.array(all_preds), np.array(all_targets)

# ==========================================
# 4. Основной запуск
# ==========================================

if __name__ == "__main__":
    # 1. Подготовка данных
    print("Generating synthetic data...")
    X_raw, y_raw, meta_df = generate_synthetic_data(num_regions=30, years_per_region=15)
    
    # Нормализация данных (важно для LSTM)
    # Reshape для scaler: (Samples * Timesteps, Features)
    original_shape = X_raw.shape
    X_flat = X_raw.reshape(-1, original_shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(original_shape)
    
    # Разделение Train/Test
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_scaled, y_raw, meta_df, test_size=0.2, random_state=42
    )
    
    train_dataset = YieldDataset(X_train, y_train)
    test_dataset = YieldDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Настройка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_Model(input_dim=3).to(device) # 3 фичи: Temp, Precip, NDVI
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Обучение
    epochs = 50
    history = {'train_loss': [], 'test_loss': []}
    
    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, mae, rmse, preds, targets = evaluate_model(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # 4. Визуализация результатов
    print("\nPlotting results...")
    
    # График 1: Прогноз vs Факт (Scatter)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel("Actual Yield (c/ha)")
    plt.ylabel("Predicted Yield (c/ha)")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    
    # График 2: Урожайность по регионам (Bar plot)
    plt.subplot(1, 2, 2)
    results_df = meta_test.copy()
    results_df['Actual'] = targets.flatten()
    results_df['Predicted'] = preds.flatten()
    
    # Усредним по регионам для наглядности
    region_means = results_df.groupby('region_id').mean().reset_index()
    
    x = np.arange(len(region_means))
    width = 0.35
    
    plt.bar(x - width/2, region_means['Actual'], width, label='Actual', color='skyblue')
    plt.bar(x + width/2, region_means['Predicted'], width, label='Predicted', color='orange')
    plt.xticks(x, region_means['region_id'], rotation=45)
    plt.xlabel("Region ID")
    plt.ylabel("Yield (c/ha)")
    plt.title("Mean Yield by Region (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()
