import time
import random
import numpy as np
import pandas as pd
import joblib
import psutil
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Sabit tohum (seed) ayarları
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. Veri Setini Yükleme ve Bölme (70/30 stratified, random_state=42)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

# Modellerin tanımlanması
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED)
}

accuracies = []
metrics_data = []

os.makedirs("lab01/results", exist_ok=True)
process = psutil.Process(os.getpid())

for name, model in models.items():
    # --- EĞİTİM SÜRESİ VE BELLEK (1 warm-up, 5 tekrarın medyanı) ---
    train_times = []
    train_memories = []

    # Warm-up (1 kez)
    model.fit(X_train, y_train)

    for _ in range(5):
        mem_before = process.memory_info().rss
        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        end_time = time.perf_counter()
        mem_after = process.memory_info().rss

        train_times.append(end_time - start_time)
        train_memories.append(mem_after / (1024 * 1024)) # MB cinsinden

    median_train_time = np.median(train_times)
    peak_train_mem = np.max(train_memories)

    # Test Doğruluğu
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    accuracies.append({"Model": name, "Accuracy": round(acc, 4)})

    # --- TEKİL ÖRNEKLEM ÇIKARIMI (100 tekrar, medyan) ---
    sample = X_test[0:1]
    inf_times = []

    # Warm-up çıkarım
    model.predict(sample)

    for _ in range(100):
        start_inf = time.perf_counter()
        model.predict(sample)
        end_inf = time.perf_counter()
        inf_times.append((end_inf - start_inf) * 1000) # milisaniye (ms)

    median_inf_latency = np.median(inf_times)

    # --- MODEL BOYUTU ---
    model_path = f"lab01/results/model_{name}.joblib"
    joblib.dump(model, model_path)
    model_size_bytes = os.path.getsize(model_path)
    model_size_kb = model_size_bytes / 1024

    metrics_data.append({
        "Model": name,
        "Training_Time_s": round(median_train_time, 4),
        "Inference_Latency_ms": round(median_inf_latency, 4),
        "Model_Size_KB": round(model_size_kb, 2),
        "Peak_Train_Memory_MB": round(peak_train_mem, 2)
    })

# Sonuçların CSV olarak kaydedilmesi
df_acc = pd.DataFrame(accuracies)
df_acc.to_csv("lab01/results/baseline_accuracy.csv", index=False)

df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv("lab01/results/system_costs.csv", index=False)

print("Lab 1 ölçümleri başarıyla tamamlandı ve sonuçlar kaydedildi!")
print(df_metrics)
