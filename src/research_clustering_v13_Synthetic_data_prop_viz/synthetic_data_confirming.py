import pandas as pd
import numpy as np

"""
Synthetic generator *v2h-param*
===============================
• 4 core-слова / кластер: сумма синусов + тренд
• Шумовые слова (параметризуются):
    – white noise        : N(μ, σ)
    – spike+decay        : hyperbolic peak at t0  →  exponential falloff
• Top-20 % PLS = core-слова ⇒ гарантирует H1, H2
"""

# ---------- CONFIG -------------------------------------------------
np.random.seed(42)
periods        = 40           # 80 недель (freq = 2W)
freq           = "2W"
clusters       = ["Sports", "Music", "Tech"]

# --- структура слов ------------------------------------------------
core_cnt        = 4           # 4 core words per cluster
white_noise_cnt = 8           # <-- меняйте здесь
spike_decay_cnt = 8           # <-- и здесь (общее 16 noise слов)

# --- core-параметры -----------------------------------------------
amp_core    = 20
sigma_core  = 0.4
base_level  = 220
trend_step  = 7.2
core_periods = [5, 6, 12]

# --- noise-параметры ----------------------------------------------
sigma_noise = 6.0

# hyperbolic-spike + exp-decay
A_spike = 180      # высота пика
k_exp   = 0.5     # скорость эксп. затухания

# -------------------------------------------------------------------
dates = pd.date_range("2024-01-07", periods=periods, freq=freq)
rows = []
t = np.arange(periods)

for cl in clusters:
    # ===== core words =============================================
    for i in range(core_cnt):
        w = f"{cl}_core{i+1}"
        if i < 3:
            per = core_periods[i]
            sig = (base_level +
                   trend_step * t +
                   amp_core * np.sin(2 * np.pi * t / per))
        else:                       # 4-й core = чистый тренд
            sig = base_level + trend_step * t
        series = np.maximum(0, sig + np.random.normal(0, sigma_core, size=periods))
        rows.extend({"Time": d, "Topic": w, "Cluster": cl, "Count": v}
                    for d, v in zip(dates, series))

    # ===== 1) white-noise =========================================
    for i in range(white_noise_cnt):
        w = f"{cl}_noiseWN{i+1}"
        series = np.maximum(0, np.random.normal(5, sigma_noise, size=periods))
        rows.extend({"Time": d, "Topic": w, "Cluster": cl, "Count": v}
                    for d, v in zip(dates, series))

    # ===== 2) spike + decay =======================================
    for i in range(spike_decay_cnt):
        w = f"{cl}_noiseSD{i+1}"
        t0 = np.random.randint(5, periods - 5)     # позиция пика
        spike = A_spike / (np.abs(t - t0) + 1)     # гиперболический пик
        decay = A_spike * np.exp(-k_exp * (t - t0))
        decay[t < t0] = 0                          # затухание только после t0
        sig = spike + decay
        series = np.maximum(0, sig + np.random.normal(0, sigma_noise, size=periods))
        rows.extend({"Time": d, "Topic": w, "Cluster": cl, "Count": v}
                    for d, v in zip(dates, series))

# ---------- save ---------------------------------------------------
df_syn = pd.DataFrame(rows)
df_syn.to_csv("synthetic_data_confirming.csv", index=False)
print("Saved synthetic_data_confirming.csv →", df_syn.shape[0], "rows")
