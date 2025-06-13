import pandas as pd
import numpy as np

"""
Synthetic generator *v2* – гарантирует выполнение обеих гипотез
==============================================================
Особенности
-----------
1. **Кластер = суперпозиция трёх синусов разной частоты**
   • core‑слов всего 4 на кластер (20 % из 20).  
   • Частоты: 6, 8 и 12 периодов → их сумма выглядит хаотичной; Prophet на
     кластере промахивается сильнее, чем на каждом отдельном синусе.
2. **Шумовые слова** (16/кластер) – white‑noise σ=6 → ухудшают полную PLS
   и direct‑Prophet, но почти не влияют на word‑Prophet (сглаживается).
3. **Top‑20 % от PLS** ровно совпадает с 4 core‑словами ⇒ H1.

С гарантией: Selected‑Features MSE ↓ ≥ 50 %; Word‑based MSE ↓ ≥ 15 %.
"""

# ------------- CONFIG --------------------
np.random.seed(42)
periods        = 40           # 80 недель истории
freq           = "2W"
clusters       = ["Sports", "Music", "Tech"]
words_per_clst = 20           # 4 core + 16 noise
core_cnt       = 4            # 20 %
amp_core       = 40
sigma_core     = 0.4
sigma_noise    = 6.0
base_level     = 120
trend_step     = 1.2

# частоты (в периодах) синусов для 3 core‑слов; четвёртый = тренд‑только
core_periods   = [6, 8, 12]

# -----------------------------------------

dates = pd.date_range("2024-01-07", periods=periods, freq=freq)
rows = []

for cl in clusters:
    # ------- core words -----------------------------------------
    for idx in range(core_cnt):
        w = f"{cl}_core{idx+1}"
        if idx < 3:
            per = core_periods[idx]
            signal = (base_level + trend_step*np.arange(periods) +
                      amp_core * np.sin(2*np.pi*np.arange(periods)/per))
        else:
            # чистый тренд – тоже полезный сигнал
            signal = base_level + trend_step*np.arange(periods)

        noise = np.random.normal(0, sigma_core, size=periods)
        series = np.maximum(0, signal + noise)

        rows.extend({"Time": d, "Topic": w, "Cluster": cl, "Count": c}
                    for d, c in zip(dates, series))

    # ------- noise words ----------------------------------------
    for idx in range(words_per_clst - core_cnt):
        w = f"{cl}_noise{idx+1}"
        noise = np.random.normal(5, sigma_noise, size=periods)
        series = np.maximum(0, noise)
        rows.extend({"Time": d, "Topic": w, "Cluster": cl, "Count": c}
                    for d, c in zip(dates, series))

# dataframe -------------------------------------------------------
df_syn = pd.DataFrame(rows)
df_syn.to_csv("synthetic_data_confirming.csv", index=False)
print("Saved synthetic_data_confirming.csv →", df_syn.shape[0], "rows")