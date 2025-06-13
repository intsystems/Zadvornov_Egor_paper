#!/usr/bin/env python
"""
Показывает, почему выполняются H1 и H2.
Сохраняет одну фигуру hypotheses_overview.png (4 панели).


Поподробнее:
Что хотим увидеть	                                            Почему это подтверждает гипотезу
1. Core-слова ≈ кластер без шума	                            Если убрать noise-слова, ряд становится гораздо «чище», значит отбор признаков полезен (H1).
2. Noise-слова ≈ белый шум	                                    Их вклад раздувает совокупный кластер ⇒ direct-Prophet промахивается (H2).
3. Корреляция “core ↔ cluster» высокая, “noise ↔ cluster” ≈ 0	PLS правильно выберет core-слова (H1).
4. Когерентность word-прогнозов	                                Когда суммируем хорошие прогнозы по core-словам, получается точнее (H2).


4 панели:
A. Heat-map корреляций (core vs noise vs cluster).
– Core-ячейки ярко-красные, noise вокруг нуля.

B. Cуммарная траектория кластера
– две линии: cluster (все слова) и cluster-без-noise (только core).
– Видно, как шум «распушивает» серию.

C. Boxplot MAE Prophet-кластер vs Prophet-core-сумма
– На валидации (train-test split из вашего скрипта) показываем, что word-based MAE меньше.

D. PLS-коэффициенты
– Только top-10 слов: core-столбцы высокие, noise ≈ 0.

"""

import os, warnings, itertools, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
plt.rc("font", size=9)
CSV = "synthetic_data_confirming.csv"
OUT = "hypotheses_overview.png"
TEST_H = 2            # горизонт предсказания Prophet
TOP_SHOW = 10         # сколько слов отобразить в PLS bar
OUT_PNG  = "hypotheses_overview.png"

# ---------- load ---------------------------------------------------
df       = pd.read_csv(CSV, parse_dates=["Time"])
CLUSTER  = "Sports"            # один кластер достаточно
cl_df    = df[df["Cluster"] == CLUSTER].copy()

core   = [t for t in cl_df["Topic"].unique() if "_core"  in t]
noise  = [t for t in cl_df["Topic"].unique() if "_noise" in t]
topics = core + noise          # ← все слова

# pivot: Time × Topic
mat            = cl_df.pivot_table(index="Time", columns="Topic",
                                   values="Count", aggfunc="sum").fillna(0)
cluster_series       = mat.sum(axis=1)
cluster_core_only    = mat[core].sum(axis=1)

# ---------- figure layout -----------------------------------------
fig = plt.figure(figsize=(11, 8))
gs  = fig.add_gridspec(2, 2)

# A. correlation heat-map ------------------------------------------
ax = fig.add_subplot(gs[0, 0])
corr = mat.assign(**{CLUSTER: cluster_series}).corr()

sel_core  = core
sel_noise = noise[:len(core)]
sel_cols  = sel_core + [CLUSTER] + sel_noise

c = ax.imshow(corr.loc[sel_cols, sel_cols],
              cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(sel_cols)))
ax.set_xticklabels(sel_cols, rotation=90, ha="right")
ax.set_yticks(range(len(sel_cols)))
ax.set_yticklabels(sel_cols)
ax.set_title("A. Pearson correlation (core vs noise)")
fig.colorbar(c, ax=ax, fraction=0.045)

# B. cluster trajectory --------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.plot(cluster_series.index,      cluster_series.values,
        label="Cluster (ALL words)", lw=1.6)
ax.plot(cluster_core_only.index,   cluster_core_only.values,
        label="Cluster w/o noise",  ls="--", lw=1.6)
ax.set_title("B. Noise inflates cluster trajectory")
ax.legend()

# ---------- Prophet MAE panel -------------------------------------
train_idx = slice(None, -TEST_H)
test_idx  = slice(-TEST_H, None)

# direct Prophet on full cluster
m_dir = Prophet(yearly_seasonality=True)
m_dir.fit(cluster_series.reset_index()
                        .rename(columns={"Time": "ds", 0: "y"}))
fut_dir   = m_dir.make_future_dataframe(periods=TEST_H, freq="2W")
direct_pred = m_dir.predict(fut_dir)["yhat"].values[-TEST_H:]

# word-level Prophet on *all* words
word_preds = np.zeros((TEST_H, len(topics)))
for j, w in enumerate(topics):
    mw = Prophet(yearly_seasonality=True)
    mw.fit(mat[w].reset_index().rename(columns={"Time": "ds", w: "y"}))
    fw = mw.make_future_dataframe(periods=TEST_H, freq="2W")
    word_preds[:, j] = mw.predict(fw)["yhat"].values[-TEST_H:]

sum_word_pred = word_preds.sum(axis=1)

mae_direct = mean_absolute_error(cluster_series.values[test_idx], direct_pred)
mae_word   = mean_absolute_error(cluster_series.values[test_idx], sum_word_pred)

ax = fig.add_subplot(gs[1, 0])
ax.bar(["Direct\n(cluster)", "Sum of\nALL words"],
       [mae_direct, mae_word],
       color=["tab:orange", "tab:green"])
ax.set_title("C. Prophet MAE on test horizon")
ax.set_ylabel("MAE")
ax.set_ylim(0, max(mae_direct, mae_word) * 1.25)
for i, v in enumerate([mae_direct, mae_word]):
    ax.text(i, v * 1.05, f"{v:.1f}", ha="center")

# ---------- PLS importance ----------------------------------------
scaler  = StandardScaler(with_mean=False).fit(mat.values)
Xsc     = scaler.transform(mat.values)
pls     = PLSRegression(n_components=3).fit(Xsc[train_idx],
                                            cluster_series.values[train_idx, None])
coefs   = pd.Series(np.abs(pls.coef_).reshape(-1), index=mat.columns)
top_imp = coefs.nlargest(TOP_SHOW)

ax = fig.add_subplot(gs[1, 1])
colors = ["tab:green" if t in core else "tab:gray" for t in top_imp.index[::-1]]
ax.barh(top_imp.index[::-1], top_imp.values[::-1], color=colors)
ax.set_title("D. |PLS coefficient| (top 10)")
ax.set_xlabel("Importance")
ax.set_xlim(0, top_imp.values.max() * 1.1)

# ---------- save ---------------------------------------------------
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print("✓ saved", OUT_PNG)
