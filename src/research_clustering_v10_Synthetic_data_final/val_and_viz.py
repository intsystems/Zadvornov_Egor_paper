#!/usr/bin/env python
"""
Объединённый скрипт из двух частей:
 1) Валидация гипотез H1 и H2 (код из val.ipynb)
 2) Визуализация результатов (4 панели, hypotheses_overview.png),
    причём для «панели C» используем результаты, уже вычисленные в части 1.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress Prophet and PLS warnings

# ---------------------------------------------------------------------
# ЧАСТЬ 1: Валидация гипотез H1 и H2 (взят из val.ipynb)
# ---------------------------------------------------------------------

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    return {'mse': mse, 'mae': mae}

def fit_prophet_safely(df, periods=1):
    """Safely fit Prophet: fallback к среднему, если учёт не сходится."""
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            mcmc_samples=0
        )
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq='2W')
        forecast = model.predict(future)
        return forecast['yhat'].values[-periods:]
    except:
        return [df['y'].tail(5).mean()] * periods

def fit_pls_safely(X_tr, y_tr, X_te, n_components, return_model=False):
    """
    Fit PLS с «попытками» уменьшить n_components при ошибках.
    В fallback возвращает среднее y_tr.
    """
    try:
        eps = 1e-10
        X_tr_stable = X_tr + eps
        X_te_stable = X_te + eps

        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_tr_stable, y_tr)
        if return_model:
            return pls.predict(X_te_stable), pls
        return pls.predict(X_te_stable)
    except:
        try:
            n_components_new = max(1, n_components - 1)
            print(f"Retrying PLS with {n_components_new} components...")
            pls = PLSRegression(n_components=n_components_new, scale=False)
            pls.fit(X_tr_stable, y_tr)
            if return_model:
                return pls.predict(X_te_stable), pls
            return pls.predict(X_te_stable)
        except:
            print("PLS fallback to mean prediction...")
            mean_pred = np.tile(np.mean(y_tr, axis=0), (X_te.shape[0], 1))
            if return_model:
                return mean_pred, None
            return mean_pred

print("Loading synthetic data for validation...")
df = pd.read_csv('synthetic_data_confirming.csv', parse_dates=['Time'])

# --- Подготовка матриц ---
vocab    = df['Topic'].unique()
clusters = df['Cluster'].unique()
dates    = df['Time'].unique()
num_dates = len(dates)
test_size = 2  # две точки вперёд (4 недели)

# Word frequency matrix: Time × Topic
word_matrix = pd.pivot_table(
    df, values='Count', index='Time', columns='Topic', aggfunc='sum'
).fillna(0)

# Cluster matrix: Time × Cluster
cluster_matrix = pd.pivot_table(
    df, values='Count', index='Time', columns='Cluster', aggfunc='sum'
).fillna(0)

# Train/test split
X = word_matrix.values                # (num_dates, num_words)
y = cluster_matrix.values             # (num_dates, num_clusters)
X_train = X[:-test_size]
y_train = y[:-test_size]
X_test = X[-test_size:]
y_test = y[-test_size:]

# Стандартизация признаков
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ================
# H1: Feature Selection через PLS
# ================
print("\n=== Hypothesis 1: Feature Selection ===")
results = []

n_components = min(3, X_train_scaled.shape[1] - 1)
# 1) Full model (все признаки)
y_pred_full, pls_model = fit_pls_safely(
    X_train_scaled, y_train, X_test_scaled, n_components, return_model=True
)
full_metrics = evaluate_model(y_test, y_pred_full, "Full Model (All Features)")
results.append({"method": "Full Model", **full_metrics})

# 2) Рассчитываем importance признаков по |coef|
coef_abs = np.abs(pls_model.coef_)
if coef_abs.shape[0] == X_train_scaled.shape[1]:
    importance = coef_abs.sum(axis=1)   # (n_features,)
else:
    importance = coef_abs.sum(axis=0)   # (n_features,)

top_k = int(len(vocab) * 0.2)           # 20% от всех слов
selected_idx   = np.argsort(importance)[::-1][:top_k]
selected_vocab = [vocab[i] for i in selected_idx]

# 3) PLS только на отобранных признаках
X_train_sel_scaled = X_train_scaled[:, selected_idx]
X_test_sel_scaled  = X_test_scaled[:,  selected_idx]
X_train_sel_orig   = X_train[:, selected_idx]  # для Prophet нужны нефасшированные ряды

pls_selected = PLSRegression(
    n_components=min(3, X_train_sel_scaled.shape[1] - 1),
    scale=False
)
pls_selected.fit(X_train_sel_scaled, y_train)
y_pred_selected = pls_selected.predict(X_test_sel_scaled)
selected_metrics = evaluate_model(y_test, y_pred_selected, "Selected Features Model")
results.append({"method": "Selected Features", **selected_metrics})

# ================
# H2: Word-level Prediction → Cluster
# ================
print("\n=== Hypothesis 2: Word-level vs Direct Prediction ===")

# 1) Direct Prophet на агрегате (все кластеры сразу)
cluster_predictions = np.zeros((test_size, len(clusters)))
for i, cluster in enumerate(clusters):
    df_cluster = pd.DataFrame({
        'ds': dates[:-test_size],
        'y': y_train[:, i]
    })
    cluster_predictions[:, i] = fit_prophet_safely(df_cluster, periods=test_size)

direct_metrics = evaluate_model(y_test, cluster_predictions, "Direct Cluster Prediction")
results.append({"method": "Direct Cluster", **direct_metrics})

# 2) Word-level Prophet по selected_vocab (нефасшированные ряды)
word_predictions = np.zeros((test_size, len(selected_vocab)))
for i, word in enumerate(selected_vocab):
    df_word = pd.DataFrame({
        'ds': dates[:-test_size],
        'y': X_train_sel_orig[:, i]
    })
    word_predictions[:, i] = fit_prophet_safely(df_word, periods=test_size)

# 3) Агрегируем прогнозы слов → кластеры
cluster_from_words = np.zeros((test_size, len(clusters)))
for ci, cluster in enumerate(clusters):
    idxs = [
        idx_sel for idx_sel, w in enumerate(selected_vocab)
        if df[df['Topic'] == w]['Cluster'].iloc[0] == cluster
    ]

    # Отладочный вывод: True vs Direct vs Word-based
    print(f"\nCluster: {cluster}")
    print(f"  True values    t+1={y_test[0, ci]:.2f}   t+2={y_test[1, ci]:.2f}")
    print(f"  Direct Prophet t+1={cluster_predictions[0, ci]:.2f}   "
          f"t+2={cluster_predictions[1, ci]:.2f}")

    if idxs:
        cluster_from_words[:, ci] = word_predictions[:, idxs].sum(axis=1)
        print(f"  Word-based     t+1={cluster_from_words[0, ci]:.2f}   "
              f"t+2={cluster_from_words[1, ci]:.2f}")
        print("  Selected words forecasts:")
        for idx_sel in idxs:
            w = selected_vocab[idx_sel]
            preds = word_predictions[:, idx_sel]
            print(f"    {w:<25s}  t+1={preds[0]:.2f}   t+2={preds[1]:.2f}")
    else:
        cluster_from_words[:, ci] = np.mean(y_train[:, ci])
        print("  Word-based     — нет выбранных слов, использовано среднее train")

word_based_metrics = evaluate_model(
    y_test, cluster_from_words, "Word-based Prediction (Selected Words)"
)
results.append({"method": "Word-based", **word_based_metrics})

# Сохраняем результаты части 1
results_df = pd.DataFrame(results)
results_df.to_csv('hypothesis_test_results.csv', index=False)
print("\nResults saved to hypothesis_test_results.csv")

plt.figure(figsize=(12, 6))
plt.bar(results_df['method'], results_df['mse'])
plt.title('MSE Comparison Across Methods')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('hypothesis_comparison.png')
plt.close()
print("Bar chart saved as hypothesis_comparison.png")

# ---------------------------------------------------------------------
# ЧАСТЬ 2: Визуализация «hypotheses overview» (4 панели)
#           (используем данные, посчитанные выше, где это возможно)
# ---------------------------------------------------------------------

OUT_PNG = "hypotheses_overview.png"
TEST_H  = test_size   # уже равен 2
TOP_SHOW= 15
CLUSTER = "Sports"    # фиксируем один кластер для панелей

# 1) Фильтрация по CLUSTER
df_vis    = df[df["Cluster"] == CLUSTER].copy()
core      = [t for t in df_vis["Topic"].unique() if "_core" in t]
noise     = [t for t in df_vis["Topic"].unique() if "_noise" in t]
topics    = core + noise

# Матрица Time × Topic для выбранного кластера
mat       = df_vis.pivot_table(
    index="Time", columns="Topic", values="Count", aggfunc="sum"
).fillna(0)

cluster_series    = mat.sum(axis=1)    # суммарный ряд по всем словам кластера
cluster_core_only = mat[core].sum(axis=1)  # суммарный ряд только core-слов

# Индексы train/test внутри mat
train_idx = slice(None, -TEST_H)
test_idx  = slice(-TEST_H, None)

# ---------------------------------------------------
# A: Heat-map корреляций (core vs noise vs cluster)
# ---------------------------------------------------
fig = plt.figure(figsize=(11, 8))
gs  = fig.add_gridspec(2, 2)


ax = fig.add_subplot(gs[0, 0])

# Список всех слов (только в рамках данного кластера)
all_topics = mat.columns.tolist()

# Делим шумихи на два типа по имени: white-noise и spike+decay
white_noise_topics = [t for t in all_topics if "_noiseWN" in t]
spike_decay_topics = [t for t in all_topics if "_noiseSD" in t]

# Возьмём по len(core) случайных (или просто первых) представителей каждого типа
n_show = len(core)
sel_white = white_noise_topics[:n_show]
sel_spike = spike_decay_topics[:n_show]

# Итоговый список колонок: core-слова, кластер, затем white-noise и spike+decay
sel_cols = core + [CLUSTER] + sel_white + sel_spike

corr = mat.assign(**{CLUSTER: cluster_series}).corr()
c = ax.imshow(corr.loc[sel_cols, sel_cols],
              cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(sel_cols)))
ax.set_xticklabels(sel_cols, rotation=90, ha="right")
ax.set_yticks(range(len(sel_cols)))
ax.set_yticklabels(sel_cols)
ax.set_title("A. Pearson correlation (core vs noiseWN vs noiseSD)")
fig.colorbar(c, ax=ax, fraction=0.045)


# ---------------------------------------------------
# B: Суммарная траектория кластера (all vs core-only)
# ---------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.plot(cluster_series.index,      cluster_series.values,
        label="Cluster (ALL words)", lw=1.6)
ax.plot(cluster_core_only.index,   cluster_core_only.values,
        label="Cluster w/o noise",  ls="--", lw=1.6)
ax.set_title("B. Noise inflates cluster trajectory")
ax.legend()

# ---------------------------------------------------
# C: MAE Direct Prophet vs Prophet(sum of selected words)
#    Используем результаты из части 1, а не пересчитываем заново.
# ---------------------------------------------------
# 1) Найдём индекс CLUSTER в общем списке clusters
cluster_idx = list(clusters).index(CLUSTER)

# direct_pred_vis — прогноз Prophet на агрегате (взято из cluster_predictions)
direct_pred_vis = cluster_predictions[:, cluster_idx]

# sum_word_pred_vis — прогноз Word-based (сумма по selected_vocab → cluster)
#   взят из cluster_from_words[:, cluster_idx]
sum_word_pred_vis = cluster_from_words[:, cluster_idx]

mae_direct_vis = mean_absolute_error(cluster_series.values[test_idx], direct_pred_vis)
mae_word_vis   = mean_absolute_error(cluster_series.values[test_idx], sum_word_pred_vis)

ax = fig.add_subplot(gs[1, 0])
ax.bar(
    ["Direct\n(cluster)", "Sum of\nselected words"],
    [mae_direct_vis, mae_word_vis],
    color=["tab:orange", "tab:green"]
)
ax.set_title("C. Prophet MAE on test horizon")
ax.set_ylabel("MAE")
ax.set_ylim(0, max(mae_direct_vis, mae_word_vis) * 1.25)
for i, v in enumerate([mae_direct_vis, mae_word_vis]):
    ax.text(i, v * 1.05, f"{v:.1f}", ha="center")

# ---------------------------------------------------
# D: PLS-коэффициенты (|coef|) для топ-10 слов (пересчитываем PLS на мат)
# ---------------------------------------------------
# Для визуализации PLS отдельно строим модель на одном кластере:
scaler_vis = StandardScaler(with_mean=False).fit(mat.values)
Xsc_vis    = scaler_vis.transform(mat.values)
pls_vis    = PLSRegression(n_components=3).fit(
    Xsc_vis[train_idx], cluster_series.values[train_idx, None]
)
coefs_vis  = pd.Series(
    np.abs(pls_vis.coef_).reshape(-1),
    index=mat.columns
)
top_imp_vis = coefs_vis.nlargest(TOP_SHOW)

ax = fig.add_subplot(gs[1, 1])
colors = ["tab:green" if t in core else "tab:gray" for t in top_imp_vis.index[::-1]]
ax.barh(top_imp_vis.index[::-1], top_imp_vis.values[::-1], color=colors)
ax.set_title("D. |PLS coefficient| (top 10)")
ax.set_xlabel("Importance")
ax.set_xlim(0, top_imp_vis.values.max() * 1.1)

# ---------------------------------------------------
# Сохраняем окончательный рисунок
# ---------------------------------------------------
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print("✓ saved", OUT_PNG)
