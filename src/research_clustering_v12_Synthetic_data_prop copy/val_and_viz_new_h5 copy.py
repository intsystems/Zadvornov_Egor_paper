#!/usr/bin/env python
"""
Скрипт с увеличенным горизонтом предсказания (5 точек) для анализа поведения моделей на более длинной перспективе
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
import warnings
warnings.filterwarnings('ignore')  # Suppress Prophet and PLS warnings

def convert_to_proportions(values):
    """Convert array of values to proportions that sum to 1."""
    total = np.sum(values)
    if total == 0:
        return np.zeros_like(values)  # avoid division by zero
    return values / total

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    return {'mse': mse, 'mae': mae}

def fit_prophet_safely(df, periods=1):
    """Safely fit Prophet with fallback to weighted average."""
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
        # Use weighted average of last 5 values as fallback
        weights = np.array([0.4, 0.25, 0.15, 0.12, 0.08])  # More recent values have higher weight
        if len(df['y']) >= 5:
            return [np.average(df['y'].tail(5), weights=weights)] * periods
        else:
            return [df['y'].mean()] * periods

def fit_pls_safely(X_tr, y_tr, X_te, n_components, return_model=False):
    """
    Fit PLS с попытками уменьшить n_components при ошибках.
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
test_size = 5  # пять точек вперёд (10 недель)

# Word frequency matrix: Time × Topic
word_matrix = pd.pivot_table(
    df, values='Count', index='Time', columns='Topic', aggfunc='sum'
).fillna(0)

# Cluster matrix: Time × Cluster
cluster_matrix = pd.pivot_table(
    df, values='Count', index='Time', columns='Cluster', aggfunc='sum'
).fillna(0)

# Train/test split для сырых значений
X_raw = word_matrix.values                # (num_dates, num_words)
y_raw = cluster_matrix.values             # (num_dates, num_clusters)

# Convert both features and targets to proportions
X_props = np.array([convert_to_proportions(X_raw[i]) for i in range(X_raw.shape[0])])  # word proportions
y_props = np.array([convert_to_proportions(y_raw[i]) for i in range(y_raw.shape[0])])  # cluster proportions

# Split into train/test для raw значений
X_train_raw = X_raw[:-test_size]
y_train_raw = y_raw[:-test_size]
X_test_raw = X_raw[-test_size:]
y_test_raw = y_raw[-test_size:]

# Split для proportions
X_train_props = X_props[:-test_size]
y_train_props = y_props[:-test_size]
X_test_props = X_props[-test_size:]
y_test_props = y_props[-test_size:]

# Стандартизация признаков для raw значений
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# ================
# H1: Feature Selection через PLS (используем raw значения)
# ================
print("\n=== Hypothesis 1: Feature Selection ===")
results = []

n_components = min(3, X_train_scaled.shape[1] - 1)
# 1) Full model (все признаки)
y_pred_full, pls_model = fit_pls_safely(
    X_train_scaled, y_train_raw, X_test_scaled, n_components, return_model=True
)
full_metrics = evaluate_model(y_test_raw, y_pred_full, "Full Model (All Features, Raw)")
results.append({"method": "Full Model (Raw)", **full_metrics})

# 2) Рассчитываем importance признаков по |coef|
coef_abs = np.abs(pls_model.coef_)
if coef_abs.shape[0] == X_train_scaled.shape[1]:
    importance = coef_abs.sum(axis=1)   # (n_features,)
else:
    importance = coef_abs.sum(axis=0)   # (n_features,)

top_k = int(len(vocab) * 0.2)           # 20% от всех слов
selected_idx   = np.argsort(importance)[::-1][:top_k]
selected_vocab = [vocab[i] for i in selected_idx]

# 3) PLS только на отобранных признаках (raw значения)
X_train_sel_scaled = X_train_scaled[:, selected_idx]
X_test_sel_scaled  = X_test_scaled[:,  selected_idx]

# Для H2 нам понадобятся пропорции для отобранных признаков
X_train_sel_props = X_train_props[:, selected_idx]

pls_selected = PLSRegression(
    n_components=min(3, X_train_sel_scaled.shape[1] - 1),
    scale=False
)
pls_selected.fit(X_train_sel_scaled, y_train_raw)
y_pred_selected = pls_selected.predict(X_test_sel_scaled)
selected_metrics = evaluate_model(y_test_raw, y_pred_selected, "Selected Features Model (Raw)")
results.append({"method": "Selected Features (Raw)", **selected_metrics})

# ================
# H2: Word-level Prediction → Cluster (используем пропорции)
# ================
print("\n=== Hypothesis 2: Word-level vs Direct Prediction (Proportions) ===")

# 1) Direct Prophet на пропорциях кластеров
cluster_predictions = np.zeros((test_size, len(clusters)))
for i, cluster in enumerate(clusters):
    df_cluster = pd.DataFrame({
        'ds': dates[:-test_size],
        'y': y_train_props[:, i]  # используем пропорции
    })
    cluster_predictions[:, i] = fit_prophet_safely(df_cluster, periods=test_size)

# 2) Word-level Prophet по selected_vocab (тоже пропорции)
word_predictions = np.zeros((test_size, len(selected_vocab)))
for i, word in enumerate(selected_vocab):
    df_word = pd.DataFrame({
        'ds': dates[:-test_size],
        'y': X_train_sel_props[:, i]  # используем пропорции отобранных слов
    })
    word_predictions[:, i] = fit_prophet_safely(df_word, periods=test_size)

# 3) Агрегируем прогнозы слов → кластеры и проверяем суммы
print("\nDebug sums before aggregation:")
print(f"Sum of word predictions t+1: {word_predictions[0].sum():.3f}")
print(f"Sum of word predictions t+2: {word_predictions[1].sum()}")

cluster_from_words = np.zeros((test_size, len(clusters)))

# Сначала собираем слова по кластерам
cluster_word_mapping = {}
for ci, cluster in enumerate(clusters):
    idxs = [
        idx_sel for idx_sel, w in enumerate(selected_vocab)
        if df[df['Topic'] == w]['Cluster'].iloc[0] == cluster
    ]
    cluster_word_mapping[cluster] = idxs

# Выводим информацию по каждому кластеру до нормализации
print("\nPre-normalization analysis:")
for cluster, idxs in cluster_word_mapping.items():
    ci = list(clusters).index(cluster)
    cluster_from_words[:, ci] = word_predictions[:, idxs].sum(axis=1) if len(idxs) > 0 else 0
    
    print(f"\nCluster: {cluster}")
    print(f"Number of selected words: {len(idxs)}")
    for t in range(test_size):
        print(f"\nt+{t+1}:")
        print(f"True proportion: {y_test_props[t, ci]:.3f}")
        print(f"Direct Prophet: {cluster_predictions[t, ci]:.3f}")
        print(f"Word-based sum: {cluster_from_words[t, ci]:.3f}")
    
    if idxs:
        print("\nContributing words:")
        for idx_sel in idxs:
            w = selected_vocab[idx_sel]
            print(f"\n{w}:")
            for t in range(test_size):
                print(f"t+{t+1}: {word_predictions[t, idx_sel]:.3f}")

print("\nSums across clusters before normalization:")
for t in range(test_size):
    print(f"t+{t+1}: {cluster_from_words[t].sum():.3f}")

# Нормализуем предсказания, чтобы сумма по кластерам = 1
print("\nAfter normalization:")
for t in range(test_size):
    cluster_from_words[t] = convert_to_proportions(cluster_from_words[t])
print("\nFinal normalized proportions:")
for t in range(test_size):
    print(f"\nTime point t+{t+1}, sum={cluster_from_words[t].sum():.3f}:")
    for ci, cluster in enumerate(clusters):
        print(f"  {cluster:<6s}: {cluster_from_words[t, ci]:.3f}")

def evaluate_model_detailed(y_true, y_pred, model_name):
    # Проверяем входные данные
    print(f"\n{model_name} Input Validation:")
    print(f"Shape y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    print("\nTime point sums:")
    for t in range(test_size):
        print(f"t+{t+1}:")
        print(f"y_true sum: {y_true[t].sum():.3f}")
        print(f"y_pred sum: {y_pred[t].sum():.3f}")
    
    # Нормализуем предсказания в пропорции
    y_pred_norm = np.array([convert_to_proportions(y_pred[i]) for i in range(y_pred.shape[0])])
    
    # Масштабируем относительно исходного диапазона значений
    def scale_to_01(x):
        # Используем теоретический диапазон для пропорций [0, 1]
        # и добавляем небольшой отступ для численной стабильности
        eps = 1e-6
        global_min = 0 - eps
        global_max = 1 + eps
        return (x - global_min) / (global_max - global_min)
    
    y_true_scaled = scale_to_01(y_true)
    y_pred_scaled = scale_to_01(y_pred_norm)
    
    print("\nAfter scaling to [0,1]:")
    print(f"Global range: [{min(y_true.min(), y_pred_norm.min()):.3f}, {max(y_true.max(), y_pred_norm.max()):.3f}]")
    print(f"y_true range: [{y_true_scaled.min():.3f}, {y_true_scaled.max():.3f}]")
    print(f"y_pred range: [{y_pred_scaled.min():.3f}, {y_pred_scaled.max():.3f}]")
    
    # Проверяем сохранение относительных разниц
    print("\nValidating relative differences preservation:")
    orig_diffs = np.abs(y_true - y_pred_norm)
    scaled_diffs = np.abs(y_true_scaled - y_pred_scaled)
    print(f"Original max diff: {orig_diffs.max():.6f}")
    print(f"Scaled max diff: {scaled_diffs.max():.6f}")
    correlation = np.corrcoef(orig_diffs.flatten(), scaled_diffs.flatten())[0,1]
    print(f"Correlation between original and scaled differences: {correlation:.6f}")
    
    # Считаем метрики на масштабированных данных
    mse = mean_squared_error(y_true_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Детальный анализ ошибок
    print("\nDetailed error analysis by cluster:")
    cluster_names = ['Sports', 'Music', 'Tech']
    for t in range(y_true.shape[0]):
        print(f"\nTime point t+{t+1}:")
        for c in range(y_true.shape[1]):
            error = y_pred_norm[t,c] - y_true[t,c]
            rel_error = abs(error / y_true[t,c]) * 100 if y_true[t,c] != 0 else float('inf')
            print(f"  {cluster_names[c]:<6s}: true={y_true[t,c]:.4f}, pred={y_pred_norm[t,c]:.4f}, "
                  f"error={error:.4f} ({rel_error:.1f}%)")
    
    # Возвращаем метрики для дальнейшего анализа
    return {'mse': mse, 'mae': mae}

# Вычисляем и анализируем метрики
direct_metrics = evaluate_model_detailed(y_test_props, cluster_predictions, "Direct Cluster Prediction (Proportions)")
word_based_metrics = evaluate_model_detailed(y_test_props, cluster_from_words, "Word-based Prediction (Proportions)")

# Добавляем наивный метод (использование последнего известного значения)
naive_predictions = np.zeros_like(cluster_predictions)
for i in range(len(clusters)):
    naive_predictions[:, i] = y_train_props[-1, i]

# Оцениваем все методы
print("\nEvaluating all methods:")
naive_metrics = evaluate_model_detailed(y_test_props, naive_predictions, "Naive Last Value Method")
direct_metrics = evaluate_model_detailed(y_test_props, cluster_predictions, "Direct Cluster Prediction")
word_based_metrics = evaluate_model_detailed(y_test_props, cluster_from_words, "Word-based Prediction")

# Считаем улучшение относительно наивного метода
print("\nImprovement over naive method:")
direct_vs_naive_mse = ((naive_metrics['mse'] - direct_metrics['mse']) / naive_metrics['mse']) * 100
word_vs_naive_mse = ((naive_metrics['mse'] - word_based_metrics['mse']) / naive_metrics['mse']) * 100
print(f"Direct method MSE improvement vs naive: {direct_vs_naive_mse:.1f}%")
print(f"Word-based MSE improvement vs naive: {word_vs_naive_mse:.1f}%")

# Считаем относительное улучшение word-based vs direct
rel_improvement_mse = ((direct_metrics['mse'] - word_based_metrics['mse']) / direct_metrics['mse']) * 100
rel_improvement_mae = ((direct_metrics['mae'] - word_based_metrics['mae']) / direct_metrics['mae']) * 100

print("\nWord-based vs Direct improvement:")
print(f"MSE improvement: {rel_improvement_mse:.1f}%")
print(f"MAE improvement: {rel_improvement_mae:.1f}%")

# Сохраняем результаты в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('hypothesis_test_results_h5.csv', index=False)
print("\nResults saved to hypothesis_test_results_h5.csv")

# ---------------------------------------------------------------------
# ЧАСТЬ 3: Визуализация предсказаний по каждому кластеру
# ---------------------------------------------------------------------

# Подготовка данных для визуализации
cluster_matrix_full = pd.pivot_table(
    df, values='Count', index='Time', columns='Cluster', aggfunc='sum'
).fillna(0)

# Конвертируем в доли
total_counts = cluster_matrix_full.sum(axis=1)
y_true_full = cluster_matrix_full.div(total_counts, axis=0).values

plt.figure(figsize=(15, 5*len(clusters)))
for ci, cluster in enumerate(clusters):
    plt.subplot(len(clusters), 1, ci+1)
    
    # Фильтруем слова для текущего кластера
    df_cluster = df[df["Cluster"] == cluster].copy()
    core_words = [t for t in df_cluster["Topic"].unique() if "_core" in t]
    noise_words = [t for t in df_cluster["Topic"].unique() if "_noise" in t]
    
    # Матрицы для core и noise слов
    mat_raw = df_cluster.pivot_table(
        index="Time", columns="Topic", values="Count", aggfunc="sum"
    ).fillna(0)
    
    # Конвертируем в доли относительно всех кластеров
    total_counts = df.pivot_table(
        index="Time", columns="Cluster", values="Count", aggfunc="sum"
    ).fillna(0).sum(axis=1)
    mat = mat_raw.div(total_counts, axis=0)
      # Суммируем core и noise отдельно
    if core_words:
        core_sum = mat[core_words].sum(axis=1)
        plt.fill_between(mat.index, 0, core_sum.values, 
                        color='lightgreen', alpha=0.3, label='Core words sum')
    
    if noise_words:
        noise_sum = mat[noise_words].sum(axis=1)
        plt.fill_between(mat.index, 0, noise_sum.values,
                        color='gray', alpha=0.2, label='Noise words sum')
    
    # Истинные значения для всех методов
    plt.plot(dates, y_true_full[:, ci],
             label='Actual values', color='red', linestyle='-', linewidth=2)
    plt.plot(dates, core_sum.values,
             label='Word-based method', color='green', linestyle='-', linewidth=2)
    plt.plot(dates[:-test_size], y_props[:-test_size, ci],
             label='Direct method', color='blue', linestyle='-', linewidth=2)
    
    # Предсказания (test)
    plt.plot(dates[-test_size:], cluster_predictions[:, ci],
             label='Direct (predicted)', color='blue', linestyle='--', linewidth=2)
    plt.plot(dates[-test_size:], cluster_from_words[:, ci],
             label='Word-based (predicted)', color='green', linestyle='--', linewidth=2)
    
    plt.title(f'Cluster: {cluster} - Proportions Over Time (5-point horizon)')
    plt.ylabel('Proportion')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_predictions_comparison_h5.png', dpi=150)
print("✓ saved cluster_predictions_comparison_h5.png")
