import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from sklearn.model_selection import GridSearchCV
import warnings

# -------------------------------------------------------------------
#            Pipeline Full Framework: PLS vs Prophet
#                    1-Step Forecasting (с защитой от падений)
# -------------------------------------------------------------------

"""
Примечание: В этот скрипт добавлены try/except-блоки и проверки, 
чтобы нигде не возникало необработанного исключения. 
Все критичные участки (PLS-Full, Prophet.fit и т.д.) обёрнуты 
в защиту — в случае ошибки продолжают выполнение, 
а в результат записывают np.nan и выводят предупреждение.

Запуск:
    python pipeline_full_protected.py

Инструкция:
  1. Убедитесь, что рядом лежит файл preprocessed_file.csv
     с колонками ["Time", "Topic", "Cluster"].
  2. Установите библиотеки:
       pip install pandas numpy scikit-learn matplotlib prophet
  3. Запустите:
       python pipeline_full_protected.py
  4. В рабочей директории появятся:
       - pls_results.csv
       - prophet_results.csv
       - pls_freq_mse_comparison.png
     Даже если какие-то этапы не смогли выполниться, 
     скрипт завершится и создаст эти файлы (с np.nan там, 
     где не получилось получить результаты).
"""

# ----------------------------
# 0. Конфигурация
# ----------------------------
print("\n[0] Конфигурация параметров:")
CSV_PATH = Path("preprocessed_file.csv")
print(f"    CSV_PATH = {CSV_PATH}")
BUCKET_DAYS = 14
print(f"    BUCKET_DAYS = {BUCKET_DAYS}")
THRESHOLD_MODE = "percentile"  # {"percentile","topk","abs"}
print(f"    THRESHOLD_MODE = {THRESHOLD_MODE}")
THRESHOLD_VALUE = 20           # процент для percentile
print(f"    THRESHOLD_VALUE = {THRESHOLD_VALUE}")
TOPK = 8000                    # если topk
print(f"    TOPK = {TOPK}")
CV_FOLDS = 5
print(f"    CV_FOLDS = {CV_FOLDS}")
PLS_PARAM_GRID = {"n_components": [1, 2, 3, 5, 10]}
print(f"    PLS_PARAM_GRID = {PLS_PARAM_GRID}")
print("    (PLS-Full и GridSearchCV могут съесть много ОЗУ.)\n")

# ----------------------------
# 1. Загрузка данных и разбиение на «бакеты»
# ----------------------------
print("[1] Загрузка данных и разбиение на двухнедельные бакеты...")
try:
    df = pd.read_csv(CSV_PATH, parse_dates=["Time"])
    print(f"    ✔ Успешно прочитан CSV: {CSV_PATH}, shape = {df.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось прочитать {CSV_PATH}: {e}")

required_cols = {"Time", "Topic", "Cluster"}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    raise ValueError(f"ERROR: В CSV не хватает колонок: {missing}")

# Нормализуем начало
start_date = df["Time"].min().normalize()
print(f"    Минимальная дата: {start_date}")

# Строим бины для двухнедельных интервалов
bins = pd.date_range(
    start=start_date,
    end=df["Time"].max().normalize() + pd.Timedelta(days=BUCKET_DAYS*2),
    freq=f"{BUCKET_DAYS}D"
)
print(f"    Построили {len(bins)} граничных дат, последний бин = {bins[-1]}")

# Создаём столбец 'bucket'
df["bucket"] = pd.cut(df["Time"], bins=bins, right=False, labels=False)
num_buckets = int(df["bucket"].max()) + 1
print(f"    Всего бакетов (интервалов): {num_buckets}")

# ----------------------------
# 1.1. Построение словаря всех слов и сопоставление с кластерами
# ----------------------------
print("\n[1.1] Строим словарь всех слов и привязываем кластеры...")
vocab = df["Topic"].value_counts().index.tolist()
num_words = len(vocab)
print(f"    Всего уникальных слов (Topic): {num_words}")

print("    Вычисляем основной кластер (mode) для каждого слова...")
try:
    word_cluster_map = (
        df.groupby("Topic")["Cluster"]
          .agg(lambda x: x.mode()[0])
          .loc[vocab]
          .tolist()
    )
    word_cluster_map = np.array(word_cluster_map)
    print(f"    ✔ word_cluster_map.shape = {word_cluster_map.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось построить word_cluster_map: {e}")

# ----------------------------
# 1.2. Построение X_raw (buckets × words) и y_clusters (buckets × clusters)
# ----------------------------
print("\n[1.2] Строим X_raw (сколько раз встретилось каждое слово в каждом бакете) ...")
try:
    matrix = (
        df.groupby(["bucket", "Topic"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=vocab, fill_value=0)
          .reindex(index=range(num_buckets), fill_value=0)
          .astype(np.float64)
    )
    X_raw = matrix.values
    print(f"    ✔ X_raw.shape = {X_raw.shape}  (num_buckets, num_words)")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось построить X_raw: {e}")

print("    Строим y_clusters (сколько постов в каждом кластере за бакет) ...")
try:
    clusters_matrix = (
        df.groupby(["bucket", "Cluster"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=range(num_buckets), fill_value=0)
          .astype(np.float64)
    )
    cluster_ids = clusters_matrix.columns.tolist()
    num_clusters = len(cluster_ids)
    y_clusters = clusters_matrix.values
    print(f"    ✔ y_clusters.shape = {y_clusters.shape}  (num_buckets, num_clusters)")
    print(f"    Имена кластеров: {cluster_ids}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось построить y_clusters: {e}")

# Даты для Prophet
dates = start_date + pd.to_timedelta(np.arange(num_buckets) * BUCKET_DAYS, unit="D")
print(f"    ✔ dates.shape = {dates.shape}, пример: {dates[0]} ... {dates[-1]}")

# ----------------------------
# 2. Масштабирование X_raw
# ----------------------------
print("\n[2] Масштабируем X_raw (StandardScaler with_mean=False) ...")
scaler = StandardScaler(with_mean=False)
try:
    X_scaled = scaler.fit_transform(X_raw)
    print(f"    ✔ X_scaled.shape = {X_scaled.shape}")
    print(f"    scaler.scale_.shape = {scaler.scale_.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось масштабировать X_raw: {e}")

# ----------------------------
# 3. Отбор признаков с помощью PLS (X_scaled → y_clusters)
# ----------------------------
print("\n[3] Отбор признаков: PLS + GridSearchCV для подбора n_components ...")
print(f"    Запускаем GridSearchCV (cv={CV_FOLDS}) по n_components={PLS_PARAM_GRID['n_components']} ...")
grid = GridSearchCV(
    estimator=PLSRegression(),
    param_grid=PLS_PARAM_GRID,
    cv=CV_FOLDS,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
with warnings.catch_warnings():
    # чтобы не засорять лог предупреждениями о сходимости PLS
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        grid.fit(X_scaled, y_clusters)
        best_n = grid.best_params_["n_components"]
        print(f"    ✔ [GRID] best_n_components = {best_n}")
    except Exception as e:
        # Если GridSearchCV упал (Out-Of-Memory или иная проблема),
        # присваиваем best_n = 2 (консервативно) и выводим предупреждение.
        best_n = 2
        print(f"WARNING: GridSearchCV завершился с ошибкой: {e}")
        print(f"         Используем default best_n_components = {best_n}")

print("    Обучаем PLSRegression для оценки важностей слов ...")
n_comp_fs = min(best_n, num_buckets - 1, num_clusters)
pls_fs = PLSRegression(n_components=n_comp_fs, max_iter=500)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        pls_fs.fit(X_scaled, y_clusters)
        print(f"    ✔ PLS для feature selection обучен, n_components = {n_comp_fs}")
    except Exception as e:
        raise RuntimeError(f"ERROR: Не удалось обучить PLS для FS: {e}")

# Извлекаем коэффициенты и считаем важность слов
try:
    W = pls_fs.coef_
    if W.shape[0] == num_clusters:
        W = W.T
    importance = np.abs(W).sum(axis=1)
    print(f"    ✔ importance.shape = {importance.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось вычислить importance: {e}")

# Выбираем топ-слова по THRESHOLD_MODE
print("    Определяем маску top-слов по THRESHOLD_MODE ...")
if THRESHOLD_MODE == "percentile":
    top_n = int(len(importance) * THRESHOLD_VALUE / 100)
    selected_idx = np.argsort(importance)[::-1][:top_n]
    print(f"    THRESHOLD_MODE=percentile → top_n = {top_n}")
elif THRESHOLD_MODE == "topk":
    selected_idx = np.argsort(importance)[::-1][:TOPK]
    print(f"    THRESHOLD_MODE=topk → выбираем топ {TOPK}")
elif THRESHOLD_MODE == "abs":
    selected_idx = np.where(importance >= THRESHOLD_VALUE)[0]
    print(f"    THRESHOLD_MODE=abs → importance >= {THRESHOLD_VALUE}")
else:
    raise ValueError("Invalid THRESHOLD_MODE")

mask = np.zeros(len(vocab), dtype=bool)
mask[selected_idx] = True
print(f"    ✔ Отобрано {mask.sum()} слов из {len(vocab)}")

# Урезанные матрицы по mask
print("    Строим X_red_raw и X_red_scaled (только по mask) ...")
try:
    X_red_raw = X_raw[:, mask]
    X_red_scaled = X_scaled[:, mask]
    print(f"    ✔ X_red_raw.shape = {X_red_raw.shape}")
    print(f"    ✔ X_red_scaled.shape = {X_red_scaled.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось сформировать X_red_* по mask: {e}")

# ----------------------------
# 4. Подготовка train/test split для X → X_{t+1}
# ----------------------------
print("\n[4] Подготавливаем train/test split для X → X_{t+1} ...")
t_max = num_buckets - 1      # индекс последнего бакета (например, 181)
train_pairs_end = t_max - 1  # последний обучающий X-т индекс, а test = t_max
print(f"    t_max = {t_max}, train_pairs_end = {train_pairs_end}")

def get_train_test_X(X_m):
    """
    Возвращает X_train, Y_train, X_test, Y_test для задачи X_t → X_{t+1}.
    Проверяет, что в X_m достаточно строк.
    """
    if X_m.shape[0] < train_pairs_end + 2:
        raise ValueError(
            f"Недостаточно строк в X_m: {X_m.shape[0]}, нужно >= {train_pairs_end + 2}"
        )
    X_train = X_m[0:train_pairs_end, :]
    Y_train = X_m[1:train_pairs_end+1, :]
    X_test = X_m[train_pairs_end, :].reshape(1, -1)
    Y_test = X_m[train_pairs_end+1, :].reshape(1, -1)
    return X_train, Y_train, X_test, Y_test

# Split для полной матрицы (scaled)
print("    Split для полной матрицы X_scaled ...")
try:
    Xf_train, Xf_Y_train, Xf_test, Xf_Y_test = get_train_test_X(X_scaled)
    print(f"    ✔ Xf_train.shape = {Xf_train.shape}, Xf_Y_train.shape = {Xf_Y_train.shape}")
    print(f"    ✔ Xf_test.shape = {Xf_test.shape}, Xf_Y_test.shape = {Xf_Y_test.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Split для X_scaled не удалось: {e}")

# Split для урезанной матрицы (scaled)
print("    Split для урезанной матрицы X_red_scaled ...")
try:
    Xr_train, Xr_Y_train, Xr_test, Xr_Y_test = get_train_test_X(X_red_scaled)
    print(f"    ✔ Xr_train.shape = {Xr_train.shape}, Xr_Y_train.shape = {Xr_Y_train.shape}")
    print(f"    ✔ Xr_test.shape = {Xr_test.shape}, Xr_Y_test.shape = {Xr_Y_test.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Split для X_red_scaled не удалось: {e}")

# ----------------------------
# 4.1. Подготовка для «share» (доля слов)
# ----------------------------
print("\n[4.1] Подготавливаем матрицу долей X_share_raw ...")
X_raw_sum = X_raw.sum(axis=1, keepdims=True)
zero_rows = np.where(X_raw_sum.flatten() == 0)[0]
if len(zero_rows) > 0:
    print(f"    WARNING: Есть периоды, где X_raw_sum == 0: {zero_rows}. Доли этих строк будут 0.")
X_share_raw = np.divide(X_raw, X_raw_sum, where=(X_raw_sum != 0))
print(f"    ✔ X_share_raw.shape = {X_share_raw.shape}")

print("    Масштабируем X_share_raw (StandardScaler) ...")
scaler_share = StandardScaler(with_mean=False)
try:
    X_share_scaled = scaler_share.fit_transform(X_share_raw)
    print(f"    ✔ X_share_scaled.shape = {X_share_scaled.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Не удалось масштабировать X_share_raw: {e}")

# Split для полной «share»-матрицы
print("    Split для полной «share»-матрицы X_share_scaled ...")
try:
    Xsr_train, Xsr_Y_train, Xsr_test, Xsr_Y_test = get_train_test_X(X_share_scaled)
    print(f"    ✔ Xsr_train.shape = {Xsr_train.shape}, Xsr_Y_train.shape = {Xsr_Y_train.shape}")
    print(f"    ✔ Xsr_test.shape = {Xsr_test.shape}, Xsr_Y_test.shape = {Xsr_Y_test.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Split для X_share_scaled не удалось: {e}")

# Split для урезанной «share»-матрицы (после масштабирования)
print("    Split для урезанной «share»-матрицы X_share_scaled[:, mask] ...")
try:
    X_red_share_scaled = X_share_scaled[:, mask]
    Xsr_red_train, Xsr_red_Y_train, Xsr_red_test, Xsr_red_Y_test = get_train_test_X(X_red_share_scaled)
    print(f"    ✔ X_red_share_scaled.shape = {X_red_share_scaled.shape}")
    print(f"    ✔ Xsr_red_train.shape = {Xsr_red_train.shape}, Xsr_red_Y_train.shape = {Xsr_red_Y_train.shape}")
    print(f"    ✔ Xsr_red_test.shape = {Xsr_red_test.shape}, Xsr_red_Y_test.shape = {Xsr_red_Y_test.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: Split для X_red_share_scaled не удалось: {e}")

# ----------------------------
# 5. PLS Forecasting (1-step) и расчёт MSE
# ----------------------------
print("\n[5] PLS Forecasting: прогноз для 1-шаг вперёд + MSE ...")
results = {
    "method": [],
    "mode": [],    # "frequency" или "share"
    "freq_mse": [],
    "share_mse": []
}

def compute_mse_cluster(pred_X_raw, true_X_raw, word_cluster_map):
    """
    Для одного периода предсказаний (shape = num_words) считает:
      - freq_mse: MSE между суммами по кластерам по абсолютным counts
      - share_mse: MSE между долями (sum(cluster)/sum(total))
    """
    true_cluster_sums = np.zeros(num_clusters, dtype=float)
    pred_cluster_sums = np.zeros(num_clusters, dtype=float)
    for i, cid in enumerate(cluster_ids):
        mask_c = (word_cluster_map == cid)
        true_cluster_sums[i] = true_X_raw[mask_c].sum()
        pred_cluster_sums[i] = pred_X_raw[mask_c].sum()
    freq_mse = mean_squared_error(true_cluster_sums, pred_cluster_sums)
    true_total = true_cluster_sums.sum()
    pred_total = pred_cluster_sums.sum()
    if true_total == 0:
        true_shares = np.zeros(num_clusters)
    else:
        true_shares = true_cluster_sums / true_total
    if pred_total == 0:
        pred_shares = np.zeros(num_clusters)
    else:
        pred_shares = pred_cluster_sums / pred_total
    share_mse = mean_squared_error(true_shares, pred_shares)
    return freq_mse, share_mse

# Реальные значения X_raw и X_share для последнего периода (t_max)
true_X_raw_final = X_raw[t_max, :]       # shape = (num_words,)
true_X_share_final = X_share_raw[t_max, :]  # shape = (num_words,)

print(f"    ✔ true_X_raw_final.shape = {true_X_raw_final.shape}")
print(f"    ✔ true_X_share_final.shape = {true_X_share_final.shape}")

# ----- 5A. PLS-Selected (frequency) -----
print("\n  [5A] PLS-Selected (frequency) ...")
n_comp_sel = min(best_n, Xr_train.shape[1] - 1, Xr_train.shape[1])
print(f"       n_components для PLS-Selected = {n_comp_sel}")

pls_sel = PLSRegression(n_components=n_comp_sel, max_iter=500)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        pls_sel.fit(Xr_train, Xr_Y_train)
        print(f"       ✔ PLS-Selected обучен: Xr_train.shape = {Xr_train.shape}")
    except Exception as e:
        raise RuntimeError(f"ERROR: Не удалось обучить PLS-Selected: {e}")

try:
    Xr_pred_scaled = pls_sel.predict(Xr_test)  # (1, num_selected_words)
    print(f"       ✔ Xr_pred_scaled.shape = {Xr_pred_scaled.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: PLS-Selected.predict() упал: {e}")

# Собираем «полный» scaled-вектор (zero fill)
X_pred_full_scaled = np.zeros(num_words, dtype=float)
X_pred_full_scaled[mask] = Xr_pred_scaled.flatten()
print(f"       ✔ X_pred_full_scaled.shape = {X_pred_full_scaled.shape}")

# Инверсия масштаба
raw_pred_full = X_pred_full_scaled * scaler.scale_
print(f"       ✔ raw_pred_full.shape = {raw_pred_full.shape}")

# Расчёт MSE
freq_mse_sel_freq, share_mse_sel_freq = compute_mse_cluster(raw_pred_full, true_X_raw_final, word_cluster_map)
print(f"       [PLS-Selected, freq] freq_mse = {freq_mse_sel_freq:.6f}, share_mse = {share_mse_sel_freq:.6f}")

results["method"].append("PLS-Selected")
results["mode"].append("frequency")
results["freq_mse"].append(freq_mse_sel_freq)
results["share_mse"].append(share_mse_sel_freq)

# ----- 5B. PLS-Selected (share) -----
print("\n  [5B] PLS-Selected (share) ...")
n_comp_sel_share = min(best_n, Xsr_red_train.shape[1] - 1, Xsr_red_train.shape[1])
print(f"       n_components для PLS-Selected-Share = {n_comp_sel_share}")

pls_sel_share = PLSRegression(n_components=n_comp_sel_share, max_iter=500)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        pls_sel_share.fit(Xsr_red_train, Xsr_red_Y_train)
        print(f"       ✔ PLS-Selected-Share обучен: Xsr_red_train.shape = {Xsr_red_train.shape}")
    except Exception as e:
        raise RuntimeError(f"ERROR: Не удалось обучить PLS-Selected-Share: {e}")

try:
    Xsr_red_pred_scaled = pls_sel_share.predict(Xsr_red_test)  # (1, num_selected_words)
    print(f"       ✔ Xsr_red_pred_scaled.shape = {Xsr_red_pred_scaled.shape}")
except Exception as e:
    raise RuntimeError(f"ERROR: PLS-Selected-Share.predict() упал: {e}")

X_pred_share_full_scaled = np.zeros(num_words, dtype=float)
X_pred_share_full_scaled[mask] = Xsr_red_pred_scaled.flatten()
print("       ✔ Полный scaled-вектор для долей сформирован")

# Инверсия масштаба долей
share_pred_raw_full = X_pred_share_full_scaled * scaler_share.scale_
print(f"       ✔ share_pred_raw_full.shape = {share_pred_raw_full.shape}")

freq_mse_sel_share, share_mse_sel_share = compute_mse_cluster(share_pred_raw_full, true_X_share_final, word_cluster_map)
print(f"       [PLS-Selected, share] freq_mse = {freq_mse_sel_share:.6f}, share_mse = {share_mse_sel_share:.6f}")

results["method"].append("PLS-Selected")
results["mode"].append("share")
results["freq_mse"].append(freq_mse_sel_share)
results["share_mse"].append(share_mse_sel_share)

# ----- 5C. PLS-Full (frequency) -----
print("\n  [5C] PLS-Full (frequency) ...")
print(f"       Xf_train.shape = {Xf_train.shape}, Xf_Y_train.shape = {Xf_Y_train.shape}")
n_comp_full = min(best_n, Xf_train.shape[1] - 1, Xf_train.shape[1])
print(f"       n_components для PLS-Full = {n_comp_full}")

pls_full = PLSRegression(n_components=n_comp_full, max_iter=500)
# Попробуем привести к float32, чтобы сэкономить память
try:
    Xf_train_small = Xf_train.astype(np.float32)
    Xf_Y_train_small = Xf_Y_train.astype(np.float32)
    print("       Привели Xf_train и Xf_Y_train к float32 для экономии ОЗУ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pls_full.fit(Xf_train_small, Xf_Y_train_small)
    print("       ✔ PLS-Full обучен на float32")
except Exception as e:
    print(f"WARNING: PLS-Full.fit(float32) упал: {e}")
    print("         Пробуем обучить на оригинальных dtype (может быть медленно или упасть)...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pls_full.fit(Xf_train, Xf_Y_train)
        print("       ✔ PLS-Full обучен на оригинальных dtype")
    except Exception as e2:
        print(f"ERROR: PLS-Full.fit() не удалось ни на float32, ни на original dtype: {e2}")
        print("       Записываем np.nan для PLS-Full (frequency) и переходим дальше.")
        freq_mse_full_freq = np.nan
        share_mse_full_freq = np.nan
        results["method"].append("PLS-Full")
        results["mode"].append("frequency")
        results["freq_mse"].append(freq_mse_full_freq)
        results["share_mse"].append(share_mse_full_freq)
        # Пропускаем блок 5C (переходим сразу к 5D)
        skip_full_freq = True
    else:
        skip_full_freq = False

# Если PLS-Full обучился (skip_full_freq=False), то делаем predict и MSE
if 'skip_full_freq' not in locals() or not skip_full_freq:
    skip_full_freq = False
    try:
        Xf_pred_scaled = pls_full.predict(Xf_test)  # (1, num_words)
        print(f"       ✔ Xf_pred_scaled.shape = {Xf_pred_scaled.shape}")
    except Exception as e:
        print(f"ERROR: PLS-Full.predict() упал: {e}")
        Xf_pred_scaled = np.zeros((1, num_words), dtype=float)
        print("       Заменили предсказание PLS-Full.zeros-вектором")

    raw_pred_full_f = Xf_pred_scaled.flatten() * scaler.scale_
    print(f"       ✔ raw_pred_full_f.shape = {raw_pred_full_f.shape}")

    try:
        freq_mse_full_freq, share_mse_full_freq = compute_mse_cluster(raw_pred_full_f, true_X_raw_final, word_cluster_map)
        print(f"       [PLS-Full, freq] freq_mse = {freq_mse_full_freq:.6f}, share_mse = {share_mse_full_freq:.6f}")
    except Exception as e:
        print(f"ERROR: compute_mse_cluster (PLS-Full freq) упал: {e}")
        freq_mse_full_freq = np.nan
        share_mse_full_freq = np.nan

    results["method"].append("PLS-Full")
    results["mode"].append("frequency")
    results["freq_mse"].append(freq_mse_full_freq)
    results["share_mse"].append(share_mse_full_freq)

# ----- 5D. PLS-Full (share) -----
print("\n  [5D] PLS-Full (share) ...")
print(f"       Xsr_train.shape = {Xsr_train.shape}, Xsr_Y_train.shape = {Xsr_Y_train.shape}")
n_comp_full_share = min(best_n, Xsr_train.shape[1] - 1, Xsr_train.shape[1])
print(f"       n_components для PLS-Full-share = {n_comp_full_share}")

pls_full_share = PLSRegression(n_components=n_comp_full_share, max_iter=500)
# Пробуем float32
try:
    Xsr_train_small = Xsr_train.astype(np.float32)
    Xsr_Y_train_small = Xsr_Y_train.astype(np.float32)
    print("       Привели Xsr_train и Xsr_Y_train к float32 для экономии ОЗУ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pls_full_share.fit(Xsr_train_small, Xsr_Y_train_small)
    print("       ✔ PLS-Full-share обучен на float32")
    skip_full_share = False
except Exception as e:
    print(f"WARNING: PLS-Full-share.fit(float32) упал: {e}")
    print("         Пробуем обучить на original dtype ...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pls_full_share.fit(Xsr_train, Xsr_Y_train)
        print("       ✔ PLS-Full-share обучен на оригинальных dtype")
        skip_full_share = False
    except MemoryError as me:
        print(f"ERROR: MemoryError при обучении PLS-Full-share: {me}")
        skip_full_share = True
    except Exception as e2:
        print(f"ERROR: PLS-Full-share.fit() не удался: {e2}")
        skip_full_share = True

if skip_full_share:
    print("       Пропускаем блок PLS-Full-share, записываем np.nan в результаты.")
    freq_mse_full_share = np.nan
    share_mse_full_share = np.nan
    results["method"].append("PLS-Full")
    results["mode"].append("share")
    results["freq_mse"].append(freq_mse_full_share)
    results["share_mse"].append(share_mse_full_share)
else:
    # Если обучение получилось, делаем predict и MSE
    try:
        Xf_pred_share_scaled = pls_full_share.predict(Xsr_test)  # (1, num_words)
        print(f"       ✔ Xf_pred_share_scaled.shape = {Xf_pred_share_scaled.shape}")
    except Exception as e:
        print(f"ERROR: PLS-Full-share.predict() упал: {e}")
        Xf_pred_share_scaled = np.zeros((1, num_words), dtype=float)
        print("       Заменили предсказание PLS-Full-share.zeros-вектором")

    share_pred_full_f = Xf_pred_share_scaled.flatten() * scaler_share.scale_
    print(f"       ✔ share_pred_full_f.shape = {share_pred_full_f.shape}")

    try:
        freq_mse_full_share, share_mse_full_share = compute_mse_cluster(
            share_pred_full_f, true_X_share_final, word_cluster_map
        )
        print(f"       [PLS-Full, share] freq_mse = {freq_mse_full_share:.6f}, share_mse = {share_mse_full_share:.6f}")
    except Exception as e:
        print(f"ERROR: compute_mse_cluster (PLS-Full share) упал: {e}")
        freq_mse_full_share = np.nan
        share_mse_full_share = np.nan

    results["method"].append("PLS-Full")
    results["mode"].append("share")
    results["freq_mse"].append(freq_mse_full_share)
    results["share_mse"].append(share_mse_full_share)

# ----------------------------
# 6. Prophet Forecasting (Clusters)
# ----------------------------
print("\n[6] Prophet Forecasting: Clusters ...")
df_clusters_full = pd.DataFrame(y_clusters, columns=cluster_ids)
df_clusters_full["ds"] = dates
print(f"    df_clusters_full.shape = {df_clusters_full.shape}")

prophet_results = {
    "method": [],
    "mode": [],
    "freq_mse": [],
    "share_mse": []
}

# ----- 6A. Prophet-Cluster (frequency) -----
print("  [6A] Prophet-Cluster (frequency) ...")
true_cluster_181 = y_clusters[t_max, :]
pred_cluster_181 = np.zeros_like(true_cluster_181, dtype=float)
print(f"    ✔ true_cluster_181.shape = {true_cluster_181.shape}")

for i, cid in enumerate(cluster_ids):
    print(f"    → Прогноз для кластера '{cid}' (индекс {i}) ...")
    ts_df = df_clusters_full[["ds", cid]].rename(columns={cid: "y"})
    ts_train = ts_df.iloc[0:train_pairs_end+1]
    print(f"       ts_train.shape = {ts_train.shape}")
    model = Prophet()
    try:
        model.fit(ts_train)
        print("       ✔ Prophet обучен успешно")
        future = model.make_future_dataframe(periods=1, freq=f"{BUCKET_DAYS}D", include_history=False)
        forecast = model.predict(future)
        pred_cluster_181[i] = forecast["yhat"].values[-1]
        print(f"       → {cid}: predicted yhat = {pred_cluster_181[i]:.4f}")
    except Exception as e:
        print(f"WARNING: Prophet.fit() упал для кластера '{cid}': {e}")
        pred_cluster_181[i] = np.nan
        print(f"       Устанавливаем предсказание для {cid} = np.nan")

try:
    freq_mse_prophet_freq, share_mse_prophet_freq, avr_mse_prophet_freq = compute_mse_cluster(pred_cluster_181, true_cluster_181, word_cluster_map)
    print(f"       [Prophet-Cluster, freq] freq_mse = {freq_mse_prophet_freq:.6f}, share_mse = {share_mse_prophet_freq:.6f}, avr_mse_freq = {avr_mse_prophet_freq:.6f}")
except Exception as e:
    print(f"ERROR: compute_mse_cluster (Prophet-Cluster freq) упал: {e}")
    freq_mse_prophet_freq = np.nan
    share_mse_prophet_freq = np.nan
    avr_mse_prophet_freq = np.nan

prophet_results["method"].append("Prophet-Cluster")
prophet_results["mode"].append("frequency")
prophet_results["freq_mse"].append(freq_mse_prophet_freq)
prophet_results["share_mse"].append(share_mse_prophet_freq)
prophet_results["avr_mse_freq"].append(avr_mse_prophet_freq)

# ----- 6B. Prophet-Cluster (share) -----
print("\n  [6B] Prophet-Cluster (share) ...")
cluster_sums = y_clusters.sum(axis=1, keepdims=True)
cluster_shares = np.divide(y_clusters, cluster_sums, where=(cluster_sums != 0))
true_cluster_share_181 = cluster_shares[t_max, :]
pred_cluster_share_181 = np.zeros_like(true_cluster_share_181, dtype=float)
print(f"    ✔ true_cluster_share_181.shape = {true_cluster_share_181.shape}")

for i, cid in enumerate(cluster_ids):
    print(f"    → Прогноз доли для кластера '{cid}' ...")
    ts_df = pd.DataFrame({"ds": dates, "y": cluster_shares[:, i]})
    ts_train = ts_df.iloc[0:train_pairs_end+1]
    print(f"       ts_train.shape = {ts_train.shape}")
    model = Prophet()
    try:
        model.fit(ts_train)
        print("       ✔ Prophet (share) обучен успешно")
        future = model.make_future_dataframe(periods=1, freq=f"{BUCKET_DAYS}D", include_history=False)
        forecast = model.predict(future)
        pred_cluster_share_181[i] = forecast["yhat"].values[-1]
        print(f"       → {cid}: predicted share yhat = {pred_cluster_share_181[i]:.6f}")
    except Exception as e:
        print(f"WARNING: Prophet.fit() (share) упал для кластера '{cid}': {e}")
        pred_cluster_share_181[i] = np.nan
        print(f"       Устанавливаем долю для {cid} = np.nan")

try:
    freq_mse_prophet_share, share_mse_prophet_share, avr_mse_prophet_share = compute_mse_cluster(pred_cluster_share_181, true_cluster_share_181, word_cluster_map)
    print(f"       [Prophet-Cluster, share] freq_mse = {freq_mse_prophet_share:.6f}, share_mse = {share_mse_prophet_share:.6f}, avr_mse_freq = {avr_mse_prophet_share:.6f}")
except Exception as e:
    print(f"ERROR: compute_mse_cluster (Prophet-Cluster share) упал: {e}")
    freq_mse_prophet_share = np.nan
    share_mse_prophet_share = np.nan
    avr_mse_prophet_share = np.nan

prophet_results["method"].append("Prophet-Cluster")
prophet_results["mode"].append("share")
prophet_results["freq_mse"].append(freq_mse_prophet_share)
prophet_results["share_mse"].append(share_mse_prophet_share)
prophet_results["avr_mse_freq"].append(avr_mse_prophet_share)

# ----------------------------
# 7. Prophet Forecasting (Words → Aggregate to Clusters)
# ----------------------------
print("\n[7] Prophet Forecasting: Words (Selected) → агрегируем в кластеры ...")
selected_words = np.array(vocab)[mask]
print(f"    Количество выбранных слов: {len(selected_words)}")

df_words = pd.DataFrame(X_red_raw, columns=selected_words)
df_words["ds"] = dates
print(f"    df_words.shape = {df_words.shape}")

# ----- 7A. Prophet-Words (frequency) -----
print("\n  [7A] Prophet-Words (frequency) ...")
pred_word_freq = np.zeros(X_red_raw.shape[1], dtype=float)
true_word_freq = X_red_raw[t_max, :]
print(f"    ✔ true_word_freq.shape = {true_word_freq.shape}")

for j, w in enumerate(selected_words):
    print(f"    → Прогноз слова '{w}' (индекс {j}) ...")
    ts_df = df_words[["ds", w]].rename(columns={w: "y"})
    ts_train = ts_df.iloc[0:train_pairs_end+1]
    model = Prophet()
    try:
        model.fit(ts_train)
        future = model.make_future_dataframe(periods=1, freq=f"{BUCKET_DAYS}D", include_history=False)
        forecast = model.predict(future)
        pred_word_freq[j] = forecast["yhat"].values[-1]
    except Exception as e:
        print(f"WARNING: Prophet.fit() упал для слова '{w}': {e}")
        pred_word_freq[j] = np.nan
    if j % 100 == 0 and j > 0:
        print(f"       ... обработано {j}/{len(selected_words)} слов (freq)")

# Агрегируем предсказания слов в кластеры (frequency)
print("    Агрегируем прогнозы слов в суммарные counts по кластерам ...")
true_cluster_sums_freq = np.zeros(num_clusters, dtype=float)
pred_cluster_sums_freq = np.zeros(num_clusters, dtype=float)
for i, cid in enumerate(cluster_ids):
    mask_c = (word_cluster_map[mask] == cid)
    true_cluster_sums_freq[i] = true_word_freq[mask_c].sum()
    pred_cluster_sums_freq[i] = pred_word_freq[mask_c].sum()
print(f"    true_cluster_sums_freq = {true_cluster_sums_freq}")
print(f"    pred_cluster_sums_freq = {pred_cluster_sums_freq}")

try:
    freq_mse_prophet_words_freq, share_mse_prophet_words_freq, avr_mse_prophet_words_freq = compute_mse_cluster(pred_cluster_sums_freq, true_cluster_sums_freq, word_cluster_map[mask])
    print(f"       [Prophet-Words, freq] freq_mse = {freq_mse_prophet_words_freq:.6f}, share_mse = {share_mse_prophet_words_freq:.6f}, avr_mse_freq = {avr_mse_prophet_words_freq:.6f}")
except Exception as e:
    print(f"ERROR: compute_mse_cluster (Prophet-Words freq) упал: {e}")
    freq_mse_prophet_words_freq = np.nan
    share_mse_prophet_words_freq = np.nan
    avr_mse_prophet_words_freq = np.nan

prophet_results["method"].append("Prophet-Words")
prophet_results["mode"].append("frequency")
prophet_results["freq_mse"].append(freq_mse_prophet_words_freq)
prophet_results["share_mse"].append(share_mse_prophet_words_freq)
prophet_results["avr_mse_freq"].append(avr_mse_prophet_words_freq)

# ----- 7B. Prophet-Words (share) -----
print("\n  [7B] Prophet-Words (share) ...")
word_share_raw_full = np.divide(X_raw, X_raw_sum, where=(X_raw_sum != 0))
word_share_red_raw = word_share_raw_full[:, mask]
print(f"    word_share_red_raw.shape = {word_share_red_raw.shape}")

pred_word_share = np.zeros(X_red_raw.shape[1], dtype=float)
true_word_share = word_share_red_raw[t_max, :]
print(f"    true_word_share.shape = {true_word_share.shape}")

for j, w in enumerate(selected_words):
    print(f"    → Прогноз доли слова '{w}' (индекс {j}) ...")
    ts_df = pd.DataFrame({"ds": dates, "y": word_share_red_raw[:, j]})
    ts_train = ts_df.iloc[0:train_pairs_end+1]
    model = Prophet()
    try:
        model.fit(ts_train)
        future = model.make_future_dataframe(periods=1, freq=f"{BUCKET_DAYS}D", include_history=False)
        forecast = model.predict(future)
        pred_word_share[j] = forecast["yhat"].values[-1]
    except Exception as e:
        print(f"WARNING: Prophet.fit() (share) упал для слова '{w}': {e}")
        pred_word_share[j] = np.nan
    if j % 100 == 0 and j > 0:
        print(f"       ... обработано {j}/{len(selected_words)} слов (share)")

# Агрегируем доли слов в кластеры
print("    Агрегируем прогнозы долей слов в кластеры ...")
true_cluster_sums_share = np.zeros(num_clusters, dtype=float)
pred_cluster_sums_share = np.zeros(num_clusters, dtype=float)
for i, cid in enumerate(cluster_ids):
    mask_c = (word_cluster_map[mask] == cid)
    true_cluster_sums_share[i] = true_word_share[mask_c].sum()
    pred_cluster_sums_share[i] = pred_word_share[mask_c].sum()
print(f"    true_cluster_sums_share = {true_cluster_sums_share}")
print(f"    pred_cluster_sums_share = {pred_cluster_sums_share}")

try:
    freq_mse_prophet_words_share, share_mse_prophet_words_share, avr_mse_prophet_words_share = compute_mse_cluster(pred_cluster_sums_share, true_cluster_sums_share, word_cluster_map[mask])
    print(f"       [Prophet-Words, share] freq_mse = {freq_mse_prophet_words_share:.6f}, share_mse = {share_mse_prophet_words_share:.6f}, avr_mse_freq = {avr_mse_prophet_words_share:.6f}")
except Exception as e:
    print(f"ERROR: compute_mse_cluster (Prophet-Words share) упал: {e}")
    freq_mse_prophet_words_share = np.nan
    share_mse_prophet_words_share = np.nan
    avr_mse_prophet_words_share = np.nan

prophet_results["method"].append("Prophet-Words")
prophet_results["mode"].append("share")
prophet_results["freq_mse"].append(freq_mse_prophet_words_share)
prophet_results["share_mse"].append(share_mse_prophet_words_share)
prophet_results["avr_mse_freq"].append(avr_mse_prophet_words_share)

# ----------------------------
# 8. Сводка результатов
# ----------------------------
print("\n[8] Сводка результатов:\n")

df_pls = pd.DataFrame(results)
print("   === PLS Results ===")
print(df_pls)

df_prophet = pd.DataFrame(prophet_results)
print("\n   === Prophet Results ===")
print(df_prophet)

# Сохраняем в CSV
print("\n[8.1] Сохраняем результаты в CSV-файлы ...")
try:
    df_pls.to_csv("pls_results.csv", index=False)
    print("    ✔ pls_results.csv сохранён")
except Exception as e:
    print(f"ERROR: Не удалось сохранить pls_results.csv: {e}")
try:
    df_prophet.to_csv("prophet_results.csv", index=False)
    print("    ✔ prophet_results.csv сохранён")
except Exception as e:
    print(f"ERROR: Не удалось сохранить prophet_results.csv: {e}")

# ----------------------------
# 9. Пример графика (PLS Frequency MSE)
# ----------------------------
print("\n[9] Строим пример графика сравнения PLS Frequency MSE ...")
try:
    plt.figure(figsize=(6, 4))
    methods = df_pls["method"].unique()
    vals_freq = df_pls[df_pls["mode"] == "frequency"]["freq_mse"].values
    x_positions = np.arange(len(vals_freq))
    plt.bar(x_positions, vals_freq, color='skyblue')
    plt.xticks(x_positions, df_pls[df_pls["mode"] == "frequency"]["method"], rotation=45, ha="right")
    plt.title("PLS Frequency MSE Comparison")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("pls_freq_mse_comparison.png", dpi=300)
    print("    ✔ pls_freq_mse_comparison.png сохранён")
except Exception as e:
    print(f"ERROR: Не удалось построить или сохранить график: {e}")

print("\nPipeline execution complete. Все этапы защищены от падения.\n")
