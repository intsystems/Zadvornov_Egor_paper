import os
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

"""
Forecast social‑media topics two‑weeks ahead (v7)
-------------------------------------------------
▶ **Lagged regressors** вместо future‑значений.
   ▸ лаг L = TEST_SIZE (=2 бакета = 4 недели).
   ▸ На дате *t* модель видит частоту слова на *t‑L* — при прогнозе «будущее»
     эти значения уже известны, поэтому правило future‑regressor не нарушается.
▶ Plotly‑диаграммы как раньше (`plots/*.html`).
"""

# ================= CONFIG ==========================================
CSV_PATH       = "synthetic_data.csv"
TEST_SIZE      = 1          # горизонты прогноза (2×2 недели) → лаг 2
TOP_K          = 50         # максимум слов через PLS
MIN_KEEP       = 10         # минимум слов, даже при слабой важности
ROLLING_WINDOW = None       # None ▸ статично; int ▸ размер окна
FREQ           = "2W"
RESULTS_CSV    = "improved_results.csv"
PLOT_DIR       = "plots"
BAR_PATH       = os.path.join(PLOT_DIR, "mse_comparison.html")
LAG            = TEST_SIZE  # критично: сдвиг регрессоров
# ================================================================

os.makedirs(PLOT_DIR, exist_ok=True)

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    print("Plotly not found — aborting visualization part.")
    PLOTLY_OK = False

# ---------------- UTILS -------------------------------------------

def evaluate(y_true, y_pred, name="model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name}: MSE={mse:.2f}  MAE={mae:.2f}")
    return {"model": name, "mse": mse, "mae": mae}


def select_top_words_pls(X, y, vocab, top_k=50, window=None):
    top_k = min(top_k, len(vocab))
    if window is None:
        pls = PLSRegression(n_components=min(3, X.shape[1] - 1), scale=False)
        pls.fit(X, y)
        imp = np.abs(pls.coef_).sum(axis=1)
        return vocab[imp.argsort()[::-1][:top_k]]

    hits = np.zeros(len(vocab))
    for start in range(0, len(X) - window + 1):
        pls = PLSRegression(n_components=2, scale=False)
        pls.fit(X[start : start + window], y[start : start + window])
        imp = np.abs(pls.coef_).sum(axis=1)
        hits[imp.argsort()[::-1][:top_k]] += 1
    thresh = 0.3 * hits.max()
    keep = np.where(hits >= thresh)[0]
    if len(keep) < top_k:
        remainder = np.setdiff1d(np.arange(len(vocab)), keep)
        add_idx = remainder[np.argsort(-hits[remainder])][: top_k - len(keep)]
        keep = np.concatenate([keep, add_idx])
    return vocab[np.sort(keep)]


# ---------------- MAIN --------------------------------------------

def main():
    print("Loading data …")
    df = pd.read_csv(CSV_PATH, parse_dates=["Time"])

    vocab = df["Topic"].unique()
    clusters = df["Cluster"].unique()
    print(f"→ {len(vocab)} unique words, {len(clusters)} clusters")

    # Build matrices
    word_mat = (
        pd.pivot_table(df, index="Time", columns="Topic", values="Count", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )
    cluster_mat = (
        pd.pivot_table(df, index="Time", columns="Cluster", values="Count", aggfunc="sum")
        .fillna(0)
        .reindex(word_mat.index)
    )

    dates = word_mat.index
    X, y = word_mat.values, cluster_mat.values
    X_train, X_test = X[:-TEST_SIZE], X[-TEST_SIZE:]
    y_train, y_test = y[:-TEST_SIZE], y[-TEST_SIZE:]
    dates_train = dates[:-TEST_SIZE]

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)

    # 1️⃣ Baseline Prophet ------------------------------------------
    print("\nFitting baseline Prophet …")
    base_pred = np.zeros_like(y_test, dtype=float)
    for i, cl in enumerate(clusters):
        m = Prophet(yearly_seasonality=True)
        m.fit(pd.DataFrame({"ds": dates_train, "y": y_train[:, i]}))
        future = m.make_future_dataframe(periods=TEST_SIZE, freq=FREQ)
        fcst = m.predict(future)
        base_pred[:, i] = fcst["yhat"].tail(TEST_SIZE).values
        if PLOTLY_OK:
            plot_plotly(m, fcst).write_html(os.path.join(PLOT_DIR, f"baseline_{cl}.html"))
            plot_components_plotly(m, fcst).write_html(os.path.join(PLOT_DIR, f"baseline_comp_{cl}.html"))
    base_metrics = evaluate(y_test, base_pred, "Prophet_base")

    # 2️⃣ Select regressors -----------------------------------------
    print("Selecting top-K words …")
    top_words = select_top_words_pls(X_train_scaled, y_train, vocab=np.array(vocab), top_k=TOP_K, window=ROLLING_WINDOW)
    if len(top_words) < MIN_KEEP:
        print(f"PLS chose only {len(top_words)} words. Fallback to most frequent {MIN_KEEP} words.")
        freq_words = word_mat.mean(axis=0).sort_values(ascending=False).head(MIN_KEEP).index.values
        top_words = np.unique(np.concatenate([top_words, freq_words]))
    print(f"Using {len(top_words)} regressors (lag {LAG}): {', '.join(top_words[:10])} …")

    # 3️⃣ Build LAGGED regressor DF (shift = LAG) -------------------
    reg_hist = np.log1p(word_mat[top_words]).shift(LAG)  # x(t-LAG)
    reg_hist = reg_hist.fillna(0)

    # Known future values for lagged regressors = last LAG observed rows
    future_dates = pd.date_range(start=dates[-1] + pd.tseries.frequencies.to_offset(FREQ),
                                 periods=TEST_SIZE, freq=FREQ)
    reg_future_vals = np.log1p(word_mat[top_words].iloc[-LAG:]).copy()
    reg_future_vals.index = future_dates

    reg_all = pd.concat([reg_hist, reg_future_vals])
    # ensure column 'ds' exists for merge
    reg_all = reg_all.copy()
    reg_all['ds'] = reg_all.index
    reg_all = reg_all.reset_index(drop=True)

    # 4️⃣ Prophet + lagged regressors ------------------------------ ------------------------------
    print("\nFitting Prophet with LAGGED regressors …")
    reg_pred = np.zeros_like(y_test, dtype=float)
    for i, cl in enumerate(clusters):
        y_df = pd.DataFrame({"ds": dates.append(future_dates),
                             "y" : np.concatenate([cluster_mat.iloc[:, i].values, [np.nan]*TEST_SIZE])})
        merged = y_df.merge(reg_all.reset_index().rename(columns={"Time": "ds"}), on="ds", how="left").fillna(0)

        m = Prophet(yearly_seasonality=True)
        for col in top_words:
            m.add_regressor(col)
        m.fit(merged.iloc[:-TEST_SIZE])  # train only on historic part
        fcst = m.predict(merged.drop(columns="y"))
        reg_pred[:, i] = fcst["yhat"].tail(TEST_SIZE).values
        if PLOTLY_OK:
            plot_plotly(m, fcst).write_html(os.path.join(PLOT_DIR, f"reg_{cl}.html"))
            plot_components_plotly(m, fcst).write_html(os.path.join(PLOT_DIR, f"reg_comp_{cl}.html"))

    reg_metrics = evaluate(y_test, reg_pred, "Prophet_lagreg")

    # delta
    delta = 100 * (base_metrics["mse"] - reg_metrics["mse"]) / base_metrics["mse"]
    print(f"\nΔMSE vs baseline: {delta:+.1f}% (using lagged regressors)")

    # save metrics
    pd.DataFrame([base_metrics, reg_metrics]).to_csv(RESULTS_CSV, index=False)

    if PLOTLY_OK:
        bar = go.Figure()
        bar.add_bar(x=["Baseline", "Prophet_lagreg"], y=[base_metrics["mse"], reg_metrics["mse"]])
        bar.update_layout(yaxis_title="MSE (↓)")
        bar.write_html(BAR_PATH)
        print("Artifacts saved in", PLOT_DIR, "and", RESULTS_CSV)
    else:
        print("Results saved →", RESULTS_CSV)


if __name__ == "__main__":
    main()
