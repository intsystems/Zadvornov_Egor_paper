import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
# headless backend for PNGs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

"""
Full visual analytics for synthetic dataset confirming H1 & H2.
Generates:
 1. Interactive Plotly dashboard (core/noise/cluster toggles) -> plots/dashboard.html
 2. Violin plot PNG comparing distributions of core vs noise words.
 3. Scatter plot PNG with R² annotations (word vs cluster).
 4. Interactive Prophet forecast with prediction intervals -> plots/prophet_forecasts.html
 5. Horizontal bar PNG of PLS importances (color‑coded core vs noise).
"""

# ---------------- CONFIG -----------------
CSV_PATH  = "synthetic_data_confirming.csv"
OUT_DIR   = "plots"
CLUSTER   = "Sports"            # визуализируем один кластер (можно изменить)
TEST_H    = 2                   # горизонт прогнозирования, как в гипотезах
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD DATA --------------
df = pd.read_csv(CSV_PATH, parse_dates=["Time"])
cl_df = df[df["Cluster"] == CLUSTER].copy()
core_topics  = [t for t in cl_df["Topic"].unique() if "_core"  in t]
noise_topics = [t for t in cl_df["Topic"].unique() if "_noise" in t]

pivot = cl_df.pivot_table(index="Time", columns="Topic", values="Count", aggfunc="sum").fillna(0)
cluster_series = pivot.sum(axis=1)
core_matrix  = pivot[core_topics]
noise_matrix = pivot[noise_topics]

# ---------------- 1. Plotly dashboard ----------------------------
fig_dash = go.Figure()
fig_dash.add_trace(go.Scatter(x=cluster_series.index, y=cluster_series.values,
                              mode="lines", name="Cluster total", line=dict(width=2)))
for w in core_topics:
    fig_dash.add_trace(go.Scatter(x=pivot.index, y=pivot[w], mode="lines",
                                  name=w, visible="legendonly"))
for w in noise_topics[:len(core_topics)]:           # показываем столько же noise‑слов
    fig_dash.add_trace(go.Scatter(x=pivot.index, y=pivot[w], mode="lines",
                                  name=w, line=dict(dash="dot"), visible="legendonly"))
fig_dash.update_layout(title=f"Interactive dashboard: {CLUSTER}", xaxis_title="Date", yaxis_title="Count")
fig_dash.write_html(os.path.join(OUT_DIR, "dashboard.html"))

# ---------------- 2. Violin plot -------------------------------
plt.figure(figsize=(6, 6))
core_counts  = cl_df[cl_df["Topic"].isin(core_topics)]["Count"]
noise_counts = cl_df[cl_df["Topic"].isin(noise_topics)]["Count"]

sns.violinplot(data=[core_counts, noise_counts], palette=["#55a868", "#c44e52"], cut=0)
plt.xticks([0, 1], ["Core words", "Noise words"])
plt.ylabel("Count"); plt.title(f"Distribution of counts ({CLUSTER})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{CLUSTER.lower()}_violin.png"), dpi=150)
plt.close()

# ---------------- 3. Scatter + R² -------------------------------
plt.figure(figsize=(6, 6))
for w in core_topics[:3]:
    plt.scatter(pivot[w], cluster_series, s=10, label=w)
from sklearn.linear_model import LinearRegression
x_core = core_matrix.mean(axis=1).values.reshape(-1, 1)
model  = LinearRegression().fit(x_core, cluster_series.values)
r2     = model.score(x_core, cluster_series.values)
plt.xlabel("Word count (core avg)"); plt.ylabel("Cluster total")
plt.title(f"Scatter core vs cluster (R²={r2:.2f})")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{CLUSTER.lower()}_scatter.png"), dpi=150)
plt.close()

# ---------------- 4. Prophet forecast intervals -----------------
train_idx = slice(None, -TEST_H)

def prophet_forecast(series: pd.DataFrame):
    """
    series — pd.DataFrame с индексом Time и ОДНОЙ числовой колонкой.
    Возвращает (модель, forecast_df).
    """
    df_fit = series.reset_index()
    df_fit.columns = ["ds", "y"]          # переименовать вторую колонку в y

    m = Prophet(yearly_seasonality=True)
    m.fit(df_fit)
    future = m.make_future_dataframe(periods=TEST_H, freq="2W")
    fcst   = m.predict(future)
    return m, fcst


m_cl, fc_cl = prophet_forecast(cluster_series.to_frame())
# sum of core forecasts
core_preds = []
for w in core_topics:
    m_w, fc_w = prophet_forecast(pivot[w].to_frame())
    core_preds.append(fc_w[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(TEST_H))
core_sum = pd.concat(core_preds).groupby("ds").sum().reset_index()

fig_p = go.Figure()
# cluster forecast
fig_p.add_trace(go.Scatter(name="Cluster yhat", x=fc_cl["ds"], y=fc_cl["yhat"],
                           mode="lines", line=dict(color="royalblue")))
fig_p.add_trace(go.Scatter(name="Cluster upper", x=fc_cl["ds"], y=fc_cl["yhat_upper"],
                           mode="lines", marker=dict(color="#A6C8FF"), showlegend=False))
fig_p.add_trace(go.Scatter(name="Cluster lower", x=fc_cl["ds"], y=fc_cl["yhat_lower"],
                           mode="lines", marker=dict(color="#A6C8FF"), fill="tonexty", showlegend=False))
# core‑sum forecast
fig_p.add_trace(go.Scatter(name="Sum core yhat", x=core_sum["ds"], y=core_sum["yhat"],
                           mode="lines", line=dict(color="seagreen")))
fig_p.update_layout(title=f"Prophet forecasts ({CLUSTER})", xaxis_title="Date", yaxis_title="Count")
fig_p.write_html(os.path.join(OUT_DIR, "prophet_forecasts.html"))

# ---------------- 5. PLS importances ----------------------------
scaler = StandardScaler(with_mean=False).fit(pivot.values)
Xsc    = scaler.transform(pivot.values)
pls    = PLSRegression(n_components=3).fit(Xsc[train_idx], cluster_series.values[train_idx, None])
coefs  = np.abs(pls.coef_).reshape(-1)
imp    = pd.Series(coefs, index=pivot.columns).sort_values(ascending=False)

top = imp.head(20)
colors = ["#55a868" if t in core_topics else "#c44e52" for t in top.index]
plt.figure(figsize=(8, 6))
plt.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
plt.xlabel("|PLS coefficient|")
plt.title(f"PLS importance (top 20) — {CLUSTER}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{CLUSTER.lower()}_pls_importance.png"), dpi=150)
plt.close()

print("All visualizations saved to", OUT_DIR)
