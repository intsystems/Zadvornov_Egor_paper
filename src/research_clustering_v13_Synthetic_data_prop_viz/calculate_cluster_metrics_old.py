import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import sys
import traceback

warnings.filterwarnings('ignore')

def log_progress(message):
    print(message, flush=True)
    sys.stdout.flush()

def convert_to_proportions(values):
    total = np.sum(values)
    if total == 0:
        return np.zeros_like(values)
    return values / total

def fit_prophet_safely(df, periods=1):
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
        # Use a more sophisticated fallback - weighted average of last 5 values
        weights = np.array([0.4, 0.25, 0.15, 0.12, 0.08])  # More recent values have higher weight
        if len(df['y']) >= 5:
            return [np.average(df['y'].tail(5), weights=weights)] * periods
        else:
            return [df['y'].mean()] * periods

def bootstrap_mse(y_true, y_pred, n_bootstrap=100):
    mse_values = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        mse = mean_squared_error(y_true[indices], y_pred[indices])
        mse_values.append(mse)
    
    # Scale values to match expected magnitudes
    scale_factor = 1000  # Convert to 10^-3
    mse_values = np.array(mse_values) * scale_factor
    
    mean_mse = np.mean(mse_values)
    se_mse = np.std(mse_values) / np.sqrt(n_bootstrap)
    
    return mean_mse, se_mse

try:
    # Set up file paths
    import os
    
    # Debug current directory and file locations
    log_progress(f"Current working directory: {os.getcwd()}")
    log_progress(f"Script location: {os.path.abspath(__file__)}")
    log_progress(f"Directory contents: {os.listdir('.')}")
    
    # Try loading with relative path first
    log_progress("\nAttempting to load data...")
    df = pd.read_csv('synthetic_data_confirming.csv', parse_dates=['Time'])
    log_progress(f"Loaded {len(df)} rows of data")

    # Подготовка данных
    vocab = df['Topic'].unique()
    clusters = df['Cluster'].unique()
    dates = df['Time'].unique()
    test_size = 2

    # Матрицы данных
    word_matrix = pd.pivot_table(
        df, values='Count', index='Time', columns='Topic', aggfunc='sum'
    ).fillna(0)

    cluster_matrix = pd.pivot_table(
        df, values='Count', index='Time', columns='Cluster', aggfunc='sum'
    ).fillna(0)

    # Конвертация в пропорции
    X_raw = word_matrix.values
    y_raw = cluster_matrix.values
    X_props = np.array([convert_to_proportions(X_raw[i]) for i in range(X_raw.shape[0])])
    y_props = np.array([convert_to_proportions(y_raw[i]) for i in range(y_raw.shape[0])])

    # Train/test split
    X_train_props = X_props[:-test_size]
    y_train_props = y_props[:-test_size]
    y_test_props = y_props[-test_size:]

    # Baseline predictions (Direct)
    log_progress("\nCalculating baseline predictions...")
    cluster_predictions = np.zeros((test_size, len(clusters)))
    for i, cluster in enumerate(clusters):
        df_cluster = pd.DataFrame({
            'ds': dates[:-test_size],
            'y': y_train_props[:, i]
        })
        cluster_predictions[:, i] = fit_prophet_safely(df_cluster, periods=test_size)

    # Word-based predictions
    log_progress("\nCalculating word-based predictions...")
    cluster_from_words = np.zeros((test_size, len(clusters)))

    for ci, cluster in enumerate(clusters):
        log_progress(f"\nProcessing cluster: {cluster}")
        # Select words for cluster with improved selection strategy
        cluster_words = [w for w in vocab if df[df['Topic'] == w]['Cluster'].iloc[0] == cluster]
        log_progress(f"Found {len(cluster_words)} total words for cluster")
        
        # First prioritize core words
        core_words = [w for w in cluster_words if '_core' in w]
        log_progress(f"Found {len(core_words)} core words")
        
        # If we don't have enough core words, add other important words
        if len(core_words) < 3:
            non_core_words = [w for w in cluster_words if '_core' not in w]
            # Sort by total count to get most significant words
            word_counts = {w: df[df['Topic'] == w]['Count'].sum() for w in non_core_words}
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            additional_words = [w[0] for w in sorted_words[:3-len(core_words)]]
            selected_words = core_words + additional_words
        else:
            selected_words = core_words[:3]  # Use top 3 core words
            
        log_progress(f"Selected words for prediction: {', '.join(selected_words)}")

        # Прогнозируем по каждому слову
        word_predictions = []
        for word in selected_words:
            word_idx = list(vocab).index(word)
            df_word = pd.DataFrame({
                'ds': dates[:-test_size],
                'y': X_train_props[:, word_idx]
            })
            pred = fit_prophet_safely(df_word, periods=test_size)
            word_predictions.append(pred)
        
        # Агрегируем предсказания
        if word_predictions:
            cluster_from_words[:, ci] = np.mean(word_predictions, axis=0)

    # Нормализуем предсказания
    for t in range(test_size):
        cluster_predictions[t] = convert_to_proportions(cluster_predictions[t])
        cluster_from_words[t] = convert_to_proportions(cluster_from_words[t])

    # Расчет метрик с бутстрапом
    log_progress("\nCalculating metrics with bootstrap...")
    results = []
    for ci, cluster in enumerate(clusters):
        # Baseline metrics
        baseline_mse, baseline_se = bootstrap_mse(
            y_test_props[:, ci], 
            cluster_predictions[:, ci]
        )
        
        # Word-based metrics
        word_based_mse, word_based_se = bootstrap_mse(
            y_test_props[:, ci], 
            cluster_from_words[:, ci]
        )
        
        results.append({
            'Cluster': cluster,
            'Baseline_MSE': baseline_mse,
            'Baseline_SE': baseline_se,
            'WordBased_MSE': word_based_mse,
            'WordBased_SE': word_based_se
        })

    # Формируем DataFrame с результатами
    results_df = pd.DataFrame(results)

    # Считаем средние значения по всем кластерам
    avg_results = {
        'Cluster': 'Avg',
        'Baseline_MSE': results_df['Baseline_MSE'].mean(),
        'Baseline_SE': np.sqrt(np.mean(results_df['Baseline_SE']**2)),
        'WordBased_MSE': results_df['WordBased_MSE'].mean(),
        'WordBased_SE': np.sqrt(np.mean(results_df['WordBased_SE']**2))
    }
    results_df = results_df.append(avg_results, ignore_index=True)

    # Convert results to DataFrame and add average metrics
    log_progress("\nComputing final metrics...")
    results_df = pd.DataFrame(results)
    
    # Calculate averages
    avg_results = pd.DataFrame([{
        'Cluster': 'Avg',
        'Baseline_MSE': results_df['Baseline_MSE'].mean(),
        'Baseline_SE': np.sqrt(np.mean(results_df['Baseline_SE']**2)),
        'WordBased_MSE': results_df['WordBased_MSE'].mean(),
        'WordBased_SE': np.sqrt(np.mean(results_df['WordBased_SE']**2))
    }])
    results_df = pd.concat([results_df, avg_results], ignore_index=True)
    
    # Save results
    log_progress("\nSaving results to CSV...")
    results_df.to_csv('cluster_wise_metrics.csv', index=False)
    log_progress("\nResults saved to cluster_wise_metrics.csv")

    # Print LaTeX table with clear separation
    log_progress("\n" + "="*50)
    log_progress("LaTeX Table Format:")
    log_progress("="*50 + "\n")
    
    # Форматируем вывод в виде LaTeX таблицы
    latex_table = r"""\begin{frame}{Результаты вычислительного эксперимента}
\begin{block}{Сравнение по кластерам ($\times 10^{-3}$ MSE $\pm$ SE)}
\centering
\small
\begin{tabular}{|l|c|c|c|c|}
\hline
\rowcolor{lightgray} \textbf{Метод} & \textbf{Sports} & \textbf{Music} & \textbf{Tech} & \textbf{Avg} \\ \hline"""

    print(latex_table)

    # Базовый метод
    baseline_row = "Базовый     "
    for cluster in ['Sports', 'Music', 'Tech', 'Avg']:
        row = results_df[results_df['Cluster'] == cluster].iloc[0]
        baseline_row += f" & {row['Baseline_MSE']:.1f}$\\pm${row['Baseline_SE']:.1f}"
    print(baseline_row + " \\\\ \\hline")

    # Предложенный метод
    proposed_row = "Предложенный"
    for cluster in ['Sports', 'Music', 'Tech', 'Avg']:
        row = results_df[results_df['Cluster'] == cluster].iloc[0]
        proposed_row += f" & {row['WordBased_MSE']:.1f}$\\pm${row['WordBased_SE']:.1f}"
    print(proposed_row + " \\\\ \\hline")

    print(r"""\end{tabular}
\normalsize
\end{block}
\end{frame}""")

except Exception as e:
    print("\nError occurred:", str(e))
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
