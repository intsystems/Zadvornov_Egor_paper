import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Параметры для генерации данных
NUM_BUCKETS = 182  # ~1 год по 2 недели
NUM_TOPICS = 3     # количество топиков
WORDS_PER_TOPIC = 10  # слов в каждом топике
NOISE_LEVEL = 0.2  # уровень шума

# Глобальные параметры трендов и сезонности
SEASONALITY_PERIODS = [26, 13, 52]  # полугодовая, квартальная и годовая сезонность
TREND_TYPES = ['linear', 'exp']  # типы трендов
BASE_COUNT = 100  # базовое количество упоминаний

# Создаем топики и слова
topics = ['sports', 'music', 'tech']
words = {
    'sports': ['football', 'basketball', 'tennis', 'soccer', 'nba', 'messi', 'ronaldo', 'olympics', 'championship', 'match'],
    'music': ['concert', 'album', 'song', 'artist', 'band', 'taylor', 'drake', 'spotify', 'festival', 'grammy'],
    'tech': ['iphone', 'android', 'google', 'app', 'ai', 'microsoft', 'update', 'launch', 'startup', 'device']
}

def generate_word_timeseries(num_points, word, topic, base_trend=0.1, seasonality=0.3, noise_scale=0.2):
    """Генерирует временной ряд для одного слова с трендом и сезонностью"""
    t = np.arange(num_points)
    
    # Выбираем тип тренда случайно
    trend_type = np.random.choice(TREND_TYPES)
    if trend_type == 'linear':
        trend = base_trend * t
    else:  # exponential
        trend = np.exp(base_trend * t / num_points) - 1

    # Сезонность (сумма синусоид разной частоты)
    season = 0
    for period in SEASONALITY_PERIODS:
        # Для каждого слова своя амплитуда сезонности
        amplitude = seasonality * np.random.uniform(0.5, 1.5)
        # Для каждого слова свой фазовый сдвиг
        phase = 2 * np.pi * np.random.random()
        season += amplitude * np.sin(2 * np.pi * t / period + phase)
    
    # Шум
    noise = np.random.normal(0, noise_scale, num_points)
    
    # Базовая популярность зависит от слова и топика
    base_popularity = BASE_COUNT * (1 + 0.5 * np.random.random())
    if word in ["messi", "iphone", "taylor"]:  # некоторые слова популярнее других
        base_popularity *= 2
    
    # Финальный ряд (гарантируем неотрицательность)
    series = base_popularity * (1 + trend + season + noise)
    series[series < 0] = 0
    return series

# Генерируем данные
print("Генерируем синтетические данные...")

# Создаем DataFrame для сохранения
data = []
start_date = datetime(2024, 1, 1)

for bucket in range(NUM_BUCKETS):
    date = start_date + timedelta(days=14*bucket)
    
    # Для каждого топика
    for topic in topics:
        # Для каждого слова в топике
        for word in words[topic]:
            # Генерируем временной ряд для слова
            word_ts = generate_word_timeseries(
                NUM_BUCKETS, 
                word,
                topic,
                base_trend=np.random.uniform(0.05, 0.15),
                seasonality=np.random.uniform(0.2, 0.4),
                noise_scale=NOISE_LEVEL)
            # Добавляем запись
            count = int(word_ts[bucket])
            data.append({
                'Time': date,
                'Topic': word,
                'Cluster': topic,
                'Count': count
            })

# Создаем DataFrame
df = pd.DataFrame(data)

# Анализируем и визуализируем данные перед сохранением
print("\nАнализ сгенерированных данных:")
print(f"Общее количество записей: {len(df)}")
print(f"Форма данных: {df.shape}")

print(f"\nРаспределение по кластерам:")
cluster_counts = df.groupby(['Time', 'Cluster'])['Count'].sum().reset_index()
pivot_clusters = cluster_counts.pivot(index='Time', columns='Cluster', values='Count')
print("\nСредние значения по кластерам:")
print(pivot_clusters.mean())
print("\nСтандартные отклонения по кластерам:")
print(pivot_clusters.std())

# Создаем графики
plt.figure(figsize=(15, 10))

# График 1: Суммарные тренды по кластерам
plt.subplot(2, 1, 1)
for cluster in pivot_clusters.columns:
    plt.plot(pivot_clusters.index, pivot_clusters[cluster], label=cluster)
plt.title('Тренды по кластерам')
plt.legend()
plt.grid(True)

# График 2: Распределение значений по кластерам (boxplot)
plt.subplot(2, 1, 2)
pivot_clusters.boxplot()
plt.title('Распределение значений по кластерам')
plt.grid(True)

plt.tight_layout()
plt.savefig('synthetic_data_analysis.png')
plt.close()

# Выводим статистику по словам
print("\nТоп-5 самых популярных слов в каждом кластере:")
for cluster in df['Cluster'].unique():
    print(f"\nКластер {cluster}:")
    top_words = df[df['Cluster'] == cluster].groupby('Topic')['Count'].sum().sort_values(ascending=False).head(5)
    print(top_words)

# Сохраняем в CSV
synthetic_file = 'synthetic_data.csv'
df.to_csv(synthetic_file, index=False)
print(f"\nСинтетические данные сохранены в {synthetic_file}")
print(f"Создан график анализа данных: synthetic_data_analysis.png")
