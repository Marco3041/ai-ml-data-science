import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.makedirs('plots', exist_ok=True)
df = pd.read_csv('D:/CODE/AIML/data.csv')

print(fr"\nQuick check: Here's the average for cholesterol, blood pressure, and heart rate:")
print(df[['chol', 'trestbps', 'thalach']].mean())

print(fr"\nHighest values recorded — these are the peak numbers:")
print(df[['chol', 'trestbps', 'thalach']].max())

print(fr"\nLowest values — the calm zone:")
print(df[['chol', 'trestbps', 'thalach']].min())

print(fr"\nFull summary — gives you the whole picture:")
print(df[['chol', 'trestbps', 'thalach']].describe())

print(fr"\nInfo dump — just so we know what we're working with:")
print(df.info())

print(fr"\nPlotting histograms to see how the data spreads out...")
plt.figure(figsize=(12, 4))
for i, col in enumerate(['chol', 'trestbps', 'thalach']):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col}')
plt.tight_layout()
plt.savefig('plots/histograms.png')
plt.show()
print(fr"Saved! Check out 'plots/histograms.png' for the visuals.")