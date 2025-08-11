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

print(fr"\nBoxplots coming up — let's spot any outliers...")
plt.figure(figsize=(8, 4))
for i, col in enumerate(['chol', 'trestbps']):
    plt.subplot(1, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'{col}')
plt.tight_layout()
plt.savefig('plots/boxplots.png')
plt.close()
print(fr"Boxplots saved to 'plots/boxplots.png' — go take a look!")

print(fr"\nRunning a T-test to compare cholesterol between males and females...")
male_chol = df[df['sex'] == 1]['chol']
female_chol = df[df['sex'] == 0]['chol']
t_stat, p_val = stats.ttest_ind(male_chol, female_chol)
print(fr"Result: t = {t_stat:.2f}, p = {p_val:.4f} — interesting difference!")

print(fr"\nChecking if the data looks normal (Shapiro-Wilk Test):")
print(fr"Cholesterol:", stats.shapiro(df['chol']))
print(fr"Max heart rate:", stats.shapiro(df['thalach']))

print(fr"\nANOVA time — does cholesterol vary by chest pain type?")
anova_result = stats.f_oneway(*(df[df['cp'] == i]['chol'] for i in df['cp'].unique()))
print(fr"ANOVA result: F = {anova_result.statistic:.2f}, p = {anova_result.pvalue:.4f}")

print(fr"\nConfidence intervals — where do the true values likely fall?")
for col in ['chol', 'trestbps']:
    mean = df[col].mean()
    sem = stats.sem(df[col])
    ci = stats.t.interval(0.95, len(df[col])-1, loc=mean, scale=sem)
    print(fr"{col}: Between {ci[0]:.2f} and {ci[1]:.2f}")

print(fr"\nComparing heart rate between people with and without heart disease...")
thalach_disease = df[df['target'] == 1]['thalach']
thalach_nodisease = df[df['target'] == 0]['thalach']
t_stat, p_val = stats.ttest_ind(thalach_disease, thalach_nodisease)
print(fr"T-test result: t = {t_stat:.2f}, p = {p_val:.4f} — looks like a strong difference!")
