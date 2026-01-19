# =============================
# Performance Analysis and Visualization of KT Models
# =============================

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. Load Results from JSON Files
# -----------------------------

RESULTS_DIR = "final_results"
summary_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_summary.json')]

# Aggregate all results from JSON into a single DataFrame
all_results = []

for file in summary_files:
    model_name = file.replace('_summary.json', '').upper()
    with open(os.path.join(RESULTS_DIR, file), 'r') as f:
        data = json.load(f)
        for item in data:
            item['model'] = model_name
        all_results.append(pd.DataFrame(data))

df = pd.concat(all_results, ignore_index=True)

# Sort values for cleaner plotting
df.sort_values(by=['dataset', 'model'], inplace=True)

# -----------------------------
# 2. Plotting Model Metrics
# -----------------------------

sns.set(style="whitegrid")

# Barplot: AUC per model
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='model', y='avg_auc', errorbar=None)
plt.title('Average AUC per Model')
plt.ylabel('Mean AUC')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_auc.pdf")

# Barplot: F1-score per model
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='model', y='avg_f1', errorbar=None)
plt.title('Average F1-score per Model')
plt.ylabel('Mean F1-score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig("plot_f1.png")

# Barplot: Accuracy per model
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='model', y='avg_accuracy', errorbar=None)
plt.title('Average Accuracy per Model')
plt.ylabel('Mean Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig("plot_accuracy.png")

# Barplot: Model size
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='model', y='model_size', errorbar=None)
plt.title('Model Size (Number of Parameters)')
plt.ylabel('Size (Parameters)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig("plot_model_size.png")

# -----------------------------
# 3. Heatmap: AUC per Dataset and Model
# -----------------------------

plt.figure(figsize=(10, 5))
pivot_auc = df.pivot(index='dataset', columns='model', values='avg_auc')
sns.heatmap(pivot_auc, annot=True, cmap="YlGnBu")
plt.title("AUC per Dataset and Model")
plt.tight_layout()
plt.savefig("auc_dataset_model.pdf")

# -----------------------------
# 4. Scatterplot: Model Size vs AUC
# -----------------------------

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='model_size', y='avg_auc', hue='model', s=150)
plt.title("Performance vs Complexity")
plt.xlabel("Model Size")
plt.ylabel("Mean AUC")
plt.tight_layout()
#plt.savefig("plot.png")

# -----------------------------
# 5. Radar Chart: Multi-Metric Comparison
# -----------------------------

# Compute average metrics per model
metrics = ['avg_auc', 'avg_f1', 'avg_accuracy', 'avg_inference_time', 'model_size']
df_mean = df.groupby('model')[metrics].mean().reset_index()

# Normalize metrics to [0, 1] scale
scaler = MinMaxScaler()
df_scaled = df_mean.copy()
df_scaled[metrics] = scaler.fit_transform(df_mean[metrics])

# Radar chart setup
categories = metrics
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # loop to close the shape

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each model's values
for _, row in df_scaled.iterrows():
    values = row[metrics].tolist()
    values += values[:1]  # close loop
    ax.plot(angles, values, label=row['model'])
    ax.fill(angles, values, alpha=0.1)

ax.set_title("Radar Chart Comparison of Models")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("radar_chart_models.pdf")
plt.show()

# -----------------------------
# 6. Prepare CSV for MCM (Multiple Comparison Matrix)
# -----------------------------

METRIC = "avg_auc"  # Can also use 'avg_f1' or others

# Reload all summary files to isolate the metric per model/dataset
flat_data = []
for file in os.listdir(RESULTS_DIR):
    if file.endswith("_summary.json"):
        model_name = file.replace("_summary.json", "").upper()
        with open(os.path.join(RESULTS_DIR, file), 'r') as f:
            data = json.load(f)
            for entry in data:
                flat_data.append({
                    "Dataset": entry["dataset"],
                    "Model": model_name,
                    METRIC: entry[METRIC]
                })

df_metric = pd.DataFrame(flat_data)
df_pivot = df_metric.pivot_table(index="Dataset", columns="Model", values=METRIC)
df_pivot.reset_index(inplace=True)

# Export to CSV for MCM analysis
csv_output = "final_results/mcm_auc_input.csv"
df_pivot.to_csv(csv_output, index=False)
print(f"CSV ready for MCM: {csv_output}")

# -----------------------------
# 7. Generate Multiple Comparison Matrix (MCM)
# -----------------------------

from multi_comp_matrix import MCM

# Load CSV results
df_results = pd.read_csv('final_results/mcm_auc_input.csv')
df_results.drop('Dataset', axis=1, inplace=True)

# Define output directory
output_dir = 'final_results/results'

# Run the MCM analysis and generate heatmap
MCM.compare(
    output_dir=output_dir,
    df_results=df_results,
    pdf_savename="heatmap",
    png_savename="heatmap",
    fig_size="14,6",
    used_statistic="AUC"
)
