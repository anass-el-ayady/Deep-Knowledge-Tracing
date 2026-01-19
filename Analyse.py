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

sns.set(style="whitegrid")

# -----------------------------
# 2. Heatmap: AUC per Dataset and Model
# -----------------------------

plt.figure(figsize=(10, 5))
pivot_auc = df.pivot(index='dataset', columns='model', values='avg_auc')
sns.heatmap(pivot_auc, annot=True, cmap="YlGnBu")
plt.title("AUC per Dataset and Model")
plt.tight_layout()
plt.savefig("auc_dataset_model.pdf")


# -----------------------------
# 3. Prepare CSV for MCM (Multiple Comparison Matrix)
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
# 4. Generate Multiple Comparison Matrix (MCM)
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


# -----------------------------
# 5. Bubble plot
# -----------------------------

# Filtrer les modèles à comparer
df_plot = df_mean[df_mean['model'].isin(['FCN', 'LSTM', 'RNN', 'DKTPLUS', 'SAKT', 'KQN', 'DKVMN'])].copy()

plt.figure(figsize=(12, 8))

# Couleurs différentes pour chaque modèle
colors = plt.cm.tab10(np.linspace(0, 1, len(df_plot)))

# Taille des bulles avec log ou sqrt compression
#bubble_sizes = np.sqrt(df_plot['model_size'])*2 # Ajuster ce facteur si besoin

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(500, 2000))  # plage visible
bubble_sizes = scaler.fit_transform(df_plot[['model_size']]).flatten()

# Tracer chaque modèle individuellement
for i, (index, row) in enumerate(df_plot.iterrows()):
    plt.scatter(
        row['avg_inference_time'],
        row['avg_auc'],
        s=bubble_sizes[index],
        color=colors[i],
        edgecolors='black',
        alpha=0.6
    )
    # Afficher nom + taille du modèle
    label = f"{row['model']} ({int(row['model_size']):,})"
    plt.text(row['avg_inference_time'] + 0.05, row['avg_auc'], label, fontsize=9)

# Axes et titres avec tailles de police augmentées
plt.xlabel("Average Inference Time (seconds)", fontsize=16)
plt.ylabel("Mean AUC", fontsize=16)
plt.title("Models Comparison- Inference Time vs AUC", fontsize=20)



# Légende fictive pour les tailles de modèle
for size in [1e3, 1e4, 1e5, 1e6]:
    plt.scatter([], [], s=np.sqrt(size) * 2, label=f"{int(size):,} params", color='gray', alpha=0.4, edgecolors='k')

plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Model size",fontsize=10,title_fontsize=11 )
plt.grid(True)
plt.tight_layout()
plt.savefig("bubble_plot.pdf")
plt.show()
