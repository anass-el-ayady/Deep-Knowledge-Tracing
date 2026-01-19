# =============================
# Analyze Dataset Density and Its Correlation with Model AUC
# =============================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------
# Function to read and process interaction data from train/test files
# ----------------------------------------

def read_custom_dataset(train_path, test_path, target_user_index=None):
    """
    Read train/test files, extract basic stats, and compute interaction density.

    Returns:
        - num_users: Total number of users
        - num_skill_tags: Total number of unique skills
        - num_unique_interactions: Total number of user-skill interactions
        - density: Ratio of observed to possible interactions
    """
    users = 0
    skill_tags_set = set()
    unique_interactions = set()
    user_interactions = {}

    files_to_read = []
    if os.path.exists(train_path):
        files_to_read.append(train_path)
    if os.path.exists(test_path):
        files_to_read.append(test_path)
    
    if not files_to_read:
        return 0, 0, 0, 0  # No file found

    user_id = 0
    for file_path in files_to_read:
        with open(file_path, "r") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        i = 0
        while i < len(lines):
            try:
                seq_length = int(lines[i])  # Line 1: sequence length
                skill_ids = list(map(int, filter(None, lines[i + 1].split(","))))  # Line 2: skill sequence
            except ValueError:
                break  # Error in parsing

            users += 1
            skill_tags_set.update(skill_ids)
            user_interactions[user_id] = set(skill_ids)

            for skill in skill_ids:
                unique_interactions.add((user_id, skill))

            user_id += 1
            i += 3  # Skip to next user sequence

    num_users = users
    num_skill_tags = len(skill_tags_set)
    num_unique_interactions = len(unique_interactions)

    total_possible_interactions = num_users * num_skill_tags
    density = num_unique_interactions / total_possible_interactions if total_possible_interactions > 0 else 0

    return num_users, num_skill_tags, num_unique_interactions, round(density, 4)

# ----------------------------------------
# Step 1: Compute density for each dataset
# ----------------------------------------

dataset_path = "dataset/"
datasets = os.listdir(dataset_path)
target_user_index = 2  # Optional debug

results = []

for dataset in tqdm(datasets):
    train_file = os.path.join(dataset_path, dataset, "builder_train.csv")
    test_file = os.path.join(dataset_path, dataset, "builder_test.csv")

    num_users, num_skills, num_interactions, density = read_custom_dataset(train_file, test_file, target_user_index)

    results.append({
        "Dataset": dataset,
        "# Users": num_users,
        "# Skill tags": num_skills,
        "# Unique Interactions": num_interactions,
        "Density": density
    })

df_results = pd.DataFrame(results)
df_results.to_csv('density.csv', index=False)

# ----------------------------------------
# Step 2: Plot Density per Dataset
# ----------------------------------------

plt.figure(figsize=(12, 6))
sns.barplot(data=df_results.sort_values(by="Density", ascending=False),
            x="Dataset", y="Density")
plt.xticks(rotation=45)
plt.title("Density per Dataset")
plt.tight_layout()
plt.savefig('final_results/density.pdf')

# ----------------------------------------
# Step 3: Merge with Model AUC results
# ----------------------------------------

# Load density and AUC tables
df_density = pd.read_csv("final_results/density.csv")
df_auc = pd.read_csv("final_results/mcm_auc_input.csv")

# Convert wide format to long: (Dataset, Model, AUC)
df_long = df_auc.melt(id_vars="Dataset", var_name="Model", value_name="AUC")

# Merge with density info
df_merged = pd.merge(df_long, df_density[["Dataset", "Density"]], on="Dataset")

# ----------------------------------------
# Step 4: Scatterplot – AUC vs Density by Model
# ----------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_merged, x="Density", y="AUC", hue="Model", s=100)
plt.title("Performance (AUC) vs Dataset Density by Model")
plt.xlabel("Density")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
#plt.savefig("final_results/density_vs_auc_per_model.png")

# ----------------------------------------
# Step 5: Pearson Correlation between Density and AUC per Model
# ----------------------------------------

correlations = df_merged.groupby("Model").apply(
    lambda group: group["Density"].corr(group["AUC"])
).sort_values()

# Print correlation values
print("Pearson Correlation between Density and AUC per Model:\n")
print(correlations)

# ----------------------------------------
# Step 6: Plot Correlation Barplot
# ----------------------------------------

plt.figure(figsize=(8, 5))
sns.barplot(x=correlations.values, y=correlations.index, palette="coolwarm")
plt.title("Correlation between Dataset Density and AUC by Model")
plt.xlabel("Pearson Correlation")
plt.ylabel("Model")
plt.axvline(0, color='gray', linestyle='--')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("final_results/density_auc_correlation_barplot.pdf")
