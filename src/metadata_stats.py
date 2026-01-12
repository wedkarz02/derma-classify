import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "metadata_stats.txt"


df = pd.read_csv(METADATA_PATH)

print("Metadata loaded successfully!")
print(df.head())
print(df.info())

stats = {}

stats["total_images"] = len(df)
stats["unique_lesions"] = df["lesion_id"].nunique()

stats["dx_distribution"] = df["dx"].value_counts()
stats["dx_type_distribution"] = df["dx_type"].value_counts()
stats["sex_distribution"] = df["sex"].value_counts(dropna=False)
stats["localization_distribution"] = df["localization"].value_counts(dropna=False)

stats["age_summary"] = df["age"].describe()
stats["missing_values"] = df.isna().sum()

with open(OUTPUT_PATH, "w") as f:
    f.write("HAM10000 METADATA STATISTICS\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Total images: {stats['total_images']}\n")
    f.write(f"Unique lesions: {stats['unique_lesions']}\n\n")

    f.write("Diagnosis (dx) distribution:\n")
    f.write(stats["dx_distribution"].to_string())
    f.write("\n\n")

    f.write("Diagnosis type (dx_type) distribution:\n")
    f.write(stats["dx_type_distribution"].to_string())
    f.write("\n\n")

    f.write("Sex distribution:\n")
    f.write(stats["sex_distribution"].to_string())
    f.write("\n\n")

    f.write("Localization distribution:\n")
    f.write(stats["localization_distribution"].to_string())
    f.write("\n\n")

    f.write("Age summary:\n")
    f.write(stats["age_summary"].to_string())
    f.write("\n\n")

    f.write("Missing values per column:\n")
    f.write(stats["missing_values"].to_string())

print(f"Stats saved to {OUTPUT_PATH}")
