import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = r".\outputs\ndvi\field_ndvi.csv"
OUTPUT_DIR = r".\outputs\ndvi\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 1. OVERALL MONTHLY NDVI
# ============================================================

monthly = (
    df.groupby("date")["median_ndvi"]
    .agg(["mean", "median"])
    .reset_index()
)

plt.figure(figsize=(9, 5))

plt.plot(
    monthly["date"],
    monthly["mean"],
    marker="o",
    label="Mean field NDVI"
)

plt.plot(
    monthly["date"],
    monthly["median"],
    marker="o",
    label="Median field NDVI"
)

plt.xlabel("Month")
plt.ylabel("NDVI")
plt.title("Overall Monthly NDVI Trend")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "overall_ndvi_trend.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


# ============================================================
# 2. SAMPLE FIELD TRENDS
# ============================================================

# Only use fields having at least 10 pixels
field_sizes = (
    df.groupby(["sample", "field_id"])["pixel_count"]
    .first()
    .reset_index()
)

reliable_fields = field_sizes[
    field_sizes["pixel_count"] >= 10
]

df_reliable = df.merge(
    reliable_fields[["sample", "field_id"]],
    on=["sample", "field_id"]
)


# Pick several fields from AT_164
sample = "AT_164_S2_10m_256"

fields = (
    df_reliable[
        df_reliable["sample"] == sample
    ]
    .groupby("field_id")["pixel_count"]
    .first()
    .sort_values(ascending=False)
    .head(6)
    .index
)


plt.figure(figsize=(10, 6))

for field_id in fields:

    x = df_reliable[
        (df_reliable["sample"] == sample)
        & (df_reliable["field_id"] == field_id)
    ]

    plt.plot(
        x["date"],
        x["median_ndvi"],
        marker="o",
        label=f"Field {field_id}"
    )

plt.xlabel("Month")
plt.ylabel("Median NDVI")
plt.title("Field-Level NDVI Trends - AT_164")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "AT_164_field_ndvi_trends.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


# ============================================================
# 3. NDVI CHANGE DISTRIBUTION
# ============================================================

pivot = (
    df_reliable
    .pivot_table(
        index=["sample", "field_id"],
        columns="date",
        values="median_ndvi"
    )
)

first_date = pivot.columns.min()
last_date = pivot.columns.max()

pivot["ndvi_change"] = (
    pivot[last_date] - pivot[first_date]
)

plt.figure(figsize=(9, 5))

plt.hist(
    pivot["ndvi_change"].dropna(),
    bins=40
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("NDVI change (last month - first month)")
plt.ylabel("Number of fields")
plt.title("Distribution of Field NDVI Change")
plt.grid(True, alpha=0.3)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "ndvi_change_distribution.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# 4. HEALTH STATUS CLASSIFICATION
# ============================================================

# Get first and last month NDVI for every reliable field
health = pivot[[first_date, last_date, "ndvi_change"]].copy()

health = health.rename(
    columns={
        first_date: "initial_ndvi",
        last_date: "final_ndvi"
    }
)


def classify_health(row):

    final_ndvi = row["final_ndvi"]
    change = row["ndvi_change"]

    # Poor: low final vegetation OR significant decline
    if final_ndvi < 0.50 or change < -0.05:
        return "Poor"

    # Healthy: good final vegetation AND meaningful improvement
    elif final_ndvi >= 0.70 and change >= 0.05:
        return "Healthy"

    # Everything between these conditions
    else:
        return "Moderate"


health["health_status"] = health.apply(
    classify_health,
    axis=1
)

# Reset index so sample and field_id become columns
health = health.reset_index()


# Save classification table
health_csv = os.path.join(
    r".\outputs\ndvi",
    "field_health_classification.csv"
)

health.to_csv(
    health_csv,
    index=False
)

print("Saved:", health_csv)


# ============================================================
# 5. HEALTH STATUS DISTRIBUTION
# ============================================================

counts = (
    health["health_status"]
    .value_counts()
    .reindex(["Healthy", "Moderate", "Poor"], fill_value=0)
)

plt.figure(figsize=(8, 5))

counts.plot(
    kind="bar"
)

plt.xlabel("Health Status")
plt.ylabel("Number of Fields")
plt.title("Field Crop Health Classification")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "health_status_distribution.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


# ============================================================
# 6. HEALTH STATUS PERCENTAGE
# ============================================================

percentages = (
    counts / counts.sum() * 100
)

print()
print("=" * 60)
print("FIELD HEALTH CLASSIFICATION")
print("=" * 60)

for status in ["Healthy", "Moderate", "Poor"]:
    print(
        f"{status:10s}: "
        f"{counts[status]:4d} fields "
        f"({percentages[status]:.2f}%)"
    )

print("Total fields:", counts.sum())


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("NDVI PLOT GENERATION COMPLETE")
print("=" * 60)

print("Reliable fields:", len(pivot))
print("First month:", first_date)
print("Last month:", last_date)

print()
print("NDVI change statistics:")
print(pivot["ndvi_change"].describe())