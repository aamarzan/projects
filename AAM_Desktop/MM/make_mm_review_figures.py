# make_mm_review_figures.py
# Generates: PRISMA (requires your counts), symptom prevalence plot, skeletal plot, lab means plot, and CSV tables.

import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "outputs_mm_review"
import os
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# 1) DATA (edit if needed)
# ---------------------------

studies = [
    {
        "study": "Chowdhury 2018 (JEMC)",
        "n": 32,
        "male_pct": 90.63,
        "age_mean": 51.94,
        "bone_pain_pct": None,
        "anemia_pct": 62.5,  # Hb <=10 g/dL (13+7)/32
        "fatigue_pct": None,
        "renal_impairment_pct": 21.87,  # creatinine >2 mg/dL
        "infections_pct": None,
        "hb_mean": None,
        "esr_mean": None,
        "creatinine_mean": None,
        "calcium_mean": None,
    },
    {
        "study": "Masud 2023 (SEEJPH)",
        "n": 60,
        "male_pct": 61.7,
        "age_mean": 59.8,
        "bone_pain_pct": 76.7,
        "anemia_pct": 70.0,
        "fatigue_pct": 65.0,
        "renal_impairment_pct": 25.0,
        "infections_pct": 30.0,
        "hb_mean": 8.9,
        "esr_mean": 96.1,
        "creatinine_mean": 2.1,
        "calcium_mean": 10.8,
    },
    {
        "study": "Clinical profile (n=58)",
        "n": 58,
        "male_pct": 62.1,
        "age_mean": None,  # age group reported, not mean
        "bone_pain_pct": 82.8,
        "anemia_pct": 69.0,
        "fatigue_pct": 62.1,
        "renal_impairment_pct": 25.9,
        "infections_pct": 20.7,
        "hb_mean": 8.9,
        "esr_mean": 75.0,
        "creatinine_mean": 2.1,
        "calcium_mean": 11.2,
    },
]

skeletal = {
    "study": "Haematol J BD 2022 skeletal cohort",
    "n": 159,
    "skeletal_involvement_pct": 76.1,
    "lytic_any_pct": 45.9,
    "lytic_single_pct": 15.1,
    "lytic_multiple_pct": 30.8,
    "fracture_any_pct": 44.7,
    "fracture_vertebral_pct": 37.1,
    "fracture_rib_pct": 9.4,
    "fracture_humerus_pct": 3.1,
    "fracture_femur_pct": 3.8,
    "fracture_neck_femur_pct": 0.6,
    "nerve_root_compression_pct": 12.6,
}

# PRISMA counts (YOU MUST EDIT THESE NUMBERS)
prisma = {
    "records_identified_db": 0,       # e.g., PubMed + other databases
    "records_identified_other": 0,    # e.g., BanglaJOL/manual
    "duplicates_removed": 0,
    "records_screened": 0,
    "records_excluded": 0,
    "full_text_assessed": 0,
    "full_text_excluded": 0,
    "studies_included": 0,
}

# ---------------------------
# 2) EXPORT TABLES
# ---------------------------
df = pd.DataFrame(studies)
df.to_csv(os.path.join(OUTDIR, "table_studies_symptoms_labs.csv"), index=False)

df_skel = pd.DataFrame([skeletal])
df_skel.to_csv(os.path.join(OUTDIR, "table_skeletal_cohort.csv"), index=False)

# ---------------------------
# 3) FIGURE 2: Symptom prevalence across cohorts
# ---------------------------
symptoms = ["bone_pain_pct", "anemia_pct", "fatigue_pct", "renal_impairment_pct", "infections_pct"]
symptom_labels = ["Bone pain", "Anemia", "Fatigue/weakness", "Renal impairment", "Infections"]

plot_df = df[["study"] + symptoms].copy()
plot_df = plot_df.set_index("study")

plt.figure(figsize=(10, 5))
for s, lab in zip(symptoms, symptom_labels):
    plt.plot(plot_df.index, plot_df[s], marker="o", label=lab)
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 100)
plt.ylabel("Prevalence (%)")
plt.title("Symptoms at presentation across Bangladeshi MM cohorts")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Figure_2_symptoms.png"), dpi=600)
plt.close()

# ---------------------------
# 4) FIGURE 3: Skeletal findings (bar chart)
# ---------------------------
sk = skeletal
labels = [
    "Skeletal involvement", "Any lytic lesion", "Any fracture", "Vertebral fracture", "Nerve root compression"
]
values = [
    sk["skeletal_involvement_pct"],
    sk["lytic_any_pct"],
    sk["fracture_any_pct"],
    sk["fracture_vertebral_pct"],
    sk["nerve_root_compression_pct"],
]

plt.figure(figsize=(9, 4.8))
plt.bar(labels, values)
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 100)
plt.ylabel("Percent (%)")
plt.title("Skeletal burden in Bangladesh MM cohort (n=159)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Figure_3_skeletal_burden.png"), dpi=600)
plt.close()

# ---------------------------
# 5) FIGURE 4 (optional): Lab means comparison (only where mean exists)
# ---------------------------
lab_cols = ["hb_mean", "esr_mean", "creatinine_mean", "calcium_mean"]
lab_labels = ["Hb (g/dL)", "ESR (mm/hr)", "Creatinine (mg/dL)", "Calcium (mg/dL)"]

lab_df = df[["study"] + lab_cols].dropna(subset=lab_cols, how="all").set_index("study")

for col, title in zip(lab_cols, lab_labels):
    tmp = lab_df[[col]].dropna()
    if tmp.empty:
        continue
    plt.figure(figsize=(8, 4))
    plt.bar(tmp.index, tmp[col])
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Mean {title} at presentation (reported cohorts)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"Figure_4_{col}.png"), dpi=600)
    plt.close()

# ---------------------------
# 6) FIGURE 1: PRISMA flow (simple box plot)
# ---------------------------
# This produces a simple PRISMA-like diagram as a figure using text boxes.
# Edit prisma dict above first.

import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, fontsize=9):
    rect = patches.Rectangle((x, y), w, h, fill=False, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

fig = plt.figure(figsize=(8.5, 11))
ax = plt.gca()
ax.set_axis_off()

# Coordinates (normalized)
w, h = 0.8, 0.08
x = 0.1

draw_box(ax, x, 0.88, w, h,
         f"Records identified from databases (n={prisma['records_identified_db']})\n"
         f"Records identified from other sources (n={prisma['records_identified_other']})")

draw_box(ax, x, 0.77, w, h, f"Duplicates removed (n={prisma['duplicates_removed']})")
draw_box(ax, x, 0.66, w, h, f"Records screened (n={prisma['records_screened']})")
draw_box(ax, x, 0.55, w, h, f"Records excluded (n={prisma['records_excluded']})")
draw_box(ax, x, 0.44, w, h, f"Full-text assessed (n={prisma['full_text_assessed']})")
draw_box(ax, x, 0.33, w, h, f"Full-text excluded (n={prisma['full_text_excluded']})")
draw_box(ax, x, 0.22, w, h, f"Studies included (n={prisma['studies_included']})")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Figure_1_PRISMA.png"), dpi=600)
plt.close()

print(f"Done. Files saved in: {OUTDIR}")