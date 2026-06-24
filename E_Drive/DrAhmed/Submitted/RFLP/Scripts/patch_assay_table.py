import pandas as pd
from pathlib import Path

csv_path = Path("assay_table.csv")
df = pd.read_csv(csv_path)

def norm_locus(s: str) -> str:
    return str(s).strip().replace("–","-").replace(" ", "").replace("\t","").replace("\n","")

df["locus"] = df["locus"].map(norm_locus)

def set_row(locus: str, **kwargs):
    m = df["locus"].eq(locus)
    n = int(m.sum())
    if n != 1:
        raise SystemExit(f"[ERROR] Expected 1 row for {locus}, found {n}")
    for k, v in kwargs.items():
        if k not in df.columns:
            raise SystemExit(f"[ERROR] Column not found: {k}")
        df.loc[m, k] = v

# -----------------------------
# Fill missing + correct mismatches vs thesis tables
# -----------------------------

# Table 8 — VG445PS
set_row(
    "VG445PS",
    amplicon_bp=441,
    enzyme="BstNI",
    wt_fragments_bp="441",
    mut_fragments_bp="59+382",
    site_effect="gain",
    nucleotide_mutation="1333_1336delGTTGinsCCTA",
    notes="Thesis Table 8: WT undigested (441); mutant digested (59+382) using BstNI."
)

# Table 12 — A701V
set_row(
    "A701V",
    amplicon_bp=156,
    enzyme="BsgI",
    wt_fragments_bp="91+65",
    mut_fragments_bp="156",
    site_effect="loss",
    nucleotide_mutation="C23664T",
    notes="Thesis Table 12: WT digested (91+65); mutant undigested (156) using BsgI."
)

# Table 21 — G142D (enzyme printed 'Bslt' in table; canonicalize to BslI but note it)
set_row(
    "G142D",
    amplicon_bp=205,
    enzyme="BslI",
    wt_fragments_bp="157+48",
    mut_fragments_bp="205",
    site_effect="loss",
    nucleotide_mutation="G21987A",
    notes="Thesis Table 21 shows enzyme as 'Bslt'; treated as BslI (typo I↔t). WT 157+48; mutant 205."
)

# Table 22 — W152C
set_row(
    "W152C",
    amplicon_bp=179,
    enzyme="BccI",
    wt_fragments_bp="93+86",
    mut_fragments_bp="179",
    site_effect="loss",
    nucleotide_mutation="G22018C",
    notes="Thesis Table 22: WT digested (93+86); mutant undigested (179) using BccI."
)

# Table 24 — K417N (conditional mutant)
set_row(
    "K417N",
    amplicon_bp=126,
    enzyme="SspI",
    wt_fragments_bp="126",
    mut_fragments_bp="52+74|126",
    site_effect="gain|none",
    nucleotide_mutation="G22813T/G22813C",
    notes="Thesis Table 24: conditional. nt–T mutant digested 52+74; nt–C mutant undigested 126. WT undigested 126."
)

# HV69-70del — internal thesis mismatch: Table prints 181+84 but paragraph implies 181+48 (must sum to 229)
set_row(
    "HV69-70del",
    amplicon_bp=229,
    mut_fragments_bp="181+48",
    notes="Thesis inconsistency: Table 18 prints 181+84, but 181+48 matches amplicon 229 (paragraph). Using 181+48."
)

# -----------------------------
# Basic integrity checks
# -----------------------------
req = ["amplicon_bp","enzyme","wt_fragments_bp","mut_fragments_bp","site_effect","nucleotide_mutation"]
missing = df[df[req].isna().any(axis=1)][["locus"] + req]
if len(missing):
    print("\n[WARNING] Still missing required fields for these loci:")
    print(missing.to_string(index=False))
else:
    print("\n[OK] No missing required fields in:", req)

# Show the key fixed rows
print("\n[CHECK] Patched rows:")
print(df[df["locus"].isin(["VG445PS","A701V","G142D","W152C","K417N","HV69-70del"])][
    ["locus","amplicon_bp","enzyme","wt_fragments_bp","mut_fragments_bp","site_effect","nucleotide_mutation"]
].to_string(index=False))

df.to_csv(csv_path, index=False)
print("\nSaved:", csv_path.resolve())
