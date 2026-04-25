import pandas as pd, re, sys

csv = r"E:\Dr. Ahmed\Ongoing\RFLP\Scripts\inputs\assay_table.csv"
df = pd.read_csv(csv)

def die(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(2)

req = ["locus","amplicon_bp","enzyme","wt_fragments_bp","mut_fragments_bp","site_effect","nucleotide_mutation"]
for c in req:
    if c not in df.columns:
        die(f"Missing column: {c}")

df["locus"] = df["locus"].astype(str).str.strip().str.replace("–","-", regex=False)

bad = df[df[req].isna().any(axis=1)]
if len(bad):
    die("NA in required fields:\n" + bad[["locus"]+req].to_string(index=False))

if df["locus"].nunique() != 25:
    die(f"Expected 25 unique loci, found {df['locus'].nunique()}")

def parse_sets(s):
    parts = str(s).split("|")
    sets = []
    for p in parts:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        if nums: sets.append(nums)
    return sets

def any_sum_ok(amp, sets):
    amp = int(amp)
    return any(sum(nums) == amp for nums in sets)

for _, r in df.iterrows():
    amp = r["amplicon_bp"]
    if not any_sum_ok(amp, parse_sets(r["wt_fragments_bp"])):
        die(f"{r['locus']}: WT fragments don't sum to amplicon (amp={amp}, wt={r['wt_fragments_bp']})")
    if not any_sum_ok(amp, parse_sets(r["mut_fragments_bp"])):
        die(f"{r['locus']}: MUT fragments don't sum to amplicon (amp={amp}, mut={r['mut_fragments_bp']})")

def must(locus, **contains):
    sub = df[df["locus"].eq(locus)]
    if len(sub) != 1:
        die(f"Expected 1 row for {locus}, found {len(sub)}")
    row = sub.iloc[0]
    for k,v in contains.items():
        if str(v) not in str(row[k]):
            die(f"{locus}: {k} expected to contain '{v}' but got '{row[k]}'")

# Hard truth checks (these killed you earlier when K417N/W152C were wrong)
must("W152C", amplicon_bp="179", enzyme="BccI", wt_fragments_bp="93", mut_fragments_bp="179")
must("K417N", amplicon_bp="126", enzyme="SspI", wt_fragments_bp="126", mut_fragments_bp="52", site_effect="gain")
must("VG445PS", amplicon_bp="441", enzyme="BstNI")
must("A701V", amplicon_bp="156", enzyme="BsgI")
must("G142D", amplicon_bp="205", enzyme="BslI")

print("\n[PASS] assay_table.csv validated successfully.")
print("Rows:", len(df), "| Unique loci:", df["locus"].nunique())
