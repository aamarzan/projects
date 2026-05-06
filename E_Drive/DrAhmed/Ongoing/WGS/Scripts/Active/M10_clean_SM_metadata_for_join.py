import os
import csv
import re

WORK = "/mnt/e/DrAhmed/Ongoing/WGS"
META = f"{WORK}/Metadata"
INCSV = f"{META}/SM_Metadata_Shahad_Sheet2.csv"
OUTCSV = f"{META}/SM_Metadata_Shahad_clean_for_join.csv"

if not os.path.isfile(INCSV):
    raise SystemExit(f"Missing input CSV: {INCSV}")

def norm(x):
    return str(x).strip()

def normalize_sm_sample(raw):
    s = norm(raw)
    if s == "":
        return ""
    s = s.replace(" ", "").replace("\t", "")
    s = s.replace("__", "_")

    # already looks like SM_107 or SM-051
    if re.match(r"^SM[_-]?\d+$", s, flags=re.I):
        s = s.upper()
        if "_" not in s and "-" not in s:
            num = re.sub(r"^SM", "", s, flags=re.I)
            if len(num) >= 3:
                return f"SM_{num}"
            return f"SM_{num}"
        return s.replace("SM-", "SM-").replace("SM_", "SM_")

    # pure number -> assume Serratia metadata sample numbering
    if re.match(r"^\d+$", s):
        num = int(s)
        if num < 100:
            return f"SM-{num:03d}"
        return f"SM_{num}"

    return s

rows = []
with open(INCSV, "r", encoding="utf-8", errors="replace") as f:
    rr = list(csv.reader(f))

header_row_idx = None
for i, row in enumerate(rr):
    low = [norm(x).lower() for x in row]
    if "sample no." in low or "sample no" in low:
        header_row_idx = i
        break

if header_row_idx is None:
    raise SystemExit("Could not find the SAMPLE NO. header row in the SM metadata CSV.")

header = rr[header_row_idx]
colmap = {norm(v).lower(): idx for idx, v in enumerate(header)}

def idx_for(cands):
    for c in cands:
        for k, v in colmap.items():
            if c.lower() == k or c.lower() in k:
                return v
    return None

i_sample   = idx_for(["sample no", "sample no."])
i_mrn      = idx_for(["mrn"])
i_gender   = idx_for(["gender", "sex"])
i_age      = idx_for(["age"])
i_date     = idx_for(["date of isolation", "isolation date"])
i_specimen = idx_for(["specimen", "site"])

with open(OUTCSV, "w", newline="", encoding="utf-8") as out:
    w = csv.DictWriter(out, fieldnames=[
        "SampleRaw", "Sample", "MRN", "Gender", "Age", "DateOfIsolation", "Specimen", "MetadataSource"
    ])
    w.writeheader()

    for row in rr[header_row_idx + 1:]:
        if not any(norm(x) for x in row):
            continue

        sample_raw = row[i_sample] if i_sample is not None and i_sample < len(row) else ""
        sample = normalize_sm_sample(sample_raw)
        if sample == "":
            continue

        outrow = {
            "SampleRaw": sample_raw,
            "Sample": sample,
            "MRN": row[i_mrn] if i_mrn is not None and i_mrn < len(row) else "",
            "Gender": row[i_gender] if i_gender is not None and i_gender < len(row) else "",
            "Age": row[i_age] if i_age is not None and i_age < len(row) else "",
            "DateOfIsolation": row[i_date] if i_date is not None and i_date < len(row) else "",
            "Specimen": row[i_specimen] if i_specimen is not None and i_specimen < len(row) else "",
            "MetadataSource": "SM_Metadata_Shahad_Sheet2.csv",
        }
        w.writerow(outrow)

print("Wrote cleaned SM metadata:", OUTCSV)