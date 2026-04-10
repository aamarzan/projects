import pandas as pd

p="assay_table.csv"
df=pd.read_csv(p)

# enzyme canonicalization to match patched tables
fix_enzyme={
  "K1191N":"ApoI-HF",
  "L452R":"AgeI-HF",
  "K417N":"SspI-HF",
  "S413R":"BtsI-V2",
}
df.loc[df["locus"].isin(fix_enzyme.keys()),"enzyme"]=df["locus"].map(fix_enzyme).fillna(df["enzyme"])

# locus naming normalization
df["locus"]=df["locus"].replace({"ERS31_33del":"ERS31-33del"})

# fill mut_amplicon_bp with amplicon_bp when missing
df["mut_amplicon_bp"]=df["mut_amplicon_bp"].fillna(df["amplicon_bp"])

df.to_csv(p,index=False)
print("Patched:",p)
