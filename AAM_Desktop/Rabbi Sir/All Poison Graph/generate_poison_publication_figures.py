#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_poison_publication_figures.py
Publication-quality, version-controlled figure pipeline for poison.xlsx.
Run:
  python generate_poison_publication_figures.py --input poison.xlsx --outdir poison_figures_output
"""
from __future__ import annotations
import argparse, collections, dataclasses, datetime as dt, hashlib, html, json, logging, math, os, platform, re, shutil, subprocess, sys, traceback, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='Data Validation extension is not supported*')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import FancyBboxPatch, Rectangle

try:
    import openpyxl  # noqa
    HAVE_OPENPYXL=True
except Exception:
    HAVE_OPENPYXL=False
try:
    import scipy
    from scipy import stats
    from scipy.cluster.hierarchy import linkage, leaves_list
    HAVE_SCIPY=True
except Exception:
    HAVE_SCIPY=False
try:
    import statsmodels.api as sm
    HAVE_STATSMODELS=True
except Exception:
    HAVE_STATSMODELS=False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_curve, auc, brier_score_loss
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAVE_SKLEARN=True
except Exception:
    HAVE_SKLEARN=False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY=True
except Exception:
    HAVE_PLOTLY=False
try:
    import networkx as nx
    HAVE_NETWORKX=True
except Exception:
    HAVE_NETWORKX=False
try:
    from upsetplot import UpSet, from_indicators
    HAVE_UPSETPLOT=True
except Exception:
    HAVE_UPSETPLOT=False
try:
    import yaml
    HAVE_YAML=True
except Exception:
    HAVE_YAML=False

RANDOM_SEED=20260613
np.random.seed(RANDOM_SEED)
MISSING_TOKENS={""," ","nan","none","null","na","n/a","nil","#ref!","#n/a","#value!","#div/0!","#name?","--","-","_","__","___","____","________","no information in file","no information","not available","unknown","unk","follow up","phone switched off","wrong phone number provided","did not receive phone","unreachable","no phone number available"}
PII={"Patient's Name","Patient’s Name","Contact number","Registration Number","Address"}
SYMPTOMS=['Vomited after ingestion','Fever','Vomiting','Diarrhoea','Abdominal pain','Abdominal distension','Cough','Shortness of breath','Heart burn','Oral ulcers','Leg swelling','Reduced urine output','Jaundice','Unconsciousness','Convulsion','Chest pain','Bleeding tendency','Shock']
LAB_MAP={'Total WBC count(/mm3)':'wbc_count_mm3','Neutrophil (%)':'neutrophil_percent','Lymphocytes(%)':'lymphocyte_percent','Platelates(/mm3)':'platelets_mm3','S. creatinine(mg/dL)':'creatinine_mg_dl','Na+':'sodium_mmol_l','k+':'potassium_mmol_l','Cl-':'chloride_mmol_l','Ca2+':'calcium_mmol_l','Mg2+':'magnesium_mmol_l','S.uric acid(mg/dL)':'uric_acid_mg_dl','Random blood sugar (mmol/L)':'random_glucose_mmol_l','S.creatinine kinase(CK-MB)':'ck_mb','pH':'ph','HCO3-':'hco3_mmol_l','PCO2':'pco2_mm_hg','PO2':'po2_mm_hg','Anion gap':'anion_gap','Hemoglobulin':'hemoglobin_g_dl','S.bilirubin(mg/dL)':'bilirubin_mg_dl','S. amylase(u/L)':'amylase_u_l','SGPT':'sgpt_u_l','SGOT':'sgot_u_l','S. troponin I':'troponin_i'}
NUM_MAP={'age_years':'Age (in years)','amount_ingested_ml':'Amount of ingestion (ml)','time_symptom_onset_hrs':'Time to symptoms onset (in hrs)','time_presentation_hrs':'Time to presentation (in hrs)','temperature_c':'Temperature','pulse_bpm':'Pulse (beats/min)','spo2_percent':'SpO2','gcs':'GCS','respiratory_rate':'Respiratory rate','fluid_l':'Total amount of fluid (in L)','oxygen_l_min':'Oxygen(highest L/min)','dialysis_cycles':'Dialysis (No of cycle)'}
PLAUSIBLE={'age_years':(0,120),'amount_ingested_ml':(0,5000),'time_symptom_onset_hrs':(0,720),'time_presentation_hrs':(0,720),'delay_ingestion_to_admission_hrs':(-24,720),'temperature_c':(25,45),'pulse_bpm':(20,250),'spo2_percent':(40,100),'gcs':(3,15),'respiratory_rate':(4,80),'bp_systolic':(40,300),'bp_diastolic':(20,180),'fluid_l':(0,30),'oxygen_l_min':(0,100),'dialysis_cycles':(0,30),'wbc_count_mm3':(500,100000),'neutrophil_percent':(0,100),'lymphocyte_percent':(0,100),'platelets_mm3':(1000,1000000),'creatinine_mg_dl':(0.1,25),'sodium_mmol_l':(90,190),'potassium_mmol_l':(1,10),'chloride_mmol_l':(60,140),'calcium_mmol_l':(0.5,5),'magnesium_mmol_l':(0.1,10),'uric_acid_mg_dl':(0.1,30),'random_glucose_mmol_l':(0.5,60),'ck_mb':(0,10000),'ph':(6.5,7.8),'hco3_mmol_l':(0,60),'pco2_mm_hg':(5,150),'po2_mm_hg':(10,700),'anion_gap':(-10,80),'hemoglobin_g_dl':(1,25),'bilirubin_mg_dl':(0,60),'amylase_u_l':(0,10000),'sgpt_u_l':(0,20000),'sgot_u_l':(0,20000),'troponin_i':(0,100000)}
AGE_BINS=[-np.inf,4,12,17,24,39,59,np.inf]
AGE_LABELS=['0–4','5–12','13–17','18–24','25–39','40–59','≥60']
TIME_BINS=[-np.inf,1,3,6,12,24,48,np.inf]
TIME_LABELS=['≤1 h','>1–3 h','>3–6 h','>6–12 h','>12–24 h','>24–48 h','>48 h']
GRADIENTS={
'deep_navy_teal_mint':['#071A2D','#006D77','#83C5BE','#EDF6F9'],'charcoal_indigo_lavender':['#1F2933','#3D348B','#8E7DBE','#E6E1F4'],'burgundy_rose_blush':['#4A0D1A','#9D174D','#E79AB3','#FFF1F2'],'forest_jade_sage':['#12372A','#1B8A5A','#96CDB3','#EEF8F1'],'plum_violet_lilac':['#2E073F','#7A1CAC','#C8A1E0','#F4EAFE'],'sienna_amber_cream':['#5B2C06','#C2691D','#E9B44C','#FFF8E7'],'slate_cyan_pearl':['#22333B','#2F7A8A','#78CDD7','#F5FBFC'],'espresso_bronze_sand':['#2B2118','#8B5E34','#C9A66B','#FAF3E3'],'crimson_coral_peach':['#4C0519','#BE3144','#F9735B','#FFE7D6'],'midnight_royal_ice':['#020617','#1D4ED8','#93C5FD','#EFF6FF'],'ink_emerald_seafoam':['#0B1320','#047857','#6EE7B7','#ECFDF5'],'aubergine_mauve_shell':['#2D1B2F','#7C3AED','#C4B5FD','#F7F2FF'],'graphite_steel_cloud':['#111827','#475569','#94A3B8','#F1F5F9'],'olive_moss_ivory':['#283618','#606C38','#BBC08B','#FEFAE0'],'ocean_denim_mist':['#001219','#005F73','#94D2BD','#E9F5F2'],'garnet_raspberry_champagne':['#3A0A12','#A4133C','#FFB3C1','#FFF0F3'],'copper_apricot_linen':['#3C2415','#B85C38','#F4A261','#FFF4E6'],'pine_turquoise_foam':['#082E2A','#138A8A','#7BDFF2','#EDFFFF'],'blackberry_bluebell':['#160F29','#246A73','#A2D2FF','#F2F7FF'],'claret_orchid_porcelain':['#351431','#B5179E','#E0AAFF','#FBF7FF'],'marine_greenstone':['#061923','#0E7490','#2DD4BF','#ECFEFF'],'oxide_gold_almond':['#3B1F0B','#A16207','#FACC15','#FEFCE8'],'mulberry_flame_cream':['#2A0A1F','#9F1239','#FB7185','#FFF7ED'],'basalt_amethyst_pearl':['#18181B','#6D28D9','#DDD6FE','#FAFAFA']}
CAT=['#0F172A','#0E7490','#7C3AED','#BE123C','#047857','#B45309','#475569','#2563EB','#9D174D','#15803D','#64748B','#0369A1']

@dataclasses.dataclass
class FigRec:
    figure_id:str; title:str; analysis_type:str; variables:str; folder:str; basename:str; png:str; pdf:str; svg:str; caption:str; gradient:str; palette:str; n:int; notes:str=''
class Ctx:
    def __init__(self,args):
        self.args=args; self.input=Path(args.input).resolve(); self.out=Path(args.outdir).resolve(); self.run_id=dt.datetime.now().strftime('run_%Y%m%d_%H%M%S'); self.started=dt.datetime.now().isoformat(timespec='seconds')
        self.fig_i=0; self.registry=[]; self.skipped=[]; self.warnings=[]; self.decisions=[]; self.dups={}; self.pal_count=collections.Counter(); self.dirs={}; self.log=logging.getLogger('poisonfig')
    def dir(self,k): return self.dirs[k]
    def next_id(self): self.fig_i+=1; return f'F{self.fig_i:03d}'
    def skip(self,a,r,vars=None): self.skipped.append({'analysis':a,'reason':r,'variables':list(vars or [])}); self.log.info('SKIP | %s | %s',a,r)
    def warn(self,m): self.warnings.append(m); self.log.warning(m)
    def decide(self,a,d): self.decisions.append({'action':a,'details':d})

def slug(s,n=90):
    s=re.sub(r'[^a-z0-9]+','_',str(s).lower()).strip('_'); return (s[:n].strip('_') or 'x')
def sha256_file(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_json(p,o): Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def ismiss(x):
    if x is None: return True
    try:
        if pd.isna(x): return True
    except Exception: pass
    return str(x).replace('\xa0',' ').strip().lower() in MISSING_TOKENS
def norm_space(x):
    if ismiss(x): return np.nan
    if isinstance(x,str):
        x=re.sub(r'\s+',' ',x.replace('\xa0',' ')).strip()
        return np.nan if x.lower() in MISSING_TOKENS else x
    return x
def title_safe(x):
    if ismiss(x): return np.nan
    s=str(x).strip(); low=s.lower().replace(' ','')
    ab={'dmch':'DMCH','cmch':'CMCH','rmch':'RMCH','kmch':'KMCH','mmch':'MMCH','szmch':'SZMCH','somch':'SOMCH','sbmch':'SBMCH','rpmch':'RpMCH','jmch':'JMCH','opd':'OPD','ipd':'IPD','icu':'ICU'}
    return ab.get(low, s if len(s)<=4 and s.isupper() else s.title())
def setup(ctx):
    if ctx.out.exists() and any(ctx.out.iterdir()) and not ctx.args.overwrite: raise SystemExit(f'Output folder exists and is not empty: {ctx.out}. Use --overwrite.')
    if ctx.out.exists() and ctx.args.overwrite: shutil.rmtree(ctx.out)
    names={'logs':'00_logs','clean':'01_clean_data','tables':'02_tables','main':'03_figures_main','supp':'04_figures_supplementary','html':'05_interactive_html','panels':'06_manuscript_panels','ver':'07_version_control'}
    for k,v in names.items():
        ctx.dirs[k]=ctx.out/v; ctx.dirs[k].mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.FileHandler(ctx.dir('logs')/f'{ctx.run_id}_run.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)])
    logging.getLogger('matplotlib').setLevel(logging.WARNING); logging.getLogger('fontTools').setLevel(logging.WARNING); logging.getLogger('PIL').setLevel(logging.WARNING)
    ctx.log=logging.getLogger('poisonfig')
    logging.getLogger('matplotlib').setLevel(logging.WARNING); logging.getLogger('fontTools').setLevel(logging.WARNING); logging.getLogger('PIL').setLevel(logging.WARNING)
def dedup(headers):
    cnt=collections.Counter(); cols=[]; dd=[]; dups={}
    for i,h in enumerate(headers,1):
        base=f'Unnamed_{i}' if h is None or str(h).strip()=='' else str(h).strip()
        cnt[base]+=1; occ=cnt[base]; name=base if occ==1 else f'{base}__dup{occ}'; cols.append(name)
        dd.append({'column_index_1based':i,'original_header':base,'analysis_column':name,'occurrence':occ,'is_duplicate_header':occ>1,'public_excluded_pii':base in PII})
    dups={k:int(v) for k,v in cnt.items() if v>1}; return cols,dups,pd.DataFrame(dd)
def read_sheet(path,sheet,max_rows=None):
    raw=pd.read_excel(path,sheet_name=sheet,header=None,dtype=object,engine='openpyxl')
    cols,dups,dd=dedup(list(raw.iloc[0])); df=raw.iloc[1:].copy(); df.columns=cols
    if max_rows: df=df.head(max_rows).copy()
    return df,dups,dd
def load_workbook(ctx):
    if not HAVE_OPENPYXL: raise RuntimeError('openpyxl is required for .xlsx reading and duplicate-header preservation.')
    xls=pd.ExcelFile(ctx.input,engine='openpyxl'); sheets={}; dds=[]; main=None
    for sh in xls.sheet_names:
        df,dups,dd=read_sheet(ctx.input,sh,ctx.args.max_rows if sh=='All Poison Data' else None)
        sheets[sh]=df; dd.insert(0,'sheet',sh); dds.append(dd)
        if dups: ctx.dups[sh]=dups
        ctx.log.info('Loaded %s: %s rows x %s cols',sh,df.shape[0],df.shape[1])
        if sh=='All Poison Data': main=df
    if main is None: raise RuntimeError("Sheet 'All Poison Data' not found")
    return main,sheets,pd.concat(dds,ignore_index=True)

# ----------------------------- cleaning helpers -----------------------------
def binary(x):
    if ismiss(x): return np.nan
    s=str(x).strip().lower()
    yes={'yes','y','1','true','present','positive','pos','done','given','needed','required','survived','death','died','expired'}
    no={'no','n','0','false','absent','negative','neg','not done','not given','nil','none'}
    if s in yes: return 1.0
    if s in no: return 0.0
    if re.search(r'\byes\b|\bpresent\b|\bdone\b|\bgiven\b|\bdied\b|\bdeath\b|expired',s): return 1.0
    if re.search(r'\bno\b|\babsent\b|\bnot\b|\bnegative\b',s): return 0.0
    return np.nan

def sex_clean(x):
    if ismiss(x): return np.nan
    s=str(x).strip().lower()
    if s.startswith('m') or s in {'boy','male patient'}: return 'Male'
    if s.startswith('f') or s in {'girl','female patient'}: return 'Female'
    return title_safe(x)

def poison_clean(x):
    if ismiss(x): return 'Unknown/Other'
    s=str(x).strip().lower(); s=re.sub(r'[^a-z0-9+ /-]+',' ',s); s=re.sub(r'\s+',' ',s).strip()
    rules=[
        ('Organophosphate / pesticide',r'\b(opc|op|organophosphate|organophosphorus|chlorpyrifos|pesticide|insecticide|cypermethrin|pyrethroid|carbamate|malathion|diazinon)\b'),
        ('Sedative / benzodiazepine',r'\b(sedative|benzodiazepine|benzo|diazepam|alprazolam|clonazepam|lorazepam|sleeping|barbiturate)\b'),
        ('Household corrosive / cleaner',r'\b(herpic|harpic|toilet cleaner|household|corrosive|acid|bleach|cleaner|phenyl|detergent|sodium hypochlorite)\b'),
        ('Paraquat / herbicide',r'\b(paraquat|herbicide|weedicide|gramoxone)\b'),
        ('Aluminium phosphide / rodenticide',r'\b(aluminium phosphide|aluminum phosphide|phosphide|gas tablet|rat killer|rodenticide|zinc phosphide)\b'),
        ('Drug overdose / multidrug',r'\b(drug overdose|multidrug|multi drug|multiple drug|polypharmacy|paracetamol|nsaid|antidepressant|antipsychotic|medicine|tablet)\b'),
        ('Street / unknown poisoning',r'\b(street|unknown|unidentified|food poisoning|mixed|other)\b'),
        ('Animal / insect envenomation',r'\b(snake|bee|scorpion|bite|sting|wasp|animal)\b'),
        ('Alcohol / substance',r'\b(alcohol|ethanol|methanol|spirit|yaba|ganja|opioid|heroin)\b')]
    for lab,pat in rules:
        if re.search(pat,s): return lab
    return title_safe(s)

def component_clean(x):
    if ismiss(x): return 'Unknown/Other'
    s=str(x).strip().lower(); s=re.sub(r'[^a-z0-9+ /-]+',' ',s); s=re.sub(r'\s+',' ',s).strip()
    rules=[('Benzodiazepine',r'benzodiazepine|diazepam|alprazolam|clonazepam|lorazepam'),('Toilet cleaner / Harpic',r'herpic|harpic|toilet cleaner'),('OPC',r'\bopc\b|organophosphate|chlorpyrifos'),('Pyrethroid',r'pyrethroid|cypermethrin'),('Paracetamol',r'paracetamol|acetaminophen'),('Paraquat',r'paraquat|gramoxone'),('Rat killer / phosphide',r'rat killer|phosphide|rodenticide|gas tablet'),('Corrosive acid/alkali',r'corrosive|acid|bleach|alkali'),('Unknown/Other',r'unknown|other|not known')]
    for lab,pat in rules:
        if re.search(pat,s): return lab
    return title_safe(s)

def parse_date(x):
    if ismiss(x): return pd.NaT
    if isinstance(x,(pd.Timestamp,dt.datetime,dt.date)): return pd.to_datetime(x,errors='coerce')
    if isinstance(x,(int,float,np.integer,np.floating)) and not pd.isna(x):
        if 20000 < float(x) < 60000:
            try: return pd.to_datetime('1899-12-30')+pd.to_timedelta(float(x),unit='D')
            except Exception: return pd.NaT
    s=str(x).strip()
    s=re.sub(r'^[A-Za-z]+,?\s+','',s)
    for dayfirst in (True,False):
        d=pd.to_datetime(s,errors='coerce',dayfirst=dayfirst)
        if not pd.isna(d):
            if 1980 <= d.year <= dt.datetime.now().year+1: return d
    return pd.NaT

def num(x):
    if ismiss(x): return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)): return float(x)
    s=str(x).strip().lower().replace(',','')
    if s in MISSING_TOKENS or s.startswith('#'): return np.nan
    s=s.replace('mmhg','').replace('mg/dl','').replace('mmol/l','').replace('ml','').replace('l/min','').replace('litre','').replace('liter','')
    m=re.findall(r'-?\d+(?:\.\d+)?',s)
    if not m: return np.nan
    vals=[float(v) for v in m]
    if len(vals)>=2 and re.search(r'\bto\b|[-–]|/',s) and not re.search(r'\d+\s*/\s*\d+',s): return float(np.mean(vals[:2]))
    return vals[0]

def parse_bp(x):
    if ismiss(x): return (np.nan,np.nan)
    s=str(x).replace('\\','/').replace('|','/').lower()
    vals=[float(v) for v in re.findall(r'\d+(?:\.\d+)?',s)]
    if len(vals)>=2:
        sysv=max(vals[0],vals[1]) if vals[0]<vals[1] else vals[0]; diav=min(vals[0],vals[1]) if vals[0]<vals[1] else vals[1]
        return sysv,diav
    return (np.nan,np.nan)

def season(m):
    if pd.isna(m): return np.nan
    m=int(m)
    if m in [3,4,5]: return 'Pre-monsoon'
    if m in [6,7,8,9]: return 'Monsoon'
    if m in [10,11]: return 'Post-monsoon'
    return 'Winter'

def clean_dataset(ctx, raw:pd.DataFrame, sheets:Dict[str,pd.DataFrame]):
    df=raw.copy(); dq=[]
    # formula/error tokens in raw cells
    for c in df.columns:
        bad=df[c].astype(str).str.strip().str.lower().isin({'#ref!','#value!','#div/0!','#name?','#n/a'}).sum()
        if bad: dq.append({'issue':'Excel/formula error token','column':c,'n':int(bad),'detail':'Converted to missing in analysis.'})
    # normalize object columns
    for c in df.columns:
        if df[c].dtype=='object': df[c]=df[c].map(norm_space)
    pii_cols=[c for c in df.columns if c.split('__dup')[0] in PII]
    public=df.drop(columns=pii_cols,errors='ignore').copy(); ctx.decide('PII exclusion',f'Dropped from public analysis/export: {pii_cols}')
    # standard categorical variables
    col=lambda name: name if name in public.columns else None
    if col('Sex'): public['sex_clean']=public['Sex'].map(sex_clean)
    if col('Study site'): public['study_site_clean']=public['Study site'].map(title_safe)
    if col('Living area'): public['living_area_clean']=public['Living area'].map(title_safe)
    if col('Presentation area'): public['presentation_area_clean']=public['Presentation area'].map(title_safe)
    if col('Occupation'): public['occupation_clean']=public['Occupation'].map(title_safe)
    if col('Types of poisoning'):
        public['poison_type_raw']=public['Types of poisoning']
        public['poison_type_clean']=public['Types of poisoning'].map(poison_clean)
    else: public['poison_type_clean']='Unknown/Other'
    if col('Name of the specific component'):
        public['specific_component_raw']=public['Name of the specific component']
        public['specific_component_clean']=public['Name of the specific component'].map(component_clean)
    else: public['specific_component_clean']='Unknown/Other'
    # dates
    date_candidates=['Date of admission','Date of ingestion','Absconded Date','Date of death','Death date','Death (mention the date)']+[c for c in public.columns if 'date' in c.lower() and c not in []]
    seen=set()
    for c in date_candidates:
        if c in public.columns and c not in seen:
            seen.add(c); new='date_'+slug(c,50); public[new]=public[c].map(parse_date)
    adm='date_date_of_admission'; ing='date_date_of_ingestion'
    if adm in public.columns:
        public['admission_year']=public[adm].dt.year; public['admission_month']=public[adm].dt.to_period('M').astype(str).where(public[adm].notna(),np.nan)
        public['admission_month_num']=public[adm].dt.month; public['admission_day_of_week']=public[adm].dt.day_name(); public['admission_season']=public['admission_month_num'].map(season)
    if ing in public.columns:
        public['ingestion_year']=public[ing].dt.year; public['ingestion_month']=public[ing].dt.to_period('M').astype(str).where(public[ing].notna(),np.nan)
    if adm in public.columns and ing in public.columns:
        public['delay_ingestion_to_admission_hrs']=(public[adm]-public[ing]).dt.total_seconds()/3600.0
    # numerics
    for new,old in NUM_MAP.items():
        if old in public.columns: public[new]=public[old].map(num)
    if 'Blood pressure' in public.columns:
        bp=public['Blood pressure'].map(parse_bp); public['bp_systolic']=[a for a,b in bp]; public['bp_diastolic']=[b for a,b in bp]
    for old,new in LAB_MAP.items():
        if old in public.columns: public[new]=public[old].map(num)
    # range flags without deletion
    for c,(lo,hi) in PLAUSIBLE.items():
        if c in public.columns:
            bad=public[c].notna() & ((public[c]<lo)|(public[c]>hi))
            if bad.any(): dq.append({'issue':'Clinically implausible numeric value','column':c,'n':int(bad.sum()),'detail':f'Plausible range {lo}–{hi}; values retained but flagged.'})
            public[c+'_implausible_flag']=bad.astype(int)
            public[c+'_analysis']=public[c].where(~bad,np.nan)
    # binaries
    base_sym=[]
    for s in SYMPTOMS:
        if s in public.columns:
            new=slug(s)+'_bin'; public[new]=public[s].map(binary); base_sym.append(new)
    for c in ['NG suction','Ventilation support','Operation','Survived without complications','Death','Absconded','Back to normal health','Not entirely back to normal health','Requires help in daily activities']:
        if c in public.columns: public[slug(c)+'_bin']=public[c].map(binary)
    # repeated medication fields
    med_names=[c for c in public.columns if c.split('__dup')[0]=='Name']
    dose_cols=[c for c in public.columns if c.split('__dup')[0]=='Dose' or c=='Dose (amp)']
    dur_cols=[c for c in public.columns if c.split('__dup')[0]=='Duration']
    ctx.decide('Repeated treatment field handling',f'Medication names={med_names}; dose={dose_cols}; duration={dur_cols}. Kept as separate columns and summarized jointly.')
    # derived classes
    if 'age_years_analysis' in public.columns:
        public['age_group']=pd.cut(public['age_years_analysis'],AGE_BINS,labels=AGE_LABELS,right=True)
    if 'time_presentation_hrs_analysis' in public.columns:
        public['time_to_presentation_category']=pd.cut(public['time_presentation_hrs_analysis'],TIME_BINS,labels=TIME_LABELS)
    if 'time_symptom_onset_hrs_analysis' in public.columns:
        public['time_to_symptom_onset_category']=pd.cut(public['time_symptom_onset_hrs_analysis'],TIME_BINS,labels=TIME_LABELS)
    if 'delay_ingestion_to_admission_hrs_analysis' not in public.columns and 'delay_ingestion_to_admission_hrs' in public.columns:
        lo,hi=PLAUSIBLE['delay_ingestion_to_admission_hrs']; bad=public['delay_ingestion_to_admission_hrs'].notna() & ((public['delay_ingestion_to_admission_hrs']<lo)|(public['delay_ingestion_to_admission_hrs']>hi)); public['delay_ingestion_to_admission_hrs_analysis']=public['delay_ingestion_to_admission_hrs'].where(~bad,np.nan)
    public['symptom_burden_score']=public[base_sym].sum(axis=1,min_count=1) if base_sym else np.nan
    public['death_bin']=public.get('death_bin',pd.Series(np.nan,index=public.index))
    date_death_cols=[c for c in public.columns if c.startswith('date_') and ('death' in c)]
    if date_death_cols: public['death_bin']=np.where(public[date_death_cols].notna().any(axis=1),1,public['death_bin'])
    public['absconded_bin']=public.get('absconded_bin',pd.Series(np.nan,index=public.index))
    absdate=[c for c in public.columns if c.startswith('date_') and 'absconded' in c]
    if absdate: public['absconded_bin']=np.where(public[absdate].notna().any(axis=1),1,public['absconded_bin'])
    public['any_complication_bin']=np.nan
    if 'Complications name' in public.columns: public['any_complication_bin']=public['Complications name'].map(lambda x: 0.0 if ismiss(x) else 1.0)
    public['altered_consciousness']=np.nanmax(np.vstack([public.get('unconsciousness_bin',pd.Series(np.nan,index=public.index)), public.get('low_gcs_dummy',pd.Series(np.nan,index=public.index))]),axis=0) if 'unconsciousness_bin' in public.columns else np.nan
    public['convulsion_feature']=public.get('convulsion_bin',np.nan)
    public['shock_feature']=public.get('shock_bin',np.nan)
    if 'gcs_analysis' in public.columns:
        public['low_gcs_bin']=(public['gcs_analysis']<8).astype(float).where(public['gcs_analysis'].notna(),np.nan)
    if 'spo2_percent_analysis' in public.columns:
        public['hypoxia_bin']=(public['spo2_percent_analysis']<90).astype(float).where(public['spo2_percent_analysis'].notna(),np.nan)
    if 'bp_systolic_analysis' in public.columns:
        public['hypotension_bin']=(public['bp_systolic_analysis']<90).astype(float).where(public['bp_systolic_analysis'].notna(),np.nan)
    if 'time_presentation_hrs_analysis' in public.columns:
        public['delayed_presentation_bin']=(public['time_presentation_hrs_analysis']>6).astype(float).where(public['time_presentation_hrs_analysis'].notna(),np.nan)
    public['high_risk_poison_group']=public['poison_type_clean'].isin(['Organophosphate / pesticide','Paraquat / herbicide','Aluminium phosphide / rodenticide','Household corrosive / cleaner']).astype(float)
    severe_parts=[c for c in ['death_bin','any_complication_bin','low_gcs_bin','hypoxia_bin','hypotension_bin','shock_bin','convulsion_bin'] if c in public.columns]
    public['severe_outcome_bin']=public[severe_parts].max(axis=1,skipna=True) if severe_parts else np.nan
    tx_parts=[]
    for c in ['fluid_l_analysis','oxygen_l_min_analysis','dialysis_cycles_analysis']:
        if c in public.columns: tx_parts.append((public[c].notna() & (public[c]>0)).astype(float))
    for c in ['ng_suction_bin','ventilation_support_bin','operation_bin']:
        if c in public.columns: tx_parts.append(public[c])
    public['treatment_intensity_score']=pd.concat(tx_parts,axis=1).sum(axis=1,min_count=1) if tx_parts else np.nan
    fcols=[c for c in ['back_to_normal_health_bin','not_entirely_back_to_normal_health_bin','requires_help_in_daily_activities_bin'] if c in public.columns]
    if fcols:
        public['followup_status']='Missing/unknown'
        if 'back_to_normal_health_bin' in public.columns: public.loc[public['back_to_normal_health_bin']==1,'followup_status']='Back to normal health'
        if 'not_entirely_back_to_normal_health_bin' in public.columns: public.loc[public['not_entirely_back_to_normal_health_bin']==1,'followup_status']='Not entirely normal'
        if 'requires_help_in_daily_activities_bin' in public.columns: public.loc[public['requires_help_in_daily_activities_bin']==1,'followup_status']='Requires help'
    else: public['followup_status']=np.nan
    dq.append({'issue':'Duplicate raw headers','column':'All Poison Data','n':sum(ctx.dups.get('All Poison Data',{}).values()),'detail':json.dumps(ctx.dups.get('All Poison Data',{}),ensure_ascii=False)})
    return public,pd.DataFrame(dq)

# ----------------------------- output and plotting core -----------------------------
def package_versions():
    mods={'python':sys.version.split()[0],'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}
    for name,flag in [('scipy',HAVE_SCIPY),('statsmodels',HAVE_STATSMODELS),('sklearn',HAVE_SKLEARN),('plotly',HAVE_PLOTLY),('networkx',HAVE_NETWORKX),('upsetplot',HAVE_UPSETPLOT),('openpyxl',HAVE_OPENPYXL),('yaml',HAVE_YAML)]:
        try: mods[name]=__import__(name).__version__ if flag else 'not installed'
        except Exception: mods[name]='installed/version unknown' if flag else 'not installed'
    return mods

def write_yamlish(path,obj):
    if HAVE_YAML:
        Path(path).write_text(yaml.safe_dump(obj,sort_keys=False,allow_unicode=True),encoding='utf-8')
    else:
        write_json(path,obj)

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':13,'axes.labelsize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':600,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})

def choose_grad(ctx):
    # each gradient used at most 3 times if possible
    for k in GRADIENTS:
        if ctx.pal_count[k] < 3:
            ctx.pal_count[k]+=1; return k
    k=min(GRADIENTS,key=lambda x:ctx.pal_count[x]); ctx.pal_count[k]+=1; return k

def get_cmap(name): return LinearSegmentedColormap.from_list(name,GRADIENTS.get(name,GRADIENTS['deep_navy_teal_mint']))
def grad_cols(name,n):
    cm=get_cmap(name); return [to_hex(cm(i/max(1,n-1))) for i in range(n)]
def pretty(x): return str(x).replace('_',' ').replace(' bin','').replace(' analysis','').title()
def enough(df,vars,minn=20):
    if isinstance(vars,str): vars=[vars]
    miss=[v for v in vars if v not in df.columns]
    if miss: return False,f'Missing columns: {miss}'
    n=df[vars].dropna().shape[0]
    return n>=minn,f'Only {n} non-missing records'

def add_n(ax,n,where='right'):
    ax.text(0.99 if where=='right' else 0.01,1.01,f'n = {n:,}',transform=ax.transAxes,ha='right' if where=='right' else 'left',va='bottom',fontsize=8,color='#475569')

def caption_file(path,txt): Path(str(path)+'.txt').write_text(txt,encoding='utf-8')

def save_fig(ctx,fig,title,analysis_type,variables,folder='supp',basename=None,caption=None,gradient=None,n=0,notes=''):
    fid=ctx.next_id(); basename=basename or slug(fid+'_'+title); gradient=gradient or choose_grad(ctx)
    outdir=ctx.dir(folder); outdir.mkdir(parents=True,exist_ok=True)
    base=outdir/basename
    fig.tight_layout()
    fig.savefig(base.with_suffix('.png'),dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    cap=caption or f'{title}. Descriptive figure based on denominator-aware available records; missing values were excluded from plotted denominators.'
    caption_file(base.with_suffix('.caption'),cap)
    rec=FigRec(fid,title,analysis_type,', '.join(variables if isinstance(variables,(list,tuple)) else [str(variables)]),folder,basename,str(base.with_suffix('.png')),str(base.with_suffix('.pdf')),str(base.with_suffix('.svg')),cap,gradient,';'.join(GRADIENTS.get(gradient,[])),int(n or 0),notes)
    ctx.registry.append(dataclasses.asdict(rec)); return rec

def top_series(s,top=15,other=True):
    vc=s.dropna().astype(str).replace({'nan':np.nan}).dropna().value_counts()
    if other and len(vc)>top:
        rest=vc.iloc[top:].sum(); vc=vc.iloc[:top]; vc.loc['Other']=rest
    return vc

def bar(ctx,df,col,title,folder='supp',top=20,horizontal=True):
    ok,rs=enough(df,col,ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[col]); return None
    vc=top_series(df[col],top=top); n=int(vc.sum()); grad=choose_grad(ctx); colors=grad_cols(grad,len(vc))
    fig,ax=plt.subplots(figsize=(7.2,max(3.6,0.28*len(vc)+1.2)))
    if horizontal:
        vc=vc.sort_values(); ax.barh(vc.index,vc.values,color=colors[:len(vc)],edgecolor='none')
        ax.set_xlabel('Patients, n'); ax.set_ylabel('');
        for i,v in enumerate(vc.values): ax.text(v+max(vc.values)*0.01,i,f'{int(v):,} ({v/n:.1%})',va='center',fontsize=7.5)
    else:
        ax.bar(vc.index,vc.values,color=colors[:len(vc)],edgecolor='none'); ax.tick_params(axis='x',rotation=45); ax.set_ylabel('Patients, n')
    ax.set_title(title,loc='left',fontweight='bold'); ax.grid(axis='x' if horizontal else 'y',lw=.35,alpha=.25); add_n(ax,n)
    return save_fig(ctx,fig,title,'ranked categorical distribution',[col],folder,slug(title),f'{title}. Bars show count and percentage among non-missing records.',grad,n)

def hist(ctx,df,col,title,folder='supp',bins=35,logx=False):
    ok,rs=enough(df,col,ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[col]); return None
    x=pd.to_numeric(df[col],errors='coerce').dropna();
    if x.empty: ctx.skip(title,'No numeric values',[col]); return None
    qlo,qhi=x.quantile([0.005,0.995]); xplot=x.clip(qlo,qhi)
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(6.8,4.2))
    ax.hist(xplot,bins=bins,color=GRADIENTS[grad][1],alpha=.88,edgecolor='white',linewidth=.25)
    if HAVE_SCIPY and xplot.nunique()>3:
        try:
            xs=np.linspace(xplot.min(),xplot.max(),250); kde=stats.gaussian_kde(xplot); ax2=ax.twinx(); ax2.plot(xs,kde(xs),lw=1.8,color=GRADIENTS[grad][0]); ax2.set_yticks([]); ax2.spines['right'].set_visible(False)
        except Exception: pass
    if logx: ax.set_xscale('log')
    ax.set_title(title,loc='left',fontweight='bold'); ax.set_ylabel('Frequency'); ax.set_xlabel(pretty(col)); ax.grid(axis='y',lw=.35,alpha=.25); add_n(ax,len(x))
    cap=f'{title}. Histogram uses the 0.5th–99.5th percentile display range to reduce visual compression by extreme values; raw values are retained in the cleaned dataset and flagged separately.'
    return save_fig(ctx,fig,title,'numeric distribution',[col],folder,slug(title),cap,grad,len(x))

def box_group(ctx,df,numcol,grp,title,folder='supp',top=8):
    ok,rs=enough(df,[numcol,grp],ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[numcol,grp]); return None
    d=df[[numcol,grp]].dropna().copy(); cats=d[grp].value_counts().head(top).index.tolist(); d=d[d[grp].isin(cats)]
    if d[grp].nunique()<2: ctx.skip(title,'<2 groups',[numcol,grp]); return None
    order=d.groupby(grp)[numcol].median().sort_values().index.tolist(); grad=choose_grad(ctx); colors=dict(zip(order,grad_cols(grad,len(order))))
    fig,ax=plt.subplots(figsize=(7.5,max(3.8,.35*len(order)+1.2)))
    data=[d.loc[d[grp]==g,numcol].dropna() for g in order]
    bp=ax.boxplot(data,vert=False,patch_artist=True,showfliers=False,labels=order,widths=.6)
    for patch,g in zip(bp['boxes'],order): patch.set(facecolor=colors[g],alpha=.82,edgecolor='#334155',linewidth=.7)
    for med in bp['medians']: med.set(color='white',linewidth=1.4)
    ax.set_title(title,loc='left',fontweight='bold'); ax.set_xlabel(pretty(numcol)); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,len(d))
    return save_fig(ctx,fig,title,'grouped numeric distribution',[numcol,grp],folder,slug(title),f'{title}. Box plots show median and IQR; outliers are hidden for readability, not removed from analysis data.',grad,len(d))

def stacked(ctx,df,group,hue,title,folder='supp',top=12,normalize=True):
    ok,rs=enough(df,[group,hue],ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[group,hue]); return None
    d=df[[group,hue]].dropna().copy(); groups=d[group].value_counts().head(top).index; d=d[d[group].isin(groups)]
    tab=pd.crosstab(d[group],d[hue]); tab=tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    plot=tab.div(tab.sum(axis=1),axis=0) if normalize else tab
    fig,ax=plt.subplots(figsize=(8,max(4,.35*len(plot)+1.2))); cols=CAT[:plot.shape[1]]
    left=np.zeros(len(plot))
    for i,c in enumerate(plot.columns):
        vals=plot[c].values; ax.barh(plot.index,vals,left=left,color=cols[i%len(cols)],label=str(c),height=.72); left+=vals
    ax.invert_yaxis(); ax.set_title(title,loc='left',fontweight='bold'); ax.set_xlabel('Percentage of non-missing records' if normalize else 'Patients, n'); ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=7); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,int(tab.values.sum()))
    return save_fig(ctx,fig,title,'stacked categorical composition',[group,hue],folder,slug(title),f'{title}. Bars are denominator-aware within each row category.',choose_grad(ctx),int(tab.values.sum()))

def heatmap(ctx,tab,title,variables,folder='supp',fmt='.0f',cbar_label='Patients, n'):
    if tab is None or tab.size==0 or tab.shape[0]<1 or tab.shape[1]<1: ctx.skip(title,'Empty table',variables); return None
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(max(6,.45*tab.shape[1]+2),max(4,.34*tab.shape[0]+1.5)))
    im=ax.imshow(tab.values,aspect='auto',cmap=get_cmap(grad))
    ax.set_xticks(np.arange(tab.shape[1])); ax.set_xticklabels(tab.columns,rotation=45,ha='right')
    ax.set_yticks(np.arange(tab.shape[0])); ax.set_yticklabels(tab.index)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.02); cb.set_label(cbar_label)
    mx=np.nanmax(tab.values) if np.isfinite(tab.values).any() else 0
    if tab.size<=180:
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v=tab.values[i,j]
                if pd.notna(v) and v!=0: ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=6.5,color='white' if v>mx*.55 else '#111827')
    ax.set_title(title,loc='left',fontweight='bold')
    return save_fig(ctx,fig,title,'heatmap',variables,folder,slug(title),f'{title}. Heatmap summarizes non-missing cell counts or rates as specified.',grad,int(np.nansum(tab.values)))

def crosstab_heat(ctx,df,row,col,title,folder='supp',toprow=15,topcol=12,normalize=False):
    ok,rs=enough(df,[row,col],ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[row,col]); return None
    d=df[[row,col]].dropna(); rows=d[row].value_counts().head(toprow).index; cols=d[col].value_counts().head(topcol).index; d=d[d[row].isin(rows)&d[col].isin(cols)]
    tab=pd.crosstab(d[row],d[col]); tab=tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    if normalize: tab=tab.div(tab.sum(axis=1),axis=0)*100
    return heatmap(ctx,tab,title,[row,col],folder,fmt='.1f' if normalize else '.0f',cbar_label='Row %' if normalize else 'Patients, n')

def rate_group(ctx,df,group,outcome,title,folder='supp',top=15):
    ok,rs=enough(df,[group,outcome],ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[group,outcome]); return None
    d=df[[group,outcome]].dropna(); cats=d[group].value_counts().head(top).index; d=d[d[group].isin(cats)]
    if d.empty or d[outcome].nunique()<2: ctx.skip(title,'Outcome has <2 classes or no data',[group,outcome]); return None
    g=d.groupby(group)[outcome].agg(['sum','count']).rename(columns={'sum':'events'}); g['rate']=g['events']/g['count']; g=g[g['count']>=max(10,ctx.args.min_group_n)].sort_values('rate')
    if g.empty: ctx.skip(title,'No group met denominator threshold',[group,outcome]); return None
    z=1.96; p=g['rate']; n=g['count']; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den; lo=center-half; hi=center+half
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7.2,max(3.6,.32*len(g)+1.2)))
    ax.errorbar(g['rate']*100,g.index,xerr=[(g['rate']-lo)*100,(hi-g['rate'])*100],fmt='o',color=GRADIENTS[grad][1],ecolor='#334155',elinewidth=.9,capsize=2,markersize=5)
    ax.set_xlabel('Rate, % (Wilson 95% CI)'); ax.set_title(title,loc='left',fontweight='bold'); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,int(g['count'].sum()))
    for y,(idx,r) in enumerate(g.iterrows()): ax.text(r['rate']*100+0.4,y,f"{int(r['events'])}/{int(r['count'])}",va='center',fontsize=7)
    return save_fig(ctx,fig,title,'rate with Wilson confidence intervals',[group,outcome],folder,slug(title),f'{title}. Points show rates with Wilson 95% confidence intervals; sparse groups below denominator threshold are excluded.',grad,int(g['count'].sum()))

def corr_heat(ctx,df,cols,title,folder='supp'):
    cols=[c for c in cols if c in df.columns]
    d=df[cols].apply(pd.to_numeric,errors='coerce').dropna(thresh=2)
    if len(cols)<2 or d.shape[0]<ctx.args.min_n: ctx.skip(title,'Insufficient complete numeric data',cols); return None
    corr=d.corr(method='spearman',min_periods=max(10,ctx.args.min_group_n))
    return heatmap(ctx,corr.round(2),title,cols,folder,fmt='.2f',cbar_label='Spearman ρ')

def time_trend(ctx,df,datecol,title,folder='supp',hue=None):
    if datecol not in df.columns: ctx.skip(title,'Missing date column',[datecol]); return None
    d=df[[datecol]+([hue] if hue else [])].dropna(subset=[datecol]).copy()
    if d.shape[0]<ctx.args.min_n: ctx.skip(title,'Sparse dates',[datecol]); return None
    d['month']=pd.to_datetime(d[datecol]).dt.to_period('M').dt.to_timestamp()
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(8,4.2))
    if hue and hue in d.columns:
        top=d[hue].value_counts().head(6).index
        for i,h in enumerate(top):
            s=d[d[hue]==h].groupby('month').size().sort_index(); ax.plot(s.index,s.values,lw=1.8,label=str(h),color=CAT[i%len(CAT)])
        ax.legend(frameon=False,ncol=2,fontsize=7)
    else:
        s=d.groupby('month').size().sort_index(); ax.plot(s.index,s.values,lw=2.2,color=GRADIENTS[grad][1]); ax.fill_between(s.index,s.values,color=GRADIENTS[grad][2],alpha=.25)
    ax.set_title(title,loc='left',fontweight='bold'); ax.set_ylabel('Patients, n'); ax.set_xlabel('Calendar month'); ax.grid(axis='y',lw=.35,alpha=.25); add_n(ax,len(d))
    return save_fig(ctx,fig,title,'temporal trend',[datecol]+([hue] if hue else []),folder,slug(title),f'{title}. Monthly counts based on available admission or ingestion dates.',grad,len(d))

# ----------------------------- special analyses and figures -----------------------------
def cohort_flow(ctx,df):
    n0=len(df); pii='PII excluded from public outputs'; ncat=df['poison_type_clean'].notna().sum() if 'poison_type_clean' in df else 0
    nout=df[['death_bin','any_complication_bin','absconded_bin']].notna().any(axis=1).sum() if {'death_bin','any_complication_bin','absconded_bin'}.intersection(df.columns) else 0
    fig,ax=plt.subplots(figsize=(7,5)); ax.axis('off'); grad=choose_grad(ctx); colors=grad_cols(grad,4)
    boxes=[('Workbook records loaded',n0,'All rows in All Poison Data'),('De-identified analysis set',n0,pii),('Exposure classifiable',ncat,'Mapped from poisoning/component fields'),('Outcome information available',nout,'Death/complication/absconded fields')]
    y=.84
    for i,(lab,n,sub) in enumerate(boxes):
        rect=FancyBboxPatch((.18,y-.08),.64,.13,boxstyle='round,pad=0.015,rounding_size=0.02',fc=colors[i],ec='#334155',lw=.8,alpha=.95,transform=ax.transAxes)
        ax.add_patch(rect); ax.text(.5,y,lab,ha='center',va='center',fontweight='bold',color='white' if i<2 else '#0f172a',transform=ax.transAxes); ax.text(.5,y-.038,f'n = {n:,} | {sub}',ha='center',va='center',fontsize=8,color='white' if i<2 else '#0f172a',transform=ax.transAxes)
        if i<len(boxes)-1: ax.annotate('',xy=(.5,y-.12),xytext=(.5,y-.18),arrowprops=dict(arrowstyle='-|>',lw=1,color='#334155'),xycoords=ax.transAxes)
        y-=.23
    ax.set_title('Cohort profile and analysis flow',loc='left',fontweight='bold')
    return save_fig(ctx,fig,'Cohort profile and analysis flow','STROBE-style flow diagram',['records','outcomes'],'main','figure_1a_cohort_flow','Cohort profile showing the number of loaded records, public de-identification, exposure mapping, and outcome availability.',grad,n0)

def missing_matrix(ctx,df,cols,title='Key-variable missingness matrix',folder='supp'):
    cols=[c for c in cols if c in df.columns]
    if len(cols)<2: ctx.skip(title,'Too few key columns',cols); return None
    d=df[cols].notna().astype(int); sample=d.iloc[:min(len(d),2000)]
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(max(7,.2*len(cols)+2),4.5))
    ax.imshow(sample.T,aspect='auto',interpolation='nearest',cmap=LinearSegmentedColormap.from_list('miss',['#f1f5f9',GRADIENTS[grad][1]]))
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([pretty(c) for c in cols]); ax.set_xticks([]); ax.set_xlabel(f'Patient records shown: first {len(sample):,} rows'); ax.set_title(title,loc='left',fontweight='bold')
    return save_fig(ctx,fig,title,'missingness matrix',cols,folder,slug(title),'Presence/absence matrix for key variables; filled cells indicate non-missing values.',grad,len(df))

def completeness(ctx,df,cols,title,folder='supp'):
    cols=[c for c in cols if c in df.columns]
    if not cols: ctx.skip(title,'No requested columns present',cols); return None
    comp=df[cols].notna().mean().sort_values(); grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7.2,max(4,.24*len(comp)+1.2)))
    ax.hlines(comp.index,0,comp.values*100,color='#CBD5E1',lw=2); ax.scatter(comp.values*100,comp.index,s=28,color=GRADIENTS[grad][1])
    ax.set_xlim(0,100); ax.set_xlabel('Completeness, %'); ax.set_title(title,loc='left',fontweight='bold'); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,len(df))
    return save_fig(ctx,fig,title,'variable completeness lollipop',cols,folder,slug(title),'Completeness of selected variables among all de-identified records.',grad,len(df))

def dq_map(ctx,dq):
    if dq is None or dq.empty: ctx.skip('Data-quality issue map','No data-quality issues recorded'); return None
    tab=dq.groupby(['issue','column'])['n'].sum().unstack(fill_value=0)
    if tab.shape[1]>30: tab=tab.loc[:,tab.sum().sort_values(ascending=False).head(30).index]
    return heatmap(ctx,tab,'Data-quality issue map',['data_quality_report'],'supp',fmt='.0f',cbar_label='Flagged cells/variables')

def age_pyramid(ctx,df):
    if not {'age_group','sex_clean'}.issubset(df.columns): ctx.skip('Age group pyramid by sex','Missing age group or sex'); return None
    d=df[['age_group','sex_clean']].dropna(); d=d[d['sex_clean'].isin(['Male','Female'])]
    if len(d)<ctx.args.min_n: ctx.skip('Age group pyramid by sex','Sparse age/sex data'); return None
    tab=pd.crosstab(d['age_group'],d['sex_clean']).reindex(AGE_LABELS).fillna(0)
    fig,ax=plt.subplots(figsize=(6.8,4.5)); grad=choose_grad(ctx)
    male=-tab.get('Male',pd.Series(0,index=tab.index)); female=tab.get('Female',pd.Series(0,index=tab.index))
    ax.barh(tab.index,male,color=GRADIENTS[grad][1],label='Male'); ax.barh(tab.index,female,color=GRADIENTS[grad][2],label='Female')
    mx=max(abs(male).max(),female.max()); ax.set_xlim(-mx*1.25,mx*1.25); ax.set_xticklabels([str(abs(int(t))) for t in ax.get_xticks()])
    ax.set_xlabel('Patients, n'); ax.set_title('Age group pyramid by sex',loc='left',fontweight='bold'); ax.legend(frameon=False); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,len(d))
    return save_fig(ctx,fig,'Age group pyramid by sex','age-sex population pyramid',['age_group','sex_clean'],'main','figure_1c_age_pyramid','Age distribution shown as a sex-stratified pyramid.',grad,len(d))

def symptom_prevalence(ctx,df):
    cols=[slug(s)+'_bin' for s in SYMPTOMS if slug(s)+'_bin' in df.columns]
    if not cols: ctx.skip('Symptom prevalence ranked bar chart','No symptom binary variables'); return None
    rates=[]
    for c in cols:
        den=df[c].notna().sum(); ev=df[c].sum(skipna=True); rates.append((pretty(c),ev,den,ev/den if den else np.nan))
    res=pd.DataFrame(rates,columns=['symptom','events','den','rate']).dropna().sort_values('rate')
    if res.empty: ctx.skip('Symptom prevalence ranked bar chart','No denominators'); return None
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7,max(4,.28*len(res)+1)))
    ax.barh(res['symptom'],res['rate']*100,color=grad_cols(grad,len(res)),edgecolor='none')
    for y,r in enumerate(res.itertuples()): ax.text(r.rate*100+.4,y,f'{int(r.events)}/{int(r.den)}',va='center',fontsize=7)
    ax.set_xlabel('Prevalence, % of non-missing'); ax.set_title('Symptom prevalence ranked bar chart',loc='left',fontweight='bold'); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,int(res['den'].max()))
    return save_fig(ctx,fig,'Symptom prevalence ranked bar chart','symptom prevalence',cols,'main','figure_3a_symptom_prevalence','Ranked symptom prevalence with denominator-aware event counts.',grad,int(res['den'].max()))

def binary_heat(ctx,df,row,bins,title,folder='supp',top=10):
    bins=[b for b in bins if b in df.columns]
    if row not in df.columns or not bins: ctx.skip(title,'Missing grouping/binary variables',[row]+bins); return None
    d=df[[row]+bins].dropna(subset=[row]); groups=d[row].value_counts().head(top).index; d=d[d[row].isin(groups)]
    if len(d)<ctx.args.min_n: ctx.skip(title,'Sparse grouped binary data',[row]+bins); return None
    mat=[]
    for g in groups:
        sub=d[d[row]==g]; mat.append([sub[b].mean(skipna=True)*100 for b in bins])
    tab=pd.DataFrame(mat,index=groups,columns=[pretty(b) for b in bins])
    return heatmap(ctx,tab,title,[row]+bins,folder,fmt='.1f',cbar_label='Prevalence, %')

def symptom_combos(ctx,df):
    cols=[slug(s)+'_bin' for s in SYMPTOMS if slug(s)+'_bin' in df.columns]
    cols=cols[:10]
    if len(cols)<3: ctx.skip('UpSet plot for symptom combinations','Too few symptom variables',cols); return None
    d=df[cols].fillna(0).astype(bool)
    if d.sum().sum()<ctx.args.min_n: ctx.skip('UpSet plot for symptom combinations','Sparse symptom positives',cols); return None
    fig=plt.figure(figsize=(8,5))
    try:
        if HAVE_UPSETPLOT:
            data=from_indicators(cols,d); UpSet(data,subset_size='count',show_counts=True,sort_by='cardinality').plot(fig=fig); title='UpSet plot for symptom combinations'
        else:
            # fallback: top combinations bar
            comb=d.astype(int).astype(str).agg(''.join,axis=1).value_counts().head(15).sort_values(); ax=fig.add_subplot(111); ax.barh(range(len(comb)),comb.values,color=GRADIENTS[choose_grad(ctx)][1]); ax.set_yticks(range(len(comb))); ax.set_yticklabels(comb.index); ax.set_xlabel('Patients, n'); title='Top symptom combinations fallback plot'; ax.set_title(title,loc='left',fontweight='bold')
    except Exception as e:
        plt.close(fig); ctx.skip('UpSet plot for symptom combinations',str(e),cols); return None
    return save_fig(ctx,fig,'UpSet plot for symptom combinations','symptom combination/upset',cols,'supp','symptom_combinations_upset','Symptom co-occurrence combinations among the most common symptom fields.',choose_grad(ctx),len(d))

def symptom_network(ctx,df):
    cols=[slug(s)+'_bin' for s in SYMPTOMS if slug(s)+'_bin' in df.columns]
    if len(cols)<3 or not HAVE_NETWORKX: ctx.skip('Network plot of symptom co-occurrence','networkx unavailable or too few symptoms',cols); return None
    d=df[cols].fillna(0).astype(int); G=nx.Graph()
    prev=d.mean();
    for c,p in prev.items():
        if p>0.01: G.add_node(pretty(c),size=p)
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            co=((d[a]==1)&(d[b]==1)).mean()
            if co>0.01 and pretty(a) in G and pretty(b) in G: G.add_edge(pretty(a),pretty(b),weight=co)
    if G.number_of_nodes()<3 or G.number_of_edges()<1: ctx.skip('Network plot of symptom co-occurrence','Sparse co-occurrence network',cols); return None
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7,5.5)); pos=nx.spring_layout(G,seed=RANDOM_SEED,k=.8)
    sizes=[1000*G.nodes[n]['size']+80 for n in G.nodes]
    widths=[G.edges[e]['weight']*18 for e in G.edges]
    nx.draw_networkx_edges(G,pos,ax=ax,width=widths,alpha=.35,edge_color='#64748B'); nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color=GRADIENTS[grad][1],alpha=.9,edgecolors='white',linewidths=.6); nx.draw_networkx_labels(G,pos,ax=ax,font_size=7)
    ax.axis('off'); ax.set_title('Network plot of symptom co-occurrence',loc='left',fontweight='bold')
    return save_fig(ctx,fig,'Network plot of symptom co-occurrence','co-occurrence network',cols,'supp','symptom_cooccurrence_network','Network edges represent symptom pairs co-occurring in at least 1% of records with available symptom coding.',grad,len(d))

def static_treemap(ctx,df,col,title,folder='supp',top=18):
    ok,rs=enough(df,col,ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[col]); return None
    vc=top_series(df[col],top=top); grad=choose_grad(ctx); colors=grad_cols(grad,len(vc))
    # simple slice-and-dice fallback treemap
    fig,ax=plt.subplots(figsize=(8,5)); ax.axis('off'); total=vc.sum(); x=y=0; w=h=1; horiz=True
    items=list(vc.items())
    for i,(lab,val) in enumerate(items):
        frac=val/total
        if horiz:
            ww=w*frac/(sum(v for _,v in items[i:])/total) if sum(v for _,v in items[i:]) else 0
            rect=(x,y,ww,h); x+=ww; w-=ww
        else:
            hh=h*frac/(sum(v for _,v in items[i:])/total) if sum(v for _,v in items[i:]) else 0
            rect=(x,y,w,hh); y+=hh; h-=hh
        horiz=not horiz
        ax.add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],transform=ax.transAxes,fc=colors[i],ec='white',lw=1.2))
        if rect[2]*rect[3]>.035: ax.text(rect[0]+rect[2]/2,rect[1]+rect[3]/2,f'{lab}\n{int(val):,}',ha='center',va='center',fontsize=7,color='white' if i<5 else '#0f172a',transform=ax.transAxes)
    ax.set_title(title,loc='left',fontweight='bold')
    return save_fig(ctx,fig,title,'treemap/slice-dice categorical share',[col],folder,slug(title),f'{title}. Rectangle area is proportional to the category count.',grad,int(total))

def bubble(ctx,df,x,y,outcome,title,folder='supp',topx=10,topy=10):
    ok,rs=enough(df,[x,y,outcome],ctx.args.min_n)
    if not ok: ctx.skip(title,rs,[x,y,outcome]); return None
    d=df[[x,y,outcome]].dropna(); xs=d[x].value_counts().head(topx).index; ys=d[y].value_counts().head(topy).index; d=d[d[x].isin(xs)&d[y].isin(ys)]
    if d.empty: ctx.skip(title,'No data after top category filtering',[x,y,outcome]); return None
    g=d.groupby([x,y]).agg(n=(outcome,'size'),rate=(outcome,'mean')).reset_index(); grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(8,5.5))
    xmap={v:i for i,v in enumerate(xs)}; ymap={v:i for i,v in enumerate(ys)}
    sc=ax.scatter(g[x].map(xmap),g[y].map(ymap),s=np.sqrt(g['n'])*28,c=g['rate']*100,cmap=get_cmap(grad),edgecolor='white',linewidth=.6,alpha=.9)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs,rotation=45,ha='right'); ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys); ax.set_title(title,loc='left',fontweight='bold'); ax.set_xlabel(pretty(x)); ax.set_ylabel(pretty(y)); cb=fig.colorbar(sc,ax=ax,fraction=.035,pad=.02); cb.set_label(f'{pretty(outcome)} rate, %'); add_n(ax,int(g['n'].sum()))
    return save_fig(ctx,fig,title,'bubble plot frequency and rate',[x,y,outcome],folder,slug(title),'Bubble size indicates patient count and color indicates endpoint rate.',grad,int(g['n'].sum()))

def pair_matrix(ctx,df,cols,title='Selected vitals scatter matrix',folder='supp'):
    cols=[c for c in cols if c in df.columns]
    d=df[cols].apply(pd.to_numeric,errors='coerce').dropna()
    if len(cols)<2 or len(d)<ctx.args.min_n: ctx.skip(title,'Insufficient numeric complete cases',cols); return None
    d=d.sample(min(len(d),1200),random_state=RANDOM_SEED)
    k=len(cols); fig,axs=plt.subplots(k,k,figsize=(2.1*k,2.1*k)); grad=choose_grad(ctx)
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            ax=axs[i,j]
            if i==j: ax.hist(d[a],bins=25,color=GRADIENTS[grad][1],alpha=.85)
            else: ax.scatter(d[b],d[a],s=5,alpha=.25,color=GRADIENTS[grad][1],edgecolor='none')
            if i==k-1: ax.set_xlabel(pretty(b),fontsize=7)
            else: ax.set_xticklabels([])
            if j==0: ax.set_ylabel(pretty(a),fontsize=7)
            else: ax.set_yticklabels([])
            ax.grid(False)
    fig.suptitle(title,x=.02,ha='left',fontweight='bold')
    return save_fig(ctx,fig,title,'scatter matrix',cols,folder,slug(title),'Exploratory scatter-matrix among selected cleaned numeric vital-sign variables.',grad,len(d))

def lab_abnormality(ctx,df):
    thresholds={'creatinine_mg_dl_analysis':('Creatinine >1.5 mg/dL',lambda s:s>1.5),'sodium_mmol_l_analysis':('Na <135 or >145',lambda s:(s<135)|(s>145)),'potassium_mmol_l_analysis':('K <3.5 or >5.5',lambda s:(s<3.5)|(s>5.5)),'ph_analysis':('pH <7.35 or >7.45',lambda s:(s<7.35)|(s>7.45)),'anion_gap_analysis':('Anion gap >16',lambda s:s>16),'hemoglobin_g_dl_analysis':('Hemoglobin <10 g/dL',lambda s:s<10),'platelets_mm3_analysis':('Platelets <150k/mm³',lambda s:s<150000),'random_glucose_mmol_l_analysis':('Glucose >11.1 mmol/L',lambda s:s>11.1),'sgpt_u_l_analysis':('SGPT >40 U/L',lambda s:s>40),'sgot_u_l_analysis':('SGOT >40 U/L',lambda s:s>40)}
    rows=[]
    for c,(lab,fn) in thresholds.items():
        if c in df.columns:
            s=df[c].dropna(); den=len(s)
            if den>=ctx.args.min_group_n:
                ev=int(fn(s).sum()); rows.append((lab,ev,den,ev/den))
    if not rows: ctx.skip('Lab abnormality prevalence plot','No lab variable met denominator threshold'); return None
    res=pd.DataFrame(rows,columns=['label','events','den','rate']).sort_values('rate'); grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7,max(4,.3*len(res)+1)))
    ax.barh(res['label'],res['rate']*100,color=grad_cols(grad,len(res)),edgecolor='none')
    for y,r in enumerate(res.itertuples()): ax.text(r.rate*100+.4,y,f'{r.events}/{r.den}',va='center',fontsize=7)
    ax.set_xlabel('Abnormality prevalence, %'); ax.set_title('Lab abnormality prevalence plot',loc='left',fontweight='bold'); ax.grid(axis='x',lw=.35,alpha=.25)
    return save_fig(ctx,fig,'Lab abnormality prevalence plot','laboratory abnormality prevalence',list(thresholds),'supp','lab_abnormality_prevalence','Laboratory abnormality prevalence based on pragmatic clinical thresholds; thresholds should be reviewed before manuscript use.',grad,int(max(r[2] for r in rows)))

def pca_labs(ctx,df):
    labs=[c for c in [v+'_analysis' for v in LAB_MAP.values()] if c in df.columns]
    if not HAVE_SKLEARN or len(labs)<4: ctx.skip('PCA exploratory lab profile','sklearn unavailable or too few labs',labs); return None
    d=df[labs].apply(pd.to_numeric,errors='coerce'); keep=d.notna().mean().sort_values(ascending=False).head(10).index.tolist(); d=d[keep]
    if d.dropna().shape[0]<ctx.args.min_n: ctx.skip('PCA exploratory lab profile','Too few complete lab cases',keep); return None
    X=StandardScaler().fit_transform(d.dropna()); pca=PCA(n_components=2,random_state=RANDOM_SEED); Z=pca.fit_transform(X)
    grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(5.5,5)); ax.scatter(Z[:,0],Z[:,1],s=12,alpha=.55,color=GRADIENTS[grad][1],edgecolor='none'); ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'); ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'); ax.set_title('PCA exploratory lab profile',loc='left',fontweight='bold'); ax.grid(lw=.35,alpha=.25); add_n(ax,len(Z))
    return save_fig(ctx,fig,'PCA exploratory lab profile','exploratory PCA',keep,'supp','pca_lab_profile','Exploratory PCA of the most complete laboratory variables; not intended as a validated clinical classifier.',grad,len(Z),notes='Exploratory')

def logistic_or_table(ctx,df,outcome,predictors,label):
    if outcome not in df.columns: return pd.DataFrame()
    rows=[]
    for p in predictors:
        if p not in df.columns: continue
        d=df[[outcome,p]].dropna()
        if d.shape[0]<ctx.args.min_n or d[outcome].nunique()<2: continue
        try:
            if pd.api.types.is_numeric_dtype(d[p]) and d[p].nunique()>8:
                x=((d[p]-d[p].mean())/(d[p].std(ddof=0) or 1)).rename(p); X=sm.add_constant(x) if HAVE_STATSMODELS else None
                if HAVE_STATSMODELS:
                    model=sm.Logit(d[outcome].astype(float),X).fit(disp=False,maxiter=100)
                    coef=model.params[p]; se=model.bse[p]; rows.append({'endpoint':outcome,'predictor':pretty(p)+' (per SD)','level':'','OR':math.exp(coef),'CI_low':math.exp(coef-1.96*se),'CI_high':math.exp(coef+1.96*se),'p_value':model.pvalues[p],'n':len(d),'events':int(d[outcome].sum())})
            else:
                cats=d[p].astype(str).value_counts().head(8).index.tolist(); ref=cats[0]
                for cat in cats[1:]:
                    sub=d[d[p].astype(str).isin([ref,cat])].copy(); sub['x']=(sub[p].astype(str)==cat).astype(int)
                    if sub['x'].nunique()<2 or sub[outcome].nunique()<2: continue
                    if HAVE_STATSMODELS:
                        model=sm.Logit(sub[outcome].astype(float),sm.add_constant(sub['x'])).fit(disp=False,maxiter=100)
                        coef=model.params['x']; se=model.bse['x']; rows.append({'endpoint':outcome,'predictor':pretty(p),'level':f'{cat} vs {ref}','OR':math.exp(coef),'CI_low':math.exp(coef-1.96*se),'CI_high':math.exp(coef+1.96*se),'p_value':model.pvalues['x'],'n':len(sub),'events':int(sub[outcome].sum())})
        except Exception as e:
            ctx.skip(f'Univariable OR for {p}',str(e),[outcome,p])
    return pd.DataFrame(rows)

def forest_plot(ctx,tab,title,folder='supp',top=25):
    if tab is None or tab.empty: ctx.skip(title,'No model rows'); return None
    d=tab.replace([np.inf,-np.inf],np.nan).dropna(subset=['OR','CI_low','CI_high']).copy().head(top)
    if d.empty: ctx.skip(title,'No finite OR rows'); return None
    d['label']=d['predictor'].astype(str)+np.where(d['level'].astype(str).str.len()>0,' — '+d['level'].astype(str),'')
    d=d.iloc[::-1]; grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7.4,max(4,.32*len(d)+1.2)))
    y=np.arange(len(d)); ax.errorbar(d['OR'],y,xerr=[d['OR']-d['CI_low'],d['CI_high']-d['OR']],fmt='o',color=GRADIENTS[grad][1],ecolor='#334155',elinewidth=.9,capsize=2)
    ax.axvline(1,color='#64748B',ls='--',lw=.9); ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels(d['label']); ax.set_xlabel('Odds ratio (log scale), 95% CI'); ax.set_title(title,loc='left',fontweight='bold'); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,int(d['n'].max()))
    return save_fig(ctx,fig,title,'univariable logistic forest plot',list(d['predictor'].unique()),folder,slug(title),f'{title}. Univariable logistic regression estimates; observational associations are not causal effects.',grad,int(d['n'].max()),notes='Exploratory/associational')

def multivariable_logistic(ctx,df,outcome,preds,title):
    if not HAVE_STATSMODELS or outcome not in df.columns: ctx.skip(title,'statsmodels unavailable or endpoint missing'); return pd.DataFrame()
    use=[]; tmp=df[[outcome]+[p for p in preds if p in df.columns]].copy().dropna(subset=[outcome])
    if tmp[outcome].nunique()<2 or tmp[outcome].sum()<max(10,ctx.args.min_events): ctx.skip(title,'Insufficient outcome events'); return pd.DataFrame()
    for p in preds:
        if p in tmp.columns and pd.api.types.is_numeric_dtype(tmp[p]) and tmp[p].notna().sum()>=ctx.args.min_n: use.append(p)
    if len(use)<2: ctx.skip(title,'Too few numeric predictors for adjusted model',use); return pd.DataFrame()
    d=tmp[[outcome]+use].dropna(); max_pred=max(1,int(d[outcome].sum()//10)); use=use[:max_pred]
    if d.shape[0]<ctx.args.min_n or len(use)<1: ctx.skip(title,'Insufficient complete cases after predictor selection',use); return pd.DataFrame()
    try:
        X=d[use].astype(float); X=(X-X.mean())/X.std(ddof=0).replace(0,1); X=sm.add_constant(X)
        model=sm.Logit(d[outcome].astype(float),X).fit(disp=False,maxiter=200)
        rows=[]
        for p in use:
            coef=model.params[p]; se=model.bse[p]; rows.append({'endpoint':outcome,'predictor':pretty(p)+' (adjusted per SD)','level':'','OR':math.exp(coef),'CI_low':math.exp(coef-1.96*se),'CI_high':math.exp(coef+1.96*se),'p_value':model.pvalues[p],'n':len(d),'events':int(d[outcome].sum())})
        return pd.DataFrame(rows)
    except Exception as e:
        ctx.skip(title,str(e),use); return pd.DataFrame()

def rf_importance(ctx,df,outcome,preds):
    if not HAVE_SKLEARN or outcome not in df.columns: ctx.skip('Exploratory random forest predictor importance','sklearn unavailable or outcome missing'); return None
    use=[p for p in preds if p in df.columns]
    d=df[[outcome]+use].dropna(subset=[outcome]).copy()
    if d[outcome].nunique()<2 or d[outcome].sum()<ctx.args.min_events or len(use)<3: ctx.skip('Exploratory random forest predictor importance','Insufficient endpoint/classes/predictors'); return None
    # numeric + one-hot categoricals
    X=pd.get_dummies(d[use],dummy_na=True).replace([np.inf,-np.inf],np.nan)
    X=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X),columns=X.columns) if len(X.columns) else pd.DataFrame()
    if X.shape[1]<3: ctx.skip('Exploratory random forest predictor importance','Too few encoded predictors'); return None
    y=d[outcome].astype(int).values
    try:
        rf=RandomForestClassifier(n_estimators=250,random_state=RANDOM_SEED,class_weight='balanced_subsample',min_samples_leaf=10,n_jobs=-1)
        rf.fit(X,y); imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values().tail(20)
        grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7,max(4,.3*len(imp)+1)))
        ax.barh(imp.index,imp.values,color=grad_cols(grad,len(imp)),edgecolor='none'); ax.set_xlabel('Mean decrease impurity importance'); ax.set_title('Exploratory random forest predictor importance',loc='left',fontweight='bold'); ax.grid(axis='x',lw=.35,alpha=.25); add_n(ax,len(d))
        return save_fig(ctx,fig,'Exploratory random forest predictor importance','machine-learning exploratory importance',use,'supp','exploratory_rf_importance','Exploratory random-forest importance; not a validated clinical prediction model and should not be interpreted causally.',grad,len(d),notes='Exploratory')
    except Exception as e: ctx.skip('Exploratory random forest predictor importance',str(e),use); return None

def roc_calibration(ctx,df,outcome,preds):
    if not HAVE_SKLEARN or outcome not in df.columns: ctx.skip('Internal ROC and calibration plot','sklearn unavailable or endpoint missing'); return None
    use=[p for p in preds if p in df.columns and pd.api.types.is_numeric_dtype(df[p])]
    d=df[[outcome]+use].dropna().copy()
    if len(use)<2 or d.shape[0]<100 or d[outcome].sum()<ctx.args.min_events or d[outcome].nunique()<2: ctx.skip('Internal ROC and calibration plot','Predictive modeling not appropriate/too sparse',use); return None
    try:
        X=StandardScaler().fit_transform(d[use]); y=d[outcome].astype(int).values
        rf=RandomForestClassifier(n_estimators=200,random_state=RANDOM_SEED,class_weight='balanced',min_samples_leaf=10)
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_SEED); prob=cross_val_predict(rf,X,y,cv=cv,method='predict_proba')[:,1]
        fpr,tpr,_=roc_curve(y,prob); au=auc(fpr,tpr); bins=pd.qcut(prob,q=8,duplicates='drop'); cal=pd.DataFrame({'prob':prob,'y':y,'bin':bins}).groupby('bin',observed=False).agg(pred=('prob','mean'),obs=('y','mean'),n=('y','size'))
        grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(5.5,5)); ax.plot(fpr,tpr,lw=2,color=GRADIENTS[grad][1],label=f'AUC={au:.2f}'); ax.plot([0,1],[0,1],'--',color='#94A3B8',lw=1); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.set_title('Internally cross-validated ROC curve',loc='left',fontweight='bold'); ax.legend(frameon=False); ax.grid(lw=.35,alpha=.25)
        save_fig(ctx,fig,'Internally cross-validated ROC curve','exploratory ROC',use,'supp','exploratory_roc_curve','Exploratory internally cross-validated ROC curve; model performance requires external validation.',grad,len(d),notes='Exploratory')
        fig,ax=plt.subplots(figsize=(5.5,5)); ax.scatter(cal['pred'],cal['obs'],s=cal['n']*3,color=GRADIENTS[grad][1],alpha=.8,edgecolor='white'); ax.plot([0,1],[0,1],'--',color='#94A3B8',lw=1); ax.set_xlabel('Mean predicted risk'); ax.set_ylabel('Observed risk'); ax.set_title('Exploratory calibration plot',loc='left',fontweight='bold'); ax.grid(lw=.35,alpha=.25)
        return save_fig(ctx,fig,'Exploratory calibration plot','exploratory calibration',use,'supp','exploratory_calibration_plot','Exploratory calibration plot using internal cross-validation; not for clinical deployment.',grad,len(d),notes='Exploratory')
    except Exception as e: ctx.skip('Internal ROC and calibration plot',str(e),use); return None

def medication_frequency(ctx,df):
    med_cols=[c for c in df.columns if c.split('__dup')[0]=='Name']
    if not med_cols: ctx.skip('Medication-name frequency plots','No repeated Name medication fields'); return None
    vals=pd.concat([df[c] for c in med_cols],ignore_index=True).dropna().map(title_safe)
    vals=vals[~vals.astype(str).str.lower().isin(MISSING_TOKENS)]
    tmp=pd.DataFrame({'medication_name':vals})
    return bar(ctx,tmp,'medication_name','Medication-name frequency from repeated Name fields','supp',top=25)

def dose_duration_summary(ctx,df):
    cols=[c for c in df.columns if c.split('__dup')[0] in {'Dose','Duration'} or c=='Dose (amp)']
    if not cols: ctx.skip('Dose and duration cleaning summaries','No dose/duration fields'); return None
    rows=[]
    for c in cols:
        s=df[c].map(num) if c in df.columns else pd.Series(dtype=float)
        rows.append({'field':c,'non_missing_raw':int(df[c].notna().sum()),'numeric_parseable':int(s.notna().sum()),'median_numeric':float(s.median()) if s.notna().any() else np.nan})
    return pd.DataFrame(rows)

def interactive_outputs(ctx,df):
    if not HAVE_PLOTLY: ctx.skip('Interactive HTML outputs','plotly unavailable'); return
    try:
        if 'poison_type_clean' in df.columns:
            vc=top_series(df['poison_type_clean'],top=20); fig=px.bar(vc.sort_values(),orientation='h',title='Interactive ranked poisoning type distribution',labels={'value':'Patients','index':'Poisoning type'}); fig.write_html(ctx.dir('html')/'interactive_poisoning_type_distribution.html',include_plotlyjs='cdn')
        if {'study_site_clean','poison_type_clean','death_bin'}.issubset(df.columns):
            d=df[['study_site_clean','poison_type_clean','death_bin']].dropna(); d=d[d['poison_type_clean'].isin(d['poison_type_clean'].value_counts().head(10).index)]
            if len(d)>=ctx.args.min_n:
                g=d.groupby(['study_site_clean','poison_type_clean']).agg(n=('death_bin','size'),death_rate=('death_bin','mean')).reset_index(); fig=px.scatter(g,x='study_site_clean',y='poison_type_clean',size='n',color='death_rate',title='Interactive site × poisoning type outcome bubble plot'); fig.write_html(ctx.dir('html')/'interactive_site_poisoning_outcome_bubble.html',include_plotlyjs='cdn')
        if {'study_site_clean','poison_type_clean','death_bin'}.issubset(df.columns):
            d=df[['study_site_clean','poison_type_clean','death_bin']].dropna().copy(); d['outcome']=np.where(d['death_bin']==1,'Death','Non-death')
            d=d[d['poison_type_clean'].isin(d['poison_type_clean'].value_counts().head(8).index)]
            if len(d)>=ctx.args.min_n:
                nodes=list(pd.Index(d['study_site_clean']).unique())+list(pd.Index(d['poison_type_clean']).unique())+['Death','Non-death']; node_index={n:i for i,n in enumerate(nodes)}
                links=[]
                for a,b in [('study_site_clean','poison_type_clean'),('poison_type_clean','outcome')]:
                    g=d.groupby([a,b]).size().reset_index(name='n')
                    for _,r in g.iterrows(): links.append({'source':node_index[r[a]],'target':node_index[r[b]],'value':int(r['n'])})
                fig=go.Figure(data=[go.Sankey(node=dict(label=nodes,pad=10,thickness=12),link=dict(source=[l['source'] for l in links],target=[l['target'] for l in links],value=[l['value'] for l in links]))]); fig.update_layout(title_text='Interactive Sankey: site → poisoning type → outcome'); fig.write_html(ctx.dir('html')/'interactive_site_poisoning_outcome_sankey.html',include_plotlyjs='cdn')
    except Exception as e: ctx.skip('Interactive HTML outputs',str(e))

# ----------------------------- tables/reports -----------------------------
def create_tables(ctx,df,dq,dd):
    tdir=ctx.dir('tables')
    pd.DataFrame({'metric':['rows','columns','figures_generated_so_far'],'value':[len(df),df.shape[1],len(ctx.registry)]}).to_csv(tdir/'dataset_overview.csv',index=False)
    miss=pd.DataFrame({'column':df.columns,'missing_n':df.isna().sum().values,'missing_percent':df.isna().mean().values*100,'non_missing_n':df.notna().sum().values}).sort_values('missing_percent',ascending=False)
    miss.to_csv(tdir/'missingness_by_variable.csv',index=False)
    if 'poison_type_clean' in df.columns: df['poison_type_clean'].value_counts(dropna=False).rename_axis('poison_type').reset_index(name='n').to_csv(tdir/'poisoning_type_counts.csv',index=False)
    if 'study_site_clean' in df.columns: df['study_site_clean'].value_counts(dropna=False).rename_axis('study_site').reset_index(name='n').to_csv(tdir/'study_site_counts.csv',index=False)
    if {'poison_type_clean','death_bin'}.issubset(df.columns):
        df.groupby('poison_type_clean')['death_bin'].agg(['sum','count','mean']).rename(columns={'sum':'deaths','mean':'death_rate'}).sort_values('count',ascending=False).to_csv(tdir/'death_rate_by_poisoning_type.csv')
    if {'poison_type_clean','any_complication_bin'}.issubset(df.columns):
        df.groupby('poison_type_clean')['any_complication_bin'].agg(['sum','count','mean']).rename(columns={'sum':'complications','mean':'complication_rate'}).sort_values('count',ascending=False).to_csv(tdir/'complication_rate_by_poisoning_type.csv')
    if dq is not None: dq.to_csv(ctx.dir('logs')/'data_quality_report.csv',index=False)
    dd.to_csv(ctx.dir('clean')/'data_dictionary.csv',index=False)
    pd.DataFrame(ctx.decisions).to_csv(ctx.dir('clean')/'cleaning_decisions.csv',index=False)
    return miss

def write_clean(ctx,df):
    df.to_csv(ctx.dir('clean')/'cleaned_deidentified_analysis_dataset.csv',index=False)
    try: df.to_parquet(ctx.dir('clean')/'cleaned_deidentified_analysis_dataset.parquet',index=False)
    except Exception as e: ctx.warn(f'Parquet export skipped: {e}')

def write_registry_manifest(ctx,input_hash,script_hash):
    reg=pd.DataFrame(ctx.registry); reg.to_csv(ctx.dir('ver')/'figure_registry.csv',index=False); write_json(ctx.dir('ver')/'figure_registry.json',ctx.registry)
    manifest={'run_id':ctx.run_id,'timestamp_started':ctx.started,'timestamp_finished':dt.datetime.now().isoformat(timespec='seconds'),'input_path':str(ctx.input),'input_sha256':input_hash,'script_sha256':script_hash,'duplicate_headers':ctx.dups,'n_figures':len(ctx.registry),'n_skipped':len(ctx.skipped),'skipped_analyses':ctx.skipped,'warnings':ctx.warnings,'package_versions':package_versions()}
    write_json(ctx.dir('ver')/'run_manifest.json',manifest)
    cfg={'random_seed':RANDOM_SEED,'min_n':ctx.args.min_n,'min_group_n':ctx.args.min_group_n,'min_events':ctx.args.min_events,'pii_excluded':sorted(PII),'plausible_ranges':PLAUSIBLE,'age_bins':AGE_LABELS,'time_bins':TIME_LABELS,'gradients':GRADIENTS,'figure_format':['png 600dpi','pdf','svg']}
    write_yamlish(ctx.dir('ver')/'analysis_config.yaml',cfg)
    Path(ctx.dir('ver')/'input_sha256.txt').write_text(input_hash,encoding='utf-8'); Path(ctx.dir('ver')/'script_sha256.txt').write_text(script_hash,encoding='utf-8')

def gallery(ctx):
    rows=[]
    for r in ctx.registry:
        rel=os.path.relpath(r['png'],ctx.out); cap=html.escape(r['caption'][:420])
        rows.append(f"<div class='card'><a href='../{html.escape(rel)}'><img src='../{html.escape(rel)}'></a><h3>{html.escape(r['figure_id'])}: {html.escape(r['title'])}</h3><p>{cap}</p><small>{html.escape(r['analysis_type'])} | Gradient: {html.escape(r['gradient'])}</small></div>")
    body=f"""<!doctype html><html><head><meta charset='utf-8'><title>Poison figures gallery</title><style>body{{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px}}.card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:12px;box-shadow:0 1px 4px #0001}}img{{width:100%;height:210px;object-fit:contain;background:white}}h1{{margin-bottom:0}}h3{{font-size:14px}}p,small{{font-size:12px;color:#475569}}</style></head><body><h1>Poison publication figure gallery</h1><p>Run: {ctx.run_id}. Generated figures: {len(ctx.registry)}.</p><div class='grid'>{''.join(rows)}</div></body></html>"""
    (ctx.out/'figure_index_gallery.html').write_text(body,encoding='utf-8')

def summary_report(ctx,df,dq):
    topdq=dq.sort_values('n',ascending=False).head(10).to_dict('records') if dq is not None and not dq.empty else []
    md=['# Poison figure-generation summary','',f'- Run ID: `{ctx.run_id}`',f'- Rows loaded into public analysis dataset: {len(df):,}',f'- Columns in public analysis dataset: {df.shape[1]:,}',f'- Duplicate headers found: `{ctx.dups}`',f'- PII excluded: {", ".join(sorted(PII))}',f'- Figures generated: {len(ctx.registry):,}',f'- Analyses skipped gracefully: {len(ctx.skipped):,}','','## Top data-quality limitations']
    for r in topdq: md.append(f"- {r.get('issue')} — {r.get('column')}: {r.get('n')} ({r.get('detail','')})")
    md+=['','## Scientific caution','All figures are descriptive or exploratory unless explicitly labelled as regression/modeling. Observational associations should not be interpreted as causal effects.']
    (ctx.out/'RUN_SUMMARY.md').write_text('\n'.join(md),encoding='utf-8')

def try_git(ctx):
    if ctx.args.no_git: return
    try:
        subprocess.run(['git','--version'],cwd=ctx.out,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','init'],cwd=ctx.out,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','add','07_version_control','00_logs','RUN_SUMMARY.md','figure_index_gallery.html'],cwd=ctx.out,check=False)
        subprocess.run(['git','commit','-m',f'Poison figure run {ctx.run_id}'],cwd=ctx.out,check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','tag',ctx.run_id],cwd=ctx.out,check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    except Exception as e: ctx.warn(f'Git initialization skipped: {e}')

# ----------------------------- manuscript panels -----------------------------
def assemble_panel(ctx,keyword_re,title,basename,folder='panels',max_items=4):
    """Fast text-index multi-panel candidate, with no raster embedding."""
    regs=[r for r in ctx.registry if re.search(keyword_re,r['basename']+' '+r['title'],re.I)][:max_items]
    if len(regs)<2: ctx.skip(title,'Not enough component figures available'); return None
    try:
        from PIL import Image, ImageDraw, ImageFont
        cols=2; rows=math.ceil(len(regs)/cols); tile_w,tile_h=950,360; margin=60; header=80
        W=cols*tile_w+2*margin; H=rows*tile_h+header+2*margin
        im=Image.new('RGB',(W,H),'white'); draw=ImageDraw.Draw(im)
        try:
            fb=ImageFont.truetype('DejaVuSans-Bold.ttf',34); fm=ImageFont.truetype('DejaVuSans-Bold.ttf',24); fs=ImageFont.truetype('DejaVuSans.ttf',19); fss=ImageFont.truetype('DejaVuSans.ttf',16)
        except Exception: fb=fm=fs=fss=None
        grad=choose_grad(ctx); colhex=GRADIENTS[grad][1]; rgb=tuple(int(colhex.lstrip('#')[i:i+2],16) for i in (0,2,4))
        draw.text((margin,30),title,fill=(15,23,42),font=fb)
        letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i,r in enumerate(regs):
            x=margin+(i%cols)*tile_w; y=margin+header+(i//cols)*tile_h
            draw.rounded_rectangle([x,y,x+tile_w-35,y+tile_h-35],radius=22,outline=(203,213,225),width=2,fill=(255,255,255))
            draw.rectangle([x,y,x+tile_w-35,y+60],fill=rgb)
            draw.text((x+22,y+16),f"{letters[i]}  {r['figure_id']}",fill=(255,255,255),font=fm)
            # crude wrapping
            tt=str(r['title']); lines=[]
            while tt:
                lines.append(tt[:70]); tt=tt[70:]
            yy=y+82
            for line in lines[:3]: draw.text((x+28,yy),line,fill=(15,23,42),font=fs); yy+=28
            draw.text((x+28,y+205),f"File: {str(r['basename'])[:72]}",fill=(71,85,105),font=fss)
            draw.text((x+28,y+238),f"Analysis: {str(r['analysis_type'])[:70]}",fill=(71,85,105),font=fss)
            draw.text((x+28,y+290),"Use the exported component PNG/PDF/SVG for final manual layout.",fill=(100,116,139),font=fss)
        outdir=ctx.dir(folder); outdir.mkdir(parents=True,exist_ok=True); base=outdir/basename
        im.save(base.with_suffix('.png'),dpi=(600,600),optimize=True)
        im.save(base.with_suffix('.pdf'),resolution=150)
        svg=[f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>","<rect width='100%' height='100%' fill='white'/>",f"<text x='{margin}' y='55' font-family='Arial' font-size='34' font-weight='700' fill='#0f172a'>{html.escape(title)}</text>"]
        for i,r in enumerate(regs):
            x=margin+(i%cols)*tile_w; y=margin+header+(i//cols)*tile_h
            svg.append(f"<rect x='{x}' y='{y}' width='{tile_w-35}' height='{tile_h-35}' rx='22' fill='white' stroke='#cbd5e1' stroke-width='2'/>")
            svg.append(f"<rect x='{x}' y='{y}' width='{tile_w-35}' height='60' fill='{colhex}'/>")
            svg.append(f"<text x='{x+22}' y='{y+38}' font-family='Arial' font-size='24' font-weight='700' fill='white'>{chr(65+i)}  {html.escape(r['figure_id'])}</text>")
            svg.append(f"<text x='{x+28}' y='{y+105}' font-family='Arial' font-size='19' font-weight='700' fill='#0f172a'>{html.escape(str(r['title'])[:95])}</text>")
            svg.append(f"<text x='{x+28}' y='{y+225}' font-family='Arial' font-size='16' fill='#475569'>File: {html.escape(str(r['basename'])[:80])}</text>")
            svg.append(f"<text x='{x+28}' y='{y+258}' font-family='Arial' font-size='16' fill='#475569'>Analysis: {html.escape(str(r['analysis_type'])[:80])}</text>")
        svg.append('</svg>'); base.with_suffix('.svg').write_text('\n'.join(svg),encoding='utf-8')
    except Exception as e:
        ctx.skip(title,f'Panel assembly failed: {e}'); return None
    fid=ctx.next_id(); cap=f'{title}. Panel-index candidate listing component figures to assemble in manuscript layout software; individual component panels are exported separately as PNG/PDF/SVG.'
    caption_file(base.with_suffix('.caption'),cap)
    nsum=sum(int(r['n']) for r in regs)
    rec=FigRec(fid,title,'assembled manuscript panel index',', '.join([r['basename'] for r in regs]),folder,basename,str(base.with_suffix('.png')),str(base.with_suffix('.pdf')),str(base.with_suffix('.svg')),cap,grad,';'.join(GRADIENTS.get(grad,[])),nsum,'panel index')
    ctx.registry.append(dataclasses.asdict(rec)); return rec

# ----------------------------- main figure generation orchestration -----------------------------
def generate_all_figures(ctx,df,dq):
    style(); key_cols=[c for c in ['age_years_analysis','sex_clean','study_site_clean','poison_type_clean','specific_component_clean','date_date_of_admission','time_presentation_hrs_analysis','gcs_analysis','spo2_percent_analysis','death_bin','any_complication_bin','absconded_bin'] if c in df.columns]
    # A. data quality
    cohort_flow(ctx,df); missing_matrix(ctx,df,key_cols,'Key-variable missingness matrix','supp')
    if 'study_site_clean' in df.columns: crosstab_heat(ctx,df,'study_site_clean','poison_type_clean','Site × poisoning-type count heatmap','supp')
    if 'study_site_clean' in df.columns:
        miss=df.groupby('study_site_clean')[key_cols].apply(lambda x:x.isna().mean()*100).round(1) if key_cols else pd.DataFrame()
        if not miss.empty: heatmap(ctx,miss,'Missingness by study site',key_cols,'supp',fmt='.1f',cbar_label='Missing, %')
    completeness(ctx,df,key_cols+[v+'_analysis' for v in LAB_MAP.values() if v+'_analysis' in df.columns],'Variable completeness lollipop plot','supp')
    dq_map(ctx,dq)
    # B. demographics
    hist(ctx,df,'age_years_analysis','Age distribution: histogram + KDE','main')
    box_group(ctx,df,'age_years_analysis','sex_clean','Age distribution by sex','main')
    age_pyramid(ctx,df)
    stacked(ctx,df,'poison_type_clean','age_group','Age group distribution by poisoning type','supp',top=12)
    for col,title in [('sex_clean','Sex distribution overall'),('occupation_clean','Occupation distribution'),('living_area_clean','Living area distribution'),('presentation_area_clean','Presentation area distribution'),('study_site_clean','Study-site enrollment ranked by sample size')]: bar(ctx,df,col,title,'main' if col in ['study_site_clean','sex_clean'] else 'supp',top=20)
    for col,title in [('sex_clean','Site × sex heatmap'),('age_group','Site × age group heatmap'),('living_area_clean','Site × living area heatmap')]: crosstab_heat(ctx,df,'study_site_clean',col,title,'supp')
    # C. temporal
    time_trend(ctx,df,'date_date_of_admission','Monthly admission trend','main')
    time_trend(ctx,df,'date_date_of_admission','Monthly poisoning-type trend','main',hue='poison_type_clean')
    time_trend(ctx,df,'date_date_of_admission','Site-specific monthly enrollment trend','supp',hue='study_site_clean')
    bar(ctx,df,'admission_day_of_week','Day-of-week admission pattern','supp',top=7,horizontal=False)
    crosstab_heat(ctx,df,'poison_type_clean','admission_season','Seasonal/monthly variation by poisoning type','supp',normalize=True)
    box_group(ctx,df,'delay_ingestion_to_admission_hrs_analysis','poison_type_clean','Admission versus ingestion date delay by poisoning type','supp')
    if 'date_date_of_admission' in df.columns and 'death_bin' in df.columns:
        # monthly death rate
        d=df[['date_date_of_admission','death_bin']].dropna();
        if len(d)>=ctx.args.min_n and d['death_bin'].nunique()>1:
            d['month']=d['date_date_of_admission'].dt.to_period('M').dt.to_timestamp(); g=d.groupby('month')['death_bin'].agg(['sum','count']); g['rate']=g['sum']/g['count']*100; fig,ax=plt.subplots(figsize=(8,4)); grad=choose_grad(ctx); ax.plot(g.index,g['rate'],color=GRADIENTS[grad][1],lw=2); ax.set_ylabel('Death rate, %'); ax.set_xlabel('Calendar month'); ax.set_title('Temporal trend of mortality',loc='left',fontweight='bold'); ax.grid(axis='y',lw=.35,alpha=.25); add_n(ax,int(g['count'].sum())); save_fig(ctx,fig,'Temporal trend of mortality','temporal outcome rate',['date_date_of_admission','death_bin'],'supp','temporal_mortality_trend','Monthly death rate among records with death coding.',grad,int(g['count'].sum()))
    if 'date_date_of_admission' in df.columns and 'high_risk_poison_group' in df.columns: time_trend(ctx,df[df['high_risk_poison_group']==1],'date_date_of_admission','Temporal trend of high-risk poisoning categories','supp')
    # D. exposure patterns
    bar(ctx,df,'poison_type_clean','Ranked poisoning-type bar chart','main',top=20)
    static_treemap(ctx,df,'poison_type_clean','Treemap of poisoning categories','supp')
    bar(ctx,df,'specific_component_clean','Component-specific ranked bar chart','main',top=25)
    crosstab_heat(ctx,df,'poison_type_clean','specific_component_clean','Poisoning type × specific component heatmap','supp',toprow=14,topcol=14)
    for hue,title in [('sex_clean','Poisoning type by sex'),('age_group','Poisoning type by age group'),('occupation_clean','Poisoning type by occupation'),('living_area_clean','Poisoning type by living area'),('study_site_clean','Poisoning type by study site'),('presentation_area_clean','Poisoning type by presentation area')]: stacked(ctx,df,'poison_type_clean',hue,title,'supp',top=12)
    bubble(ctx,df,'study_site_clean','poison_type_clean','death_bin','Bubble plot: poisoning type frequency by site and death rate','main')
    rare=df['poison_type_clean'].value_counts() if 'poison_type_clean' in df else pd.Series(dtype=int)
    if not rare.empty:
        tmp=pd.DataFrame({'rare_poison_type':rare[rare<ctx.args.rare_threshold].index.repeat(rare[rare<ctx.args.rare_threshold].values)}) if (rare<ctx.args.rare_threshold).any() else pd.DataFrame()
        if not tmp.empty: bar(ctx,tmp,'rare_poison_type','Rare poisoning categories supplementary figure','supp',top=30)
    # E. clinical presentation
    symptom_prevalence(ctx,df); symbins=[slug(s)+'_bin' for s in SYMPTOMS if slug(s)+'_bin' in df.columns]
    binary_heat(ctx,df,'poison_type_clean',symbins,'Symptom prevalence by poisoning type heatmap','main')
    if 'death_bin' in df.columns: binary_heat(ctx,df,'death_bin',symbins,'Symptom prevalence by death outcome heatmap','supp',top=2)
    symptom_combos(ctx,df); symptom_network(ctx,df)
    hist(ctx,df,'symptom_burden_score','Symptom burden score distribution','supp',bins=20)
    box_group(ctx,df,'symptom_burden_score','poison_type_clean','Symptom burden by poisoning type','main')
    box_group(ctx,df,'symptom_burden_score','death_bin','Symptom burden by death outcome','supp')
    if 'death_bin' in df.columns:
        u=logistic_or_table(ctx,df,'death_bin',symbins,'symptom_death'); u.to_csv(ctx.dir('tables')/'univariable_symptom_death_odds_ratios.csv',index=False); forest_plot(ctx,u,'Forest plot of symptoms associated with death','main')
    # F. vital signs
    for c,t in [('temperature_c_analysis','Temperature distribution'),('pulse_bpm_analysis','Pulse distribution'),('bp_systolic_analysis','Parsed systolic BP distribution'),('bp_diastolic_analysis','Parsed diastolic BP distribution'),('spo2_percent_analysis','SpO2 distribution'),('gcs_analysis','GCS distribution'),('respiratory_rate_analysis','Respiratory rate distribution')]: hist(ctx,df,c,t,'supp')
    vitals=['temperature_c_analysis','pulse_bpm_analysis','bp_systolic_analysis','bp_diastolic_analysis','spo2_percent_analysis','gcs_analysis','respiratory_rate_analysis']
    for c in vitals:
        box_group(ctx,df,c,'poison_type_clean',f'{pretty(c)} by poisoning type','supp')
        box_group(ctx,df,c,'death_bin',f'{pretty(c)} by death outcome','supp')
    corr_heat(ctx,df,vitals,'Correlation heatmap of vitals','supp'); pair_matrix(ctx,df,[c for c in vitals if c in df.columns][:5])
    for c,t in [('low_gcs_bin','Low GCS prevalence by poisoning type'),('hypoxia_bin','Hypoxia prevalence by poisoning type'),('hypotension_bin','Hypotension prevalence by poisoning type')]: rate_group(ctx,df,'poison_type_clean',c,t,'main')
    # G. labs
    labcols=[v+'_analysis' for v in LAB_MAP.values() if v+'_analysis' in df.columns]
    completeness(ctx,df,labcols,'Lab completeness figure','supp')
    for c in labcols:
        hist(ctx,df,c,f'Lab distribution: {pretty(c)}','supp',bins=30,logx=False)
        box_group(ctx,df,c,'poison_type_clean',f'{pretty(c)} by poisoning type','supp')
        box_group(ctx,df,c,'death_bin',f'{pretty(c)} by death outcome','supp')
    corr_heat(ctx,df,labcols[:16],'Correlation heatmap of labs','supp'); lab_abnormality(ctx,df); pca_labs(ctx,df)
    if 'death_bin' in df.columns:
        ul=logistic_or_table(ctx,df,'death_bin',labcols[:12],'lab_death'); ul.to_csv(ctx.dir('tables')/'univariable_lab_death_odds_ratios.csv',index=False); forest_plot(ctx,ul,'Forest plot of lab predictors of mortality','supp')
    # H. treatment/support
    for c,t in [('fluid_l_analysis','Fluid volume distribution'),('oxygen_l_min_analysis','Oxygen support distribution')]: hist(ctx,df,c,t,'supp')
    for c,t in [('ng_suction_bin','NG suction frequency by poisoning type'),('dialysis_cycles_analysis','Dialysis cycle frequency by poisoning type'),('ventilation_support_bin','Ventilation support frequency by poisoning type'),('operation_bin','Operation frequency by poisoning type')]:
        if c=='dialysis_cycles_analysis' and c in df.columns:
            tmp=df.copy(); tmp['dialysis_any_bin']=(tmp[c]>0).astype(float).where(tmp[c].notna(),np.nan); rate_group(ctx,tmp,'poison_type_clean','dialysis_any_bin',t,'main')
        else: rate_group(ctx,df,'poison_type_clean',c,t,'main')
    medication_frequency(ctx,df); ddose=dose_duration_summary(ctx,df)
    if ddose is not None: ddose.to_csv(ctx.dir('tables')/'dose_duration_cleaning_summary.csv',index=False)
    box_group(ctx,df,'treatment_intensity_score','poison_type_clean','Treatment intensity score by poisoning type','main')
    box_group(ctx,df,'treatment_intensity_score','death_bin','Treatment intensity score by death outcome','supp')
    binary_heat(ctx,df,'study_site_clean',[c for c in ['ng_suction_bin','ventilation_support_bin','operation_bin','dialysis_any_bin'] if c in df.columns],'Treatment heatmap by study site','supp')
    # I. outcomes/follow-up
    outcomes=[]
    if 'death_bin' in df.columns: outcomes.append(('Death',df['death_bin']))
    if 'any_complication_bin' in df.columns: outcomes.append(('Any complication',df['any_complication_bin']))
    if 'absconded_bin' in df.columns: outcomes.append(('Absconded/DORB',df['absconded_bin']))
    if outcomes:
        tmp=pd.DataFrame({'outcome':[k for k,s in outcomes for _ in range(int(s.fillna(0).sum()))]}); bar(ctx,tmp,'outcome','Overall outcome distribution','main',top=10)
    for grp,t in [('poison_type_clean','Death frequency and death rate by poisoning type'),('study_site_clean','Death rate by study site'),('age_group','Death rate by age group'),('sex_clean','Death rate by sex'),('time_to_presentation_category','Death rate by time-to-presentation category')]: rate_group(ctx,df,grp,'death_bin',t,'main')
    rate_group(ctx,df,'poison_type_clean','any_complication_bin','Complication frequency by poisoning type','main')
    rate_group(ctx,df,'study_site_clean','absconded_bin','Absconded/DORB frequency by site','supp')
    bar(ctx,df,'followup_status','Back-to-normal-health follow-up outcome summary','supp',top=10)
    stacked(ctx,df,'poison_type_clean','followup_status','Follow-up outcome by poisoning type','supp',top=12)
    rate_group(ctx,df,'poison_type_clean','severe_outcome_bin','Mortality and complication combined severity endpoint figure','main')
    if {'poison_type_clean','death_bin','any_complication_bin'}.issubset(df.columns):
        d=df[['poison_type_clean','death_bin','any_complication_bin']].dropna();
        if len(d)>=ctx.args.min_n:
            g=d.groupby('poison_type_clean').agg(n=('death_bin','size'),death=('death_bin','mean'),comp=('any_complication_bin','mean')).reset_index(); g=g.sort_values('n',ascending=False).head(15); grad=choose_grad(ctx); fig,ax=plt.subplots(figsize=(7,5)); sc=ax.scatter(g['death']*100,g['comp']*100,s=np.sqrt(g['n'])*28,c=g['n'],cmap=get_cmap(grad),edgecolor='white',linewidth=.6); [ax.text(r.death*100,r.comp*100,str(r.poison_type_clean)[:18],fontsize=6,ha='center',va='center') for r in g.itertuples()]; ax.set_xlabel('Death rate, %'); ax.set_ylabel('Complication rate, %'); ax.set_title('Bubble plot: poisoning frequency, death rate, and complication rate', loc='left', fontweight='bold'); fig.colorbar(sc,ax=ax,fraction=.035,pad=.02,label='Patients, n'); save_fig(ctx,fig,'Bubble plot showing poisoning type frequency, death rate, and complication rate together','outcome bubble',['poison_type_clean','death_bin','any_complication_bin'],'main','outcome_frequency_death_complication_bubble','Bubble size/color indicate frequency; axes show death and complication rates.',grad,int(g['n'].sum()))
    # J. association analyses/modeling
    predictors=['age_years_analysis','sex_clean','study_site_clean','living_area_clean','occupation_clean','poison_type_clean','delayed_presentation_bin','gcs_analysis','spo2_percent_analysis','bp_systolic_analysis','hypotension_bin','low_gcs_bin','hypoxia_bin','ng_suction_bin','ventilation_support_bin','treatment_intensity_score']+symbins[:10]
    if 'death_bin' in df.columns:
        u=logistic_or_table(ctx,df,'death_bin',predictors,'death'); u.to_csv(ctx.dir('tables')/'univariable_death_associations.csv',index=False); forest_plot(ctx,u,'Univariable association plots for death/complication: death endpoint','main')
        adj=multivariable_logistic(ctx,df,'death_bin',['age_years_analysis','time_presentation_hrs_analysis','gcs_analysis','spo2_percent_analysis','bp_systolic_analysis','symptom_burden_score','treatment_intensity_score'],'Multivariable logistic regression for death'); adj.to_csv(ctx.dir('tables')/'multivariable_death_logistic_regression.csv',index=False); forest_plot(ctx,adj,'Forest plot of adjusted odds ratios with 95% CI: death','main')
        rf_importance(ctx,df,'death_bin',predictors); roc_calibration(ctx,df,'death_bin',['age_years_analysis','time_presentation_hrs_analysis','gcs_analysis','spo2_percent_analysis','bp_systolic_analysis','symptom_burden_score','treatment_intensity_score'])
    if 'severe_outcome_bin' in df.columns:
        adj=multivariable_logistic(ctx,df,'severe_outcome_bin',['age_years_analysis','time_presentation_hrs_analysis','gcs_analysis','spo2_percent_analysis','bp_systolic_analysis','symptom_burden_score','treatment_intensity_score'],'Multivariable logistic regression for composite severe outcome'); adj.to_csv(ctx.dir('tables')/'multivariable_severe_outcome_logistic_regression.csv',index=False); forest_plot(ctx,adj,'Forest plot of adjusted odds ratios: composite severe outcome','main')
    # K. interactive and panels
    interactive_outputs(ctx,df)
    assemble_panel(ctx,'cohort_flow|study_site|sex_distribution|age_pyramid','Figure 1: Cohort profile + site distribution + demographic profile','figure_1_cohort_demographics')
    assemble_panel(ctx,'poisoning_type|monthly|component|bubble_plot_poisoning','Figure 2: Poisoning-type epidemiology and temporal trends','figure_2_exposure_temporal')
    assemble_panel(ctx,'symptom_prevalence|symptom_burden|symptoms_associated|symptom_cooccurrence','Figure 3: Clinical presentation and symptom patterns','figure_3_clinical_symptoms')
    assemble_panel(ctx,'treatment|outcome|death_rate|complication','Figure 4: Treatment/support and outcomes','figure_4_treatment_outcomes')
    assemble_panel(ctx,'forest|adjusted|rf_importance|roc','Figure 5: Risk factors for mortality/severe outcome','figure_5_risk_factors')
    assemble_panel(ctx,'missingness|data_quality|completeness','Supplementary Figure S1: missingness/data-quality','supplementary_figure_s1_missingness_quality')
    assemble_panel(ctx,'site_|site ×|Site','Supplementary Figure S2: extended site-level analyses','supplementary_figure_s2_site')
    assemble_panel(ctx,'lab|symptom','Supplementary Figure S3: extended symptom and lab heatmaps','supplementary_figure_s3_symptom_lab')
    assemble_panel(ctx,'follow|absconded|severe_outcome','Supplementary Figure S4: extended outcome/follow-up analyses','supplementary_figure_s4_outcome_followup')

# ----------------------------- command line -----------------------------
def parse_args():
    ap=argparse.ArgumentParser(description='Generate publication-quality figures from poison.xlsx')
    ap.add_argument('--input',required=True,help='Input Excel workbook, e.g., poison.xlsx')
    ap.add_argument('--outdir',required=True,help='Output directory')
    ap.add_argument('--overwrite',action='store_true',help='Overwrite output directory if it exists')
    ap.add_argument('--max-rows',type=int,default=None,help='Optional row limit for smoke testing')
    ap.add_argument('--min-n',type=int,default=30,help='Minimum non-missing records to attempt a plot')
    ap.add_argument('--min-group-n',type=int,default=20,help='Minimum denominator per group for rate/model displays')
    ap.add_argument('--min-events',type=int,default=20,help='Minimum events for modeling')
    ap.add_argument('--rare-threshold',type=int,default=30,help='Count threshold for rare category figure')
    ap.add_argument('--no-git',action='store_true',help='Do not attempt Git initialization')
    return ap.parse_args()

def main():
    args=parse_args(); ctx=Ctx(args); setup(ctx)
    script_path=Path(__file__).resolve(); script_hash=sha256_file(script_path); input_hash=sha256_file(ctx.input)
    shutil.copy2(script_path,ctx.dir('ver')/script_path.name)
    write_json(ctx.dir('logs')/'package_versions.json',package_versions())
    try:
        raw,sheets,dd=load_workbook(ctx)
        public,dq=clean_dataset(ctx,raw,sheets)
        write_clean(ctx,public)
        create_tables(ctx,public,dq,dd)
        generate_all_figures(ctx,public,dq)
        # save final tables after model figures generated
        pd.DataFrame(ctx.skipped).to_csv(ctx.dir('logs')/'skipped_analyses.csv',index=False)
        pd.DataFrame(ctx.warnings,columns=['warning']).to_csv(ctx.dir('logs')/'warnings_log.csv',index=False)
        write_registry_manifest(ctx,input_hash,script_hash)
        gallery(ctx); summary_report(ctx,public,dq); try_git(ctx)
        ctx.log.info('Completed run %s with %s figures. Output: %s',ctx.run_id,len(ctx.registry),ctx.out)
    except Exception as e:
        ctx.warn('Fatal error: '+str(e)); (ctx.dir('logs')/'fatal_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_registry_manifest(ctx,input_hash,script_hash); raise

if __name__=='__main__':
    main()
