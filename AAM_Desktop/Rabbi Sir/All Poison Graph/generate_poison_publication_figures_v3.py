#!/usr/bin/env python3
# PATCH VERSION: 20260614_V3_TEXTSAFE2
from __future__ import annotations
import argparse, calendar, hashlib, json, logging, math, os, platform, re, subprocess, sys, textwrap, traceback, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np, pandas as pd
# Quiet non-fatal Excel/pandas warnings so real errors remain visible.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='Data Validation extension is not supported*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message=r'set_ticklabels\(\) should only be used.*')

OPTIONAL_IMPORTS={}
try:
    import yaml; OPTIONAL_IMPORTS['yaml']=True
except Exception:
    yaml=None; OPTIONAL_IMPORTS['yaml']=False
try:
    from scipy.cluster.hierarchy import linkage, leaves_list; OPTIONAL_IMPORTS['scipy']=True
except Exception:
    linkage=leaves_list=None; OPTIONAL_IMPORTS['scipy']=False
try:
    import statsmodels.api as sm; OPTIONAL_IMPORTS['statsmodels']=True
except Exception:
    sm=None; OPTIONAL_IMPORTS['statsmodels']=False
try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.calibration import calibration_curve
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    OPTIONAL_IMPORTS['sklearn']=True
except Exception:
    SimpleImputer=LogisticRegression=roc_auc_score=roc_curve=StratifiedKFold=cross_val_predict=calibration_curve=PCA=StandardScaler=None
    OPTIONAL_IMPORTS['sklearn']=False
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
OPTIONAL_IMPORTS['matplotlib']=True
try:
    import seaborn as sns; OPTIONAL_IMPORTS['seaborn']=True
except Exception:
    sns=None; OPTIONAL_IMPORTS['seaborn']=False
try:
    import networkx as nx; OPTIONAL_IMPORTS['networkx']=True
except Exception:
    nx=None; OPTIONAL_IMPORTS['networkx']=False
try:
    import plotly.express as px; OPTIONAL_IMPORTS['plotly']=True
except Exception:
    px=None; OPTIONAL_IMPORTS['plotly']=False
try:
    from PIL import Image; OPTIONAL_IMPORTS['pillow']=True
except Exception:
    Image=None; OPTIONAL_IMPORTS['pillow']=False
try:
    import pyarrow  # noqa
    OPTIONAL_IMPORTS['pyarrow']=True
except Exception:
    OPTIONAL_IMPORTS['pyarrow']=False

SEED=42; np.random.seed(SEED)
MAIN_SHEET='All Poison Data'; UNIQUE_NAMES_SHEET='Unique Names'; DROPDOWNS_SHEET='DropDowns'
OUTPUT_SUBDIRS=['00_logs','01_clean_data','02_tables','03_main_figures_single','04_supplementary_figures_single','05_exploratory_figures_single','06_interactive_html','07_quality_control','08_version_control']
PII_PATTERNS=["Patient's Name",'Registration Number','Contact number','Address']
SYMPTOM_COLUMNS=['Vomited after ingestion','Fever','Vomiting','Diarrhoea','Abdominal pain','Abdominal distension','Cough','Shortness of breath','Heart burn','Oral ulcers','Leg swelling','Reduced urine output','Jaundice','Unconsciousness','Convulsion','Chest pain','Bleeding tendency','Shock']
FOLLOWUP_BASE_NAMES=['Back to normal health','Not entirely back to normal health','Requires help in daily activities','Death (mention the date)']
PLANNED_ANALYSES=[('fig1_cohort_demographics','main'),('fig2_poisoning_epidemiology','main'),('fig3_temporal_site','main'),('fig4_clinical_severity','main'),('fig5_treatment_outcomes','main'),('fig6_modeling','main'),('supp_missingness_matrix','supplementary'),('supp_data_quality_dashboard','supplementary'),('supp_all_poison_types','supplementary'),('supp_name_harmonization','supplementary'),('supp_stratified_context','supplementary'),('supp_labs_overview','supplementary'),('supp_vitals_lab_correlations','supplementary'),('supp_alluvial_site_type_outcome','supplementary'),('supp_alluvial_age_type_outcome','supplementary'),('supp_alluvial_type_treatment_outcome','supplementary'),('supp_followup_pathway','supplementary'),('supp_absconded_rates','supplementary'),('supp_delay_amount_vs_severity','supplementary'),('supp_site_heterogeneity','supplementary'),('supp_sparse_category_sensitivity','supplementary'),('supp_implausible_numeric_sensitivity','supplementary'),('supp_model_diagnostics','supplementary'),('supp_pca_labs','supplementary'),('exp_component_heatmaps','exploratory'),('exp_outcome_panels','exploratory'),('exp_temporal_qc','exploratory')]
CATEGORICAL_BASE=['#1b4965','#ca6702','#6a4c93','#2a9d8f','#bc4749','#5c677d','#355070','#b56576','#7f5539','#588157']
DIVERGING_TEAL=['#b2182b','#ef8a62','#fddbc7','#d1e5f0','#67a9cf','#2166ac']
PREMIUM_GRADIENTS={k:v for k,v in {
'navy_teal_mint':['#102542','#1f6f78','#b8f2e6'],'charcoal_indigo_lavender':['#2d3142','#4f5d75','#bfc0c0'],'burgundy_rose_blush':['#6d213c','#c1666b','#f4d1d1'],'forest_jade_sage':['#1b4332','#40916c','#d8f3dc'],'plum_violet_lilac':['#3c1642','#7f4f9a','#d9c2f0'],'sienna_amber_cream':['#8d5a3b','#d9a441','#f7eedb'],'slate_cyan_pearl':['#44546a','#5bc0be','#edf6f9'],'espresso_bronze_sand':['#4a2c2a','#a97142','#ead2ac'],'crimson_coral_peach':['#7f1d1d','#f08080','#ffd6cc'],'midnight_royal_ice':['#14213d','#3a86ff','#dff6ff'],'deep_green_lime_mist':['#234f1e','#6db65b','#eef7ea'],'aubergine_mauve_silver':['#4a1942','#9d6b9f','#e6d5e8'],'marine_turquoise_fog':['#0b3954','#087e8b','#bfd7ea'],'copper_gold_ivory':['#8c4a2f','#c89b3c','#fff8e7'],'rust_apricot_chalk':['#9c6644','#dda15e','#fefae0'],'ink_blue_sky_smoke':['#1d3557','#457b9d','#f1faee'],'petrol_surf_foam':['#264653','#2a9d8f','#e9f5f2'],'mulberry_orchid_shell':['#5a189a','#9d4edd','#f8edff'],'brick_tangerine_cream':['#8d0801','#e85d04','#fff3e2'],'graphite_azure_snow':['#3c3c3c','#2f80ed','#f7fbff']}.items()}
GRADIENT_USE_LIMIT=3

mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9,'figure.titlesize':16,'figure.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
if sns is not None: sns.set_theme(style='whitegrid',context='paper')

@dataclass
class FigureRecord:
    figure_id:str; title:str; section:str; base_filename:str; variables:str; analysis_type:str; gradient:str; n:Optional[int]
    caption_file:str; png_file:str; pdf_file:str; svg_file:str; scientific_validity:int; legibility:int; manuscript_suitability:int; statistical_appropriateness:int; visual_quality:int; recommended_tier:str; qa_notes:str

class AnalysisState:
    def __init__(self,outdir:Path,run_id:str):
        self.outdir=outdir; self.run_id=run_id; self.figure_registry=[]; self.gradient_usage=Counter(); self.skip_rows=[]
        self.coverage_rows=[{'analysis_id':a,'planned_tier':t,'status':'planned','figure_id':'','reason':''} for a,t in PLANNED_ANALYSES]
    def mark_coverage(self,analysis_id,status,figure_id='',reason=''):
        for r in self.coverage_rows:
            if r['analysis_id']==analysis_id: r.update({'status':status,'figure_id':figure_id,'reason':reason}); break
        if status=='skipped': self.skip_rows.append({'analysis_id':analysis_id,'reason':reason})
    def choose_gradient(self):
        for k in PREMIUM_GRADIENTS:
            if self.gradient_usage[k] < GRADIENT_USE_LIMIT: self.gradient_usage[k]+=1; return k
        k=list(PREMIUM_GRADIENTS)[0]; self.gradient_usage[k]+=1; return k

def ensure_dir(path:Path): path.mkdir(parents=True,exist_ok=True)
def sha256_of_file(path:Path)->str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def normalize_space(s:str)->str: return re.sub(r'\s+',' ',s).strip()
def normalize_text(x:Any)->Optional[str]:
    if x is None: return None
    try:
        if pd.isna(x): return None
    except (TypeError, ValueError):
        pass
    s=normalize_space(str(x).replace('\u00a0',' '))
    if s=='' or s.lower() in {'nan','none','n/a','na','#ref!','#value!','#name?','ref'}: return None
    return s
def normalize_text_lower(x:Any)->Optional[str]:
    s=normalize_text(x); return s.lower() if s is not None else None
wrap=lambda s,width=26:'\n'.join(textwrap.wrap(str(s if s is not None else 'Missing'),width=width))
def slugify(s:str)->str: return re.sub(r'_+','_',re.sub(r'[^a-zA-Z0-9]+','_',str(s).strip().lower())).strip('_')

def make_unique_headers(headers:Sequence[Any]):
    counts=Counter(); out=[]; dd=[]
    for idx,raw in enumerate(headers):
        raw_name=normalize_text(raw) or f'unnamed_col_{idx+1}'; counts[raw_name]+=1; new=raw_name if counts[raw_name]==1 else f'{raw_name}__dup{counts[raw_name]}'
        out.append(new); dd.append({'column_position':idx+1,'raw_header':raw_name,'analysis_header':new,'is_duplicate':counts[raw_name]>1})
    return out,dd

def parse_numeric(x:Any, plausible:Optional[Tuple[float,float]]=None):
    raw=normalize_text(x)
    if raw is None: return np.nan,'missing'
    nums=re.findall(r'-?\d+(?:\.\d+)?',raw.lower())
    if not nums: return np.nan,'unparsed'
    vals=[float(v) for v in nums]; val=float(np.mean(vals[:2])) if len(vals)>=2 and any(sep in raw for sep in ['-','to','–','/']) else float(vals[0])
    flag='ok' if plausible is None or plausible[0]<=val<=plausible[1] else 'implausible'
    return val,flag

def split_bp(x:Any):
    raw=normalize_text(x)
    if raw is None: return np.nan,np.nan,'missing'
    m=re.search(r'(\d+(?:\.\d+)?)\s*[/-]\s*(\d+(?:\.\d+)?)',raw)
    if not m: return np.nan,np.nan,'unparsed'
    sbp,dbp=float(m.group(1)),float(m.group(2)); flag=[]
    if sbp<40 or sbp>300: flag.append('implausible_sbp')
    if dbp<20 or dbp>200: flag.append('implausible_dbp')
    return sbp,dbp,';'.join(flag) if flag else 'ok'

def parse_mixed_date(x:Any):
    if pd.isna(x): return pd.NaT,'missing'
    if isinstance(x,pd.Timestamp): return x.normalize(),'ok'
    if isinstance(x,datetime): return pd.Timestamp(x).normalize(),'ok'
    if isinstance(x,(int,float)) and not pd.isna(x):
        v=float(x)
        if 20000<=v<=60000: return pd.Timestamp('1899-12-30')+pd.to_timedelta(v,unit='D'),'ok_excel_serial'
        return pd.NaT,'invalid_numeric'
    s=normalize_text(x)
    if s is None: return pd.NaT,'missing'
    m=re.search(r'(\d{1,4}[./-]\d{1,2}[./-]\d{1,4})',s)
    cand=(m.group(1) if m else s).replace('.','/').replace('-','/')
    for dayfirst in [True,False]:
        try: return pd.Timestamp(pd.to_datetime(cand,errors='raise',dayfirst=dayfirst)).normalize(), f'ok_text_dayfirst_{dayfirst}'
        except Exception: pass
    try: return pd.Timestamp(pd.to_datetime(s,errors='raise',dayfirst=True)).normalize(),'ok_longform'
    except Exception: return pd.NaT,'unparsed'

def bool_from_series_value(x:Any,treat_date_as_yes=True):
    if pd.isna(x): return None
    if isinstance(x,(pd.Timestamp,datetime)): return 1 if treat_date_as_yes else None
    s=normalize_text_lower(x)
    if s is None: return None
    yes={'yes','y','1','true','present','positive','death','dead','died','absconded','dorb','survived without complications'}; no={'no','n','0','false','absent','negative','nil','none'}
    if s in yes or s.startswith(('yes','dead','died','absconded','dorb')): return 1
    if s in no or s.startswith('no'): return 0
    if treat_date_as_yes and re.search(r'\d{1,4}[./-]\d{1,2}[./-]\d{1,4}',s): return 1
    return None

def wilson_ci(k:int,n:int):
    if n<=0: return np.nan,np.nan,np.nan
    z=1.959963984540054; phat=k/n; denom=1+z**2/n; centre=(phat+z**2/(2*n))/denom; half=z*math.sqrt((phat*(1-phat)/n)+(z**2/(4*n**2)))/denom
    return phat,max(0.0,centre-half),min(1.0,centre+half)

def maybe_title(x):
    s=normalize_text(x)
    if s is None: return None
    upper=s.upper()
    return upper if upper in {'DMCH','RMCH','CMCH','SOMCH','SZMCH','MMCH','RPMCH','MARMC','KMCH','JMCH','SBMCH'} else s.title()

def safe_lower(x):
    s=normalize_text(x)
    return s.lower() if s is not None else None

def categorical_clean(s:pd.Series)->pd.Series:
    if s is None: return pd.Series(dtype='object')
    return s.map(normalize_text).astype('object')
def top_n_with_other(s:pd.Series,n=10)->pd.Series:
    vals=s.fillna('Missing'); top=vals.value_counts().head(n).index; return vals.where(vals.isin(top),'Other')

def infer_study_window(adm_dates:pd.Series):
    cand=adm_dates.dropna(); cand=cand[(cand.dt.year>=2018)&(cand.dt.year<=datetime.now().year+2)]
    if cand.empty: return pd.Timestamp('2024-01-01'),pd.Timestamp(datetime.now().date()),{'method':'fallback_default','candidate_n':0}
    year_counts=cand.dt.year.value_counts().sort_index(); good=year_counts[year_counts>=max(5,0.01*year_counts.max())].index.tolist(); dense=cand[cand.dt.year.isin(good)]
    return dense.min(),dense.max(),{'method':'dense_year_window','candidate_n':int(cand.shape[0]),'good_years':list(map(int,good)),'year_counts':{str(k):int(v) for k,v in year_counts.to_dict().items()}}

def derive_age_group(age:pd.Series): return pd.cut(age,bins=[0,12,18,25,40,60,120],labels=['0–12','13–17','18–24','25–39','40–59','60+'],include_lowest=True)
def derive_season(dt:pd.Series):
    mapping={12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',10:'Autumn',11:'Autumn'}
    return dt.dt.month.map(mapping) if not dt.isna().all() else pd.Series(index=dt.index,dtype='object')

def rule_based_poison_type(value):
    txt=normalize_text(value)
    if txt is None: return 'Unknown'
    s=txt.lower(); rules=[(r'mixed\s*opc|combined\s*opc','Mixed OPC'),(r'\bopc\b|organoph|op\b|0pc','OPC'),(r'benz|clonaz|diazep|rivot|sedat','Sedative' if 'drug' not in s else 'Drug Overdose'),(r'mult.*drug|multi drug|multiple drug','Multidrug'),(r'drug over|overdose|amitrip|domperidone|diclofenac|paracetamol','Drug Overdose'),(r'harpic|herpic|vixol|toilet|bleach|detergent|savlon|household','Household Product' if 'acid' not in s else 'Corrosive'),(r'paraqu','Paraquat'),(r'rat|rodent','Rat Killer'),(r'aluminium phosphide|gas tablet|alp\b','Aluminium Phosphide'),(r'insect|pyreth|cyperm|lambda|cockroach|ant killer','Insecticide'),(r'acid|corros|battery acid|toilet cleaner','Corrosive'),(r'bee sting|bee bite','Bee Sting'),(r'insect bite','Insect Bite'),(r'alcohol','Alcohol Overdose'),(r'street','Street Poisoning'),(r'other|unknown|nan','Other')]
    for p,l in rules:
        if re.search(p,s): return l
    return maybe_title(txt) or 'Other'

def rule_based_component(value):
    txt=normalize_text(value)
    if txt is None: return 'Unknown/Other'
    s=txt.lower(); rules=[(r'unknown|other|nan','Unknown/Other'),(r'chlorp','Chlorpyrifos'),(r'cyperm','Cypermethrin'),(r'pyreth','Pyrethroid'),(r'toilet|harpic|herpic|vixol','Toilet Cleaner'),(r'benz|clonaz|diazep|rivot','Benzodiazepine'),(r'paracetamol','Paracetamol'),(r'deterg|bleach|wheel powder','Detergent/Bleach'),(r'disinfect|savlon','Disinfectant'),(r'rodent|rat','Rodenticide'),(r'gas tablet|aluminium phosphide|alp','Gas Tablet'),(r'herbicide|plant hormone','Herbicide'),(r'organoph|opc','OPC'),(r'insecticide','Insecticide Other'),(r'solvent|fuel|kerosene|diesel|petrol','Solvent/Fuel'),(r'acid','Acid Cleaner'),(r'sedative','Sedative'),(r'copper','Copper Sulfate'),(r'vitamin|supplement','Vitamin/Supplement'),(r'antidepress','Antidepressant')]
    for p,l in rules:
        if re.search(p,s): return l
    return maybe_title(txt) or 'Unknown/Other'

def extract_dropdown_values(dropdowns:pd.DataFrame):
    result=defaultdict(list)
    for col in dropdowns.columns:
        nonnull=[normalize_text(v) for v in dropdowns[col].tolist() if normalize_text(v) is not None]
        if len(nonnull)>=2: result[nonnull[0]]=nonnull[1:]
    return result

def build_harmonization_maps(unique_names:pd.DataFrame, dropdown_map):
    dropdown_poison=dropdown_map.get('Types of Poson',[])+dropdown_map.get('Types of poisoning',[])
    manual_types=['OPC','Mixed OPC','Sedative','Drug Overdose','Multidrug','Benzodiazepine','Herpic','Household Product','Paraquat','Rat Killer','Insecticide','Aluminium Phosphide','Corrosive','Street Poisoning','Bee Sting','Insect Bite','Alcohol Overdose','Paracetamol','Detergent','Unknown','Other','Chemical','Copper Sulfate','Plant Poisoning']
    manual_components=['Unknown/Other','Toilet Cleaner','Benzodiazepine','OPC','Pyrethroid','Paracetamol','Disinfectant','Chlorpyrifos','Herbicide','Insecticide Other','Detergent/Bleach','Rodenticide','Harpic','Antidepressant','Gas Tablet','Sedative','Vitamin/Supplement','Solvent/Fuel','Cypermethrin','Organophosphate','Copper Sulfate','AlP','Acid Cleaner']
    canonical_types=sorted({normalize_text(v) for v in dropdown_poison+manual_types if normalize_text(v)}); canonical_components=sorted({normalize_text(v) for v in manual_components if normalize_text(v)})
    poison_map={v:rule_based_poison_type(v) for v in set(unique_names.iloc[:,0].dropna().map(normalize_text_lower).dropna().tolist())}
    comp_map={v:rule_based_component(v) for v in set(unique_names.iloc[:,1].dropna().map(normalize_text_lower).dropna().tolist())}
    return poison_map,comp_map,canonical_types,canonical_components

def load_workbook_sheets(input_path:Path, logger):
    logger.info('Reading workbook with duplicate-safe header handling')
    all_raw=pd.read_excel(input_path,sheet_name=MAIN_SHEET,header=None,dtype=object); headers,data_dict=make_unique_headers(all_raw.iloc[0].tolist())
    df=all_raw.iloc[1:].copy(); df.columns=headers; df.reset_index(drop=True,inplace=True)
    unique_names=pd.read_excel(input_path,sheet_name=UNIQUE_NAMES_SHEET,dtype=object); dropdowns=pd.read_excel(input_path,sheet_name=DROPDOWNS_SHEET,dtype=object)
    return df,unique_names,dropdowns,data_dict

def build_analysis_dataset(df, unique_names, dropdowns, logger):
    metadata={}; dropdown_map=extract_dropdown_values(dropdowns); poison_map,comp_map,canonical_types,canonical_components=build_harmonization_maps(unique_names, dropdown_map)
    raw=df.copy(); clean=pd.DataFrame(index=raw.index); clean['row_id']=np.arange(1,len(raw)+1)
    for src,dst in [('Study site','study_site_raw'),('Sex','sex_raw'),('Living area','living_area_raw'),('Presentation area','presentation_area_raw'),('Occupation','occupation_raw')]:
        clean[dst]=categorical_clean(raw.get(src))
    clean['study_site']=clean['study_site_raw'].map(maybe_title)
    sex_lower=clean['sex_raw'].map(safe_lower)
    clean['sex']=sex_lower.map({'male':'Male','female':'Female','mel':'Male'})
    clean['living_area']=clean['living_area_raw'].map(maybe_title)
    clean['presentation_area']=clean['presentation_area_raw'].map(maybe_title)
    clean['occupation']=clean['occupation_raw'].map(maybe_title)
    metadata['pii_excluded_columns']=[c for c in raw.columns if any(p.lower() in c.lower() for p in PII_PATTERNS)]
    raw_poison=categorical_clean(raw.get('Types of poisoning')); clean['poison_type_raw']=raw_poison; clean['poison_type']=raw_poison.map(lambda x: poison_map.get(safe_lower(x), rule_based_poison_type(x)) if safe_lower(x) is not None else 'Unknown')
    raw_component=categorical_clean(raw.get('Name of the specific component')); clean['component_raw']=raw_component; clean['component']=raw_component.map(lambda x: comp_map.get(safe_lower(x), rule_based_component(x)) if safe_lower(x) is not None else 'Unknown/Other')
    clean['poison_type_major']=top_n_with_other(clean['poison_type'],10); clean['component_major']=top_n_with_other(clean['component'],12)
    if 'Date of admission' in raw.columns:
        parsed=raw['Date of admission'].map(parse_mixed_date); clean['admission_date']=parsed.map(lambda z:z[0]); clean['admission_date_parse_flag']=parsed.map(lambda z:z[1])
    if 'Date of ingestion' in raw.columns:
        parsed=raw['Date of ingestion'].map(parse_mixed_date); clean['ingestion_date']=parsed.map(lambda z:z[0]); clean['ingestion_date_parse_flag']=parsed.map(lambda z:z[1])
    study_start,study_end,study_meta=infer_study_window(clean['admission_date']); metadata['study_window']={'start':str(study_start.date()),'end':str(study_end.date()),**study_meta}
    def flag_date_quality(dt,parse_flag):
        if pd.isna(dt): return parse_flag
        if dt.year<2018: return f'{parse_flag};implausible_year'
        if dt<study_start or dt>study_end: return f'{parse_flag};outside_study_window'
        return parse_flag
    clean['admission_date_quality_flag']=[flag_date_quality(d,f) for d,f in zip(clean['admission_date'],clean['admission_date_parse_flag'])]; clean['ingestion_date_quality_flag']=[flag_date_quality(d,f) for d,f in zip(clean['ingestion_date'],clean['ingestion_date_parse_flag'])]
    clean['admission_date_for_plots']=clean['admission_date'].where(~clean['admission_date_quality_flag'].str.contains('outside_study_window|implausible_year',na=False)); clean['ingestion_date_for_plots']=clean['ingestion_date'].where(~clean['ingestion_date_quality_flag'].str.contains('outside_study_window|implausible_year',na=False))
    clean['admission_day_of_week']=clean['admission_date_for_plots'].dt.day_name(); clean['season']=derive_season(clean['admission_date_for_plots'])
    numeric_specs={'Age (in years)':('age_years',(0,120)),'Amount of ingestion (ml)':('amount_ingested_ml',(0,5000)),'Time to symptoms onset (in hrs)':('time_to_symptom_hrs',(0,1000)),'Time to presentation (in hrs)':('time_to_presentation_hrs',(0,1000)),'Temperature':('temperature_c',(25,45)),'Pulse (beats/min)':('pulse_bpm',(20,250)),'SpO2':('spo2',(0,100)),'GCS':('gcs',(3,15)),'Respiratory rate':('respiratory_rate',(4,80)),'Total WBC count(/mm3)':('wbc_count_mm3',(500,100000)),'S. creatinine(mg/dL)':('creatinine_mg_dl',(0,20)),'Na+':('sodium_mmol_l',(90,200)),'k+':('potassium_mmol_l',(1,10)),'Cl-':('chloride_mmol_l',(50,200)),'Random blood sugar (mmol/L)':('glucose_mmol_l',(0.5,50)),'pH':('ph',(6.5,8.0)),'HCO3-':('hco3_mmol_l',(1,60)),'Hemoglobulin':('hemoglobin_g_dl',(1,25)),'S.bilirubin(mg/dL)':('bilirubin_mg_dl',(0,50)),'SGPT':('sgpt_u_l',(0,5000)),'SGOT':('sgot_u_l',(0,5000)),'Total amount of fluid (in L)':('fluid_l',(0,100)),'Oxygen(highest L/min)':('oxygen_l_min',(0,100)),'Dialysis (No of cycle)':('dialysis_cycles',(0,50)),'S. amylase(u/L)':('amylase_u_l',(0,5000)),'PO2':('po2',(1,500)),'PCO2':('pco2',(1,200)),'Anion gap':('anion_gap',(0,100)),'Neutrophil (%)':('neutrophil_pct',(0,100)),'Lymphocytes(%)':('lymphocyte_pct',(0,100)),'Platelates(/mm3)':('platelets_mm3',(1000,2000000))}
    numq=[]
    for raw_col,(new_col,plausible) in numeric_specs.items():
        if raw_col in raw.columns:
            parsed=raw[raw_col].map(lambda x: parse_numeric(x,plausible=plausible)); clean[new_col]=parsed.map(lambda z:z[0]); clean[f'{new_col}_flag']=parsed.map(lambda z:z[1]); numq.append({'variable':new_col,'non_missing':int(clean[new_col].notna().sum()),'implausible_n':int((clean[f'{new_col}_flag']=='implausible').sum()),'median':float(clean[new_col].median()) if clean[new_col].notna().any() else np.nan})
    sbp,dbp,bpflag=zip(*raw.get('Blood pressure',pd.Series(index=raw.index)).map(split_bp)); clean['sbp']=pd.Series(sbp,index=raw.index,dtype=float); clean['dbp']=pd.Series(dbp,index=raw.index,dtype=float); clean['blood_pressure_flag']=bpflag
    binary_columns={'Vomited after ingestion':'sym_vomited_immediately','Fever':'sym_fever','Vomiting':'sym_vomiting','Diarrhoea':'sym_diarrhoea','Abdominal pain':'sym_abdominal_pain','Abdominal distension':'sym_abdominal_distension','Cough':'sym_cough','Shortness of breath':'sym_shortness_of_breath','Heart burn':'sym_heart_burn','Oral ulcers':'sym_oral_ulcers','Leg swelling':'sym_leg_swelling','Reduced urine output':'sym_reduced_urine','Jaundice':'sym_jaundice','Unconsciousness':'sym_unconsciousness','Convulsion':'sym_convulsion','Chest pain':'sym_chest_pain','Bleeding tendency':'sym_bleeding','Shock':'sym_shock','NG suction':'ng_suction','Ventilation support':'ventilation_support','Operation':'operation_support','Death':'death_flag','Absconded':'absconded_flag','Survived without complications':'survived_uncomplicated_flag'}
    for src,dst in binary_columns.items():
        if src in raw.columns: clean[dst]=raw[src].map(bool_from_series_value)
    for base in FOLLOWUP_BASE_NAMES:
        matching=[c for c in raw.columns if c==base or c.startswith(base+'__dup') or c.startswith(base)]
        if matching: clean[slugify(base)]=pd.concat([raw[c].map(bool_from_series_value) for c in matching],axis=1).max(axis=1,skipna=True)
    comp_name=categorical_clean(raw.get('Complications name')); clean['complication_name_raw']=comp_name; clean['complication_flag']=comp_name.map(lambda x:0 if safe_lower(x) in {'no','none','nil'} else (np.nan if safe_lower(x) is None else 1)); clean.loc[clean['complication_name_raw'].isna(),'complication_flag']=np.nan
    clean['dialysis_any']=np.where(clean.get('dialysis_cycles',pd.Series(index=clean.index)).fillna(0)>0,1,0); clean.loc[clean.get('dialysis_cycles',pd.Series(index=clean.index)).isna(),'dialysis_any']=0
    clean['oxygen_any']=np.where(clean.get('oxygen_l_min',pd.Series(index=clean.index)).fillna(0)>0,1,0); clean.loc[clean.get('oxygen_l_min',pd.Series(index=clean.index)).isna(),'oxygen_any']=0
    clean['low_gcs']=np.where(clean.get('gcs',pd.Series(index=clean.index)).notna(),(clean['gcs']<13).astype(int),np.nan); clean['hypoxia']=np.where(clean.get('spo2',pd.Series(index=clean.index)).notna(),(clean['spo2']<94).astype(int),np.nan); clean['hypotension']=np.where(clean.get('sbp',pd.Series(index=clean.index)).notna(),(clean['sbp']<90).astype(int),np.nan)
    clean['shock_binary']=clean.get('sym_shock',pd.Series(index=clean.index)); clean['convulsion_binary']=clean.get('sym_convulsion',pd.Series(index=clean.index)); clean['delayed_presentation']=np.where(clean.get('time_to_presentation_hrs',pd.Series(index=clean.index)).notna(),(clean['time_to_presentation_hrs']>6).astype(int),np.nan); clean['high_risk_poison_group']=clean['poison_type'].isin(['OPC','Mixed OPC','Paraquat','Aluminium Phosphide','Corrosive','Rat Killer']).astype(int)
    symptom_binary_cols=[v for v in binary_columns.values() if v.startswith('sym_') and v in clean.columns]; clean['symptom_burden_score']=clean[symptom_binary_cols].fillna(0).sum(axis=1)
    med_cols=[c for c in raw.columns if re.fullmatch(r'Name(?:__dup\d+)?',c) or c.startswith('Name')]; med_names=raw[med_cols].apply(lambda col: col.map(normalize_text_lower)) if med_cols else pd.DataFrame(index=raw.index); clean['medication_count']=med_names.notna().sum(axis=1) if not med_names.empty else 0
    clean['treatment_intensity_score']=(clean[[c for c in ['oxygen_any','ng_suction','ventilation_support','operation_support','dialysis_any'] if c in clean.columns]].apply(pd.to_numeric,errors='coerce').fillna(0).sum(axis=1)+clean['medication_count'].fillna(0).clip(upper=10))
    death_yes=clean['death_flag']==1; abs_yes=clean['absconded_flag']==1; comp_yes=clean['complication_flag']==1; surv_yes=clean['survived_uncomplicated_flag']==1
    clean['outcome_category']=np.where(death_yes,'Death',np.where(abs_yes,'Absconded/DORB',np.where(comp_yes,'Complication',np.where(surv_yes,'Survived uncomplicated','Missing/unknown'))))
    clean['severe_outcome']=(death_yes|comp_yes.fillna(False)|(clean['ventilation_support']==1)|(clean['shock_binary']==1)|(clean['low_gcs']==1)|(clean['hypoxia']==1)|(clean['dialysis_any']==1)).astype(int)
    clean['presentation_time_category']=pd.cut(clean.get('time_to_presentation_hrs',pd.Series(index=clean.index)),bins=[-np.inf,1,6,24,np.inf],labels=['≤1 h','1–6 h','6–24 h','>24 h']); clean['symptom_onset_category']=pd.cut(clean.get('time_to_symptom_hrs',pd.Series(index=clean.index)),bins=[-np.inf,1,6,24,np.inf],labels=['≤1 h','1–6 h','6–24 h','>24 h']); clean['age_group']=derive_age_group(clean.get('age_years',pd.Series(index=clean.index)))
    f_normal=clean.get('back_to_normal_health',pd.Series(index=clean.index)); f_not_normal=clean.get('not_entirely_back_to_normal_health',pd.Series(index=clean.index)); f_help=clean.get('requires_help_in_daily_activities',pd.Series(index=clean.index)); f_death=clean.get('death_mention_the_date',pd.Series(index=clean.index)); clean['followup_status']=np.where(f_death==1,'Death',np.where(f_help==1,'Requires help',np.where(f_not_normal==1,'Not entirely back to normal',np.where(f_normal==1,'Back to normal','Missing/unknown'))))
    dq=[{'variable':c,'non_missing':int(clean[c].notna().sum()),'missing':int(clean[c].isna().sum()),'missing_pct':float(clean[c].isna().mean()*100)} for c in clean.columns]; dq=pd.DataFrame(dq)
    metadata.update({'numeric_quality_summary':numq,'canonical_poison_types':canonical_types,'canonical_components':canonical_components,'n_rows':int(clean.shape[0]),'n_cols_clean':int(clean.shape[1])})
    return clean,dq,metadata

def draw_flow_diagram(ax, counts):
    ax.axis('off'); boxes=[(0.05,0.70,0.28,0.18,f"All records\nN = {counts['all']:,}"),(0.38,0.70,0.25,0.18,f"Non-missing poisoning type\nN = {counts['poison_type']:,}"),(0.68,0.70,0.25,0.18,f"Temporal plot eligible\nN = {counts['date_valid']:,}"),(0.20,0.35,0.25,0.18,f"Outcome classified\nN = {counts['outcome_known']:,}"),(0.55,0.35,0.25,0.18,f"Model complete-case\nN = {counts['model_cc']:,}")]
    for x,y,w,h,txt in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.02',facecolor='#f8fafc',edgecolor='#4e79a7',linewidth=1.2)); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10)
    for start,end in [((0.33,0.79),(0.38,0.79)),((0.63,0.79),(0.68,0.79)),((0.50,0.70),(0.32,0.53)),((0.75,0.70),(0.67,0.53))]: ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,linewidth=1.0,color='#4e79a7'))

def plot_rate_with_ci(ax, table, rate_col, low_col, high_col, label_col, color='#1d3557'):
    y=np.arange(len(table)); ax.errorbar(table[rate_col],y,xerr=[table[rate_col]-table[low_col],table[high_col]-table[rate_col]],fmt='o',color=color,ecolor='#7f8c8d',capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([wrap(v,22) for v in table[label_col]]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); despine_and_tidy(ax,'x')

def forest_plot(ax, tbl, title):
    t=tbl.copy().dropna(subset=['or']) if isinstance(tbl,pd.DataFrame) else pd.DataFrame()
    if t.empty: ax.axis('off'); ax.text(0.5,0.5,'No estimable effects',ha='center',va='center'); ax.set_title(title); return
    y=np.arange(len(t)); low=t['low'].fillna(t['or']); high=t['high'].fillna(t['or'])
    ax.errorbar(t['or'],y,xerr=[t['or']-low,high-t['or']],fmt='o',color='#1d3557',ecolor='#6c757d',capsize=3); ax.axvline(1,color='#bc4749',linestyle='--',linewidth=1)
    ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([wrap(v,26) for v in t['term_display']]); ax.set_xlabel('Odds ratio (log scale)'); ax.set_title(title); despine_and_tidy(ax,'x')

def simple_upset_like(ax, binary_df, top_n=8):
    cols=binary_df.columns.tolist(); combos=binary_df.fillna(0).astype(int).apply(lambda r: tuple([c for c,v in r.items() if v==1]),axis=1); vc=combos.value_counts(); vc=vc[vc.index.map(len)>0].head(top_n)
    if vc.empty: ax.axis('off'); ax.text(0.5,0.5,'No symptom combinations available',ha='center',va='center'); return
    nr=len(vc); mh=len(cols); ax.set_xlim(-0.5,nr-0.5); ax.set_ylim(-mh-2,vc.max()*1.25)
    for i,count in enumerate(vc.values): ax.bar(i,count,color='#4e79a7',width=0.7); ax.text(i,count+vc.max()*0.03,str(int(count)),ha='center',va='bottom',fontsize=8)
    for j,col in enumerate(cols):
        y=-(j+1); ax.text(-1.0,y,wrap(col.replace('sym_',''),18),ha='right',va='center',fontsize=8)
        for i,combo in enumerate(vc.index): ax.plot(i,y,'o',color='#1d3557' if col in combo else '#d9dde3',markersize=5)
    for i,combo in enumerate(vc.index):
        ys=[-(cols.index(c)+1) for c in combo]
        if len(ys)>=2: ax.plot([i]*len(ys),ys,color='#1d3557',linewidth=1.0)
    ax.axhline(0,color='#adb5bd',linewidth=0.8); ax.set_xticks(range(nr)); ax.set_xticklabels([f'C{i+1}' for i in range(nr)]); ax.set_ylabel('Count'); ax.set_title('Top symptom combinations'); ax.spines[['top','right','left','bottom']].set_visible(False); ax.grid(False)

def symptom_network_plot(ax, binary_df, min_edge=50):
    if nx is None: ax.axis('off'); ax.text(0.5,0.5,'networkx unavailable',ha='center',va='center'); return
    cols=binary_df.columns.tolist(); corr=binary_df.fillna(0).astype(int).T.dot(binary_df.fillna(0).astype(int)); G=nx.Graph(); [G.add_node(c) for c in cols]
    for i,c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            w=corr.loc[c1,c2]
            if w>=min_edge: G.add_edge(c1,c2,weight=w)
    if G.number_of_edges()==0: ax.axis('off'); ax.text(0.5,0.5,'No co-occurrence network after thresholding',ha='center',va='center'); return
    pos=nx.spring_layout(G,seed=SEED,k=1.0/np.sqrt(max(1,G.number_of_nodes()))); edge_widths=[0.5+G[u][v]['weight']/100 for u,v in G.edges()]
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color='#adb5bd',width=edge_widths,alpha=0.8); node_sizes=[300+25*binary_df[c].fillna(0).sum() for c in G.nodes()]
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color='#4e79a7',node_size=node_sizes,alpha=0.95); nx.draw_networkx_labels(G,pos,labels={n:wrap(n.replace('sym_',''),14) for n in G.nodes()},font_size=7,ax=ax)
    ax.set_title('Symptom co-occurrence network'); ax.axis('off')

def custom_alluvial_three_stage(ax, df, cols, top_n=6, title=''):
    d=df[list(cols)].dropna().copy();
    if d.shape[0]<20: return False
    for c in cols: top=d[c].value_counts().head(top_n).index; d[c]=d[c].where(d[c].isin(top),'Other')
    stages=[d[c].value_counts() for c in cols]; x_positions=[0.08,0.46,0.84]; stage_positions=[]
    for si,counts in enumerate(stages):
        total=counts.sum(); y0=0.05; pos={}
        for lab,cnt in counts.items(): h=0.90*(cnt/total); pos[lab]=(y0,y0+h); y0+=h+0.01
        stage_positions.append(pos)
        for lab,(y1,y2) in pos.items(): ax.add_patch(Rectangle((x_positions[si]-0.03,y1),0.06,y2-y1,facecolor='#d7e3fc',edgecolor='#4e79a7',lw=0.8,alpha=0.95)); ax.text(x_positions[si]+(0.04 if si<2 else -0.04),(y1+y2)/2,wrap(lab,18),ha='left' if si<2 else 'right',va='center',fontsize=8)
    for ls,rs in [(0,1),(1,2)]:
        lc={k:v[0] for k,v in stage_positions[ls].items()}; rc={k:v[0] for k,v in stage_positions[rs].items()}; flow=d.groupby([cols[ls],cols[rs]]).size().reset_index(name='n'); total=len(d)
        for _,row in flow.iterrows():
            l,r,n=row[cols[ls]],row[cols[rs]],row['n']; h=0.90*(n/total); y1a,y1b=lc[l],lc[l]+h; y2a,y2b=rc[r],rc[r]+h; lc[l]+=h; rc[r]+=h
            ax.add_patch(Polygon([(x_positions[ls]+0.03,y1a),(x_positions[ls]+0.03,y1b),(x_positions[rs]-0.03,y2b),(x_positions[rs]-0.03,y2a)],closed=True,facecolor='#4e79a7',alpha=0.12,edgecolor='none'))
    [ax.text(x_positions[i],0.97,wrap(cols[i].replace('_',' '),16),ha='center',va='bottom',fontweight='bold') for i in range(3)]; ax.set_title(title); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); return True

def cluster_order(df):
    if OPTIONAL_IMPORTS['scipy'] and df.shape[0]>=2 and df.shape[1]>=2:
        try: return leaves_list(linkage(df.values,method='average',metric='euclidean')).tolist(), leaves_list(linkage(df.values.T,method='average',metric='euclidean')).tolist()
        except Exception: pass
    return list(range(df.shape[0])), list(range(df.shape[1]))

def save_summary_tables(clean,dq,state):
    out=state.outdir/'02_tables'; desc=[]
    for col in ['age_years','time_to_presentation_hrs','gcs','spo2','sbp','dbp','symptom_burden_score','treatment_intensity_score']:
        if col in clean.columns and clean[col].notna().sum()>0: desc.append({'variable':col,'n':int(clean[col].notna().sum()),'median':float(clean[col].median()),'iqr_low':float(clean[col].quantile(0.25)),'iqr_high':float(clean[col].quantile(0.75))})
    pd.DataFrame(desc).to_csv(out/'descriptive_summary_continuous.csv',index=False)
    cat_frames=[]
    for col in ['study_site','sex','living_area','presentation_area','poison_type','component','outcome_category']:
        vc=clean[col].fillna('Missing').value_counts(dropna=False); cat_frames.append(pd.DataFrame({'variable':col,'category':vc.index,'n':vc.values,'pct':vc.values/len(clean)*100}))
    pd.concat(cat_frames,ignore_index=True).to_csv(out/'descriptive_summary_categorical.csv',index=False); dq.to_csv(out/'missingness_table.csv',index=False)
    strat=[]
    for outcome,sub in clean.groupby('outcome_category',dropna=False): strat.append({'outcome_category':outcome,'n':len(sub),'age_median':sub['age_years'].median(),'presentation_hrs_median':sub['time_to_presentation_hrs'].median(),'low_gcs_pct':sub['low_gcs'].mean()*100,'hypoxia_pct':sub['hypoxia'].mean()*100,'high_risk_pct':sub['high_risk_poison_group'].mean()*100})
    pd.DataFrame(strat).to_csv(out/'outcome_stratified_table.csv',index=False)

def build_model_matrix(clean, outcome_col):
    use=[c for c in ['age_years','sex','living_area','study_site','poison_type_major','delayed_presentation','low_gcs','hypoxia','hypotension','shock_binary','convulsion_binary','high_risk_poison_group'] if c in clean.columns]
    model_df=clean[use+[outcome_col]].copy();
    if 'study_site' in model_df.columns: model_df['study_site']=top_n_with_other(model_df['study_site'],8)
    if 'poison_type_major' in model_df.columns: model_df['poison_type_major']=top_n_with_other(model_df['poison_type_major'],8)
    return model_df.drop(columns=[outcome_col]), model_df[outcome_col].astype(float), use, {'initial_predictors':use}

def univariable_or_table(clean,outcome_col,logger):
    expected_cols=['term','term_display','or','low','high','pvalue','n']
    rows=[]
    preds={'Male sex':(clean['sex']=='Male').astype(float),'Rural residence':(clean['living_area']=='Rural').astype(float),'Delayed presentation >6 h':clean.get('delayed_presentation',pd.Series(index=clean.index,dtype=float)),'Low GCS <13':clean.get('low_gcs',pd.Series(index=clean.index,dtype=float)),'Hypoxia (SpO2 <94%)':clean.get('hypoxia',pd.Series(index=clean.index,dtype=float)),'Hypotension (SBP <90)':clean.get('hypotension',pd.Series(index=clean.index,dtype=float)),'Shock':clean.get('shock_binary',pd.Series(index=clean.index,dtype=float)),'Convulsion':clean.get('convulsion_binary',pd.Series(index=clean.index,dtype=float)),'High-risk poison group':clean.get('high_risk_poison_group',pd.Series(index=clean.index,dtype=float))}
    for name,series in preds.items():
        tmp=pd.DataFrame({'x':pd.to_numeric(series,errors='coerce'),'y':pd.to_numeric(clean[outcome_col],errors='coerce')}).dropna()
        tmp=tmp[tmp['y'].isin([0,1])]
        if tmp['x'].nunique()<2 or tmp['y'].sum()<5 or tmp['y'].nunique()<2: continue
        try:
            # 2x2 Wald approximation with Haldane-Anscombe correction: robust and fast for registry screening.
            a=float(((tmp['x']==1)&(tmp['y']==1)).sum())+0.5
            b=float(((tmp['x']==1)&(tmp['y']==0)).sum())+0.5
            c=float(((tmp['x']==0)&(tmp['y']==1)).sum())+0.5
            d=float(((tmp['x']==0)&(tmp['y']==0)).sum())+0.5
            log_or=math.log((a*d)/(b*c)); se=math.sqrt(1/a+1/b+1/c+1/d)
            rows.append({'term':name,'term_display':name,'or':math.exp(log_or),'low':math.exp(log_or-1.96*se),'high':math.exp(log_or+1.96*se),'pvalue':np.nan,'n':int(tmp.shape[0])})
        except Exception as e:
            logger.warning(f'Univariable OR failed for {name}: {e}')
    return pd.DataFrame(rows,columns=expected_cols)

def fit_adjusted_logistic(clean,outcome_col,impute,logger):
    X_raw,y,predictors,meta=build_model_matrix(clean,outcome_col)
    base_df=pd.concat([X_raw,y.rename(outcome_col)],axis=1)
    base_df[outcome_col]=pd.to_numeric(base_df[outcome_col],errors='coerce')
    base_df=base_df[base_df[outcome_col].isin([0,1])].copy()
    event_n=int(base_df[outcome_col].sum()) if not base_df.empty else 0
    meta['event_n']=event_n
    if base_df.empty or event_n<10 or base_df[outcome_col].nunique()<2:
        meta['status']='insufficient_events'
        return pd.DataFrame(),meta,None,None,None
    if impute:
        model_df=base_df.copy()
    else:
        model_df=base_df.dropna().copy()
    if model_df.empty or model_df[outcome_col].nunique()<2:
        meta['status']='insufficient_complete_rows'
        return pd.DataFrame(),meta,None,None,None
    y_model=model_df[outcome_col].astype(float)
    X_model=model_df.drop(columns=[outcome_col])
    # Strict numeric/categorical split.
    numeric=[]; categorical=[]
    for col in X_model.columns:
        observed=X_model[col].notna().sum()
        coerced=pd.to_numeric(X_model[col],errors='coerce')
        if observed>0 and coerced.notna().sum()==observed:
            numeric.append(col)
        else:
            categorical.append(col)
    # Keep model stable and fast.
    max_terms=min(10,max(2,event_n//12))
    keep=[p for p in ['age_years','sex','poison_type_major','delayed_presentation','low_gcs','hypoxia','hypotension','shock_binary','high_risk_poison_group'] if p in X_model.columns][:max_terms]
    X_model=X_model[keep]
    numeric=[c for c in numeric if c in X_model.columns]
    categorical=[c for c in categorical if c in X_model.columns]
    if impute and OPTIONAL_IMPORTS['sklearn']:
        Xi=X_model.copy()
        for c in categorical: Xi[c]=Xi[c].map(normalize_text).fillna('Missing').astype(str)
        if numeric:
            Xi[numeric]=Xi[numeric].apply(pd.to_numeric,errors='coerce')
            Xi[numeric]=SimpleImputer(strategy='median').fit_transform(Xi[numeric])
        X_model=Xi
    else:
        X_model=X_model.dropna().copy()
        y_model=y_model.loc[X_model.index]
        for c in categorical: X_model[c]=X_model[c].map(normalize_text).fillna('Missing').astype(str)
        if numeric: X_model[numeric]=X_model[numeric].apply(pd.to_numeric,errors='coerce')
    X_design=pd.get_dummies(X_model,drop_first=True,dtype=float)
    X_design=X_design.replace([np.inf,-np.inf],np.nan).dropna(axis=1,how='any')
    y_model=pd.to_numeric(y_model,errors='coerce')
    valid_y=y_model.isin([0,1])
    X_design=X_design.loc[valid_y]
    y_model=y_model.loc[valid_y].astype(float)
    sparse_ok=[c for c in X_design.columns if X_design[c].sum()>=10 or (X_design[c]==0).sum()>=10]
    X_design=X_design[sparse_ok]
    meta['design_terms']=X_design.columns.tolist(); meta['status']='prepared_penalized'
    if X_design.empty or y_model.nunique()<2:
        meta['status']='empty_design_or_outcome'
        return pd.DataFrame(),meta,None,None,None
    try:
        if OPTIONAL_IMPORTS['sklearn']:
            clf=LogisticRegression(max_iter=500,penalty='l2',solver='liblinear')
            clf.fit(X_design,y_model)
            rows=[{'term':term,'term_display':term.replace('_',' '),'or':math.exp(float(c)),'low':np.nan,'high':np.nan,'pvalue':np.nan} for term,c in zip(X_design.columns,clf.coef_[0])]
            meta['status']='fitted_sklearn_penalized'
            return pd.DataFrame(rows),meta,clf,X_design,y_model
    except Exception as e:
        meta['sklearn_failure']=str(e)
        logger.warning(f'Penalized logistic failed for {outcome_col}: {e}')
    meta['status']='failed'
    return pd.DataFrame(),meta,None,None,None

def cross_validated_roc_and_calibration(model_X, model_y, logger):
    if not OPTIONAL_IMPORTS['sklearn'] or model_X is None or model_y is None or model_y.sum()<30 or model_y.nunique()<2 or model_X.shape[0]<200: return None,None
    try:
        clf=LogisticRegression(max_iter=500,solver='liblinear'); cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); probs=cross_val_predict(clf,model_X,model_y,cv=cv,method='predict_proba')[:,1]; auc=roc_auc_score(model_y,probs); fpr,tpr,_=roc_curve(model_y,probs); frac_pos,mean_pred=calibration_curve(model_y,probs,n_bins=10)
        return {'auc':auc,'diag':pd.DataFrame({'fpr':fpr,'tpr':tpr}),'calib':pd.DataFrame({'mean_pred':mean_pred,'frac_pos':frac_pos}),'probs':probs}, pd.DataFrame({'mean_pred':mean_pred,'frac_pos':frac_pos})
    except Exception as e: logger.warning(f'Cross-validated ROC/calibration failed: {e}'); return None,None


# -----------------------------------------------------------------------------
# V3 SINGLE-FIGURE-ONLY OUTPUT ENGINE
# -----------------------------------------------------------------------------
PLANNED_ANALYSES = [
    ('cohort_flow','main'),('study_site_enrollment','main'),('age_sex_pyramid','main'),
    ('demographic_composition','main'),('key_variable_completeness','main'),
    ('ranked_poisoning_types','main'),('rare_poisoning_categories','main'),
    ('poisoning_type_by_age_group','main'),('poisoning_type_by_sex','main'),
    ('poisoning_type_by_study_site','main'),('top_components','main'),('type_component_heatmap','main'),
    ('monthly_admission_trend','main'),('monthly_poisoning_type_trend','main'),('calendar_heatmap','main'),
    ('day_of_week_pattern','main'),('seasonal_pattern','main'),('site_temporal_trend','main'),
    ('symptom_prevalence','main'),('type_symptom_heatmap','main'),('symptom_combinations','main'),
    ('symptom_network','main'),('severity_by_type','main'),('low_gcs_by_type','main'),
    ('hypoxia_by_type','main'),('hypotension_by_type','main'),('symptom_burden_by_outcome','main'),
    ('treatment_heatmap','main'),('intervention_rates','main'),('outcome_distribution','main'),
    ('death_rate_by_type','main'),('severe_rate_by_type','main'),('treatment_intensity_by_outcome','main'),
    ('followup_summary','main'),('followup_by_type','main'),
    ('univariable_death_or','main'),('adjusted_death_or','main'),('univariable_severe_or','main'),
    ('adjusted_severe_or','main'),('model_sensitivity','main'),('roc_curve','main'),('calibration_plot','main'),
    ('missingness_matrix','supplementary'),('date_quality_flags','supplementary'),('numeric_quality_flags','supplementary'),
    ('highest_missingness','supplementary'),('harmonization_poison','supplementary'),('harmonization_component','supplementary'),
    ('context_occupation','supplementary'),('context_living_area','supplementary'),('context_presentation_area','supplementary'),
    ('lab_completeness','supplementary'),('lab_abnormality','supplementary'),('vital_lab_correlation','supplementary'),
    ('alluvial_site_type_outcome','supplementary'),('alluvial_age_type_outcome','supplementary'),
    ('alluvial_type_treatment_outcome','supplementary'),('followup_pathway','supplementary'),
    ('delay_vs_severity','supplementary'),('amount_vs_outcome','supplementary'),('site_heterogeneity','supplementary'),
    ('sparse_category_sensitivity','supplementary'),('implausible_numeric_sensitivity','supplementary'),
    ('pca_labs','supplementary'),('component_outcome_heatmap','exploratory'),('temporal_qc','exploratory'),
    ('predictor_importance','exploratory')
]

SECTION_DIRS = {
    'main':'03_main_figures_single',
    'supplementary':'04_supplementary_figures_single',
    'exploratory':'05_exploratory_figures_single'
}

FORBIDDEN_TITLE_PREFIX = re.compile(r'^(figure\s*\d+\.?|supplementary\s+figure\s*S?\d+\.?|panel\s+[A-Z]\.?|[A-Z]\.)\s*', re.I)

def clean_figure_title(title:str)->str:
    s=normalize_space(str(title))
    # Repeatedly remove manuscript numbering prefixes if any slipped in.
    old=None
    while old != s:
        old=s; s=FORBIDDEN_TITLE_PREFIX.sub('',s).strip()
    return s

def wrap_title(title:str, width:int=72)->str:
    return '\n'.join(textwrap.wrap(clean_figure_title(title), width=width))

def section_dir(section:str)->str:
    return SECTION_DIRS.get(section,'05_exploratory_figures_single')

def initialise_output_tree(outdir:Path, overwrite:bool):
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory already exists and is not empty: {outdir}')
    ensure_dir(outdir)
    for d in OUTPUT_SUBDIRS: ensure_dir(outdir/d)

def init_logging(outdir:Path):
    logger=logging.getLogger('poison_v3'); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh=logging.FileHandler(outdir/'00_logs'/'run.log',mode='w',encoding='utf-8'); fh.setFormatter(fmt)
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def gradient_pair(state:AnalysisState):
    name=state.choose_gradient(); cols=PREMIUM_GRADIENTS.get(name,['#1d3557','#4e79a7','#e9f5f2'])
    return name, cols[0], cols[1], cols[2]

def dynamic_size(n_items:int=10, max_label_len:int=12, kind:str='barh', wide:bool=False):
    n=max(1,int(n_items)); max_label_len=max(1,int(max_label_len))
    if kind in {'barh','forest','rate'}:
        w=min(16, max(7.5, 7.0 + 0.10*max_label_len))
        h=min(20, max(5.2, 2.8 + 0.36*n))
    elif kind=='heatmap':
        w=min(18, max(8.0, 4.8 + 0.55*n + 0.05*max_label_len))
        h=min(18, max(6.0, 4.2 + 0.38*n))
    elif kind=='line':
        w=12 if not wide else 15
        h=6.6 if not wide else 7.8
    elif kind=='alluvial':
        w=14; h=8.5
    elif kind=='matrix':
        w=14; h=max(6.5, min(12, 4+0.35*n))
    else:
        w=9.5 if not wide else 13
        h=max(5.5, min(12, 3.6+0.28*n))
    return (float(w), float(h))

def fit_wrapped_labels(labels, width=22, max_chars=64):
    out=[]
    for lab in labels:
        s=str(lab if lab is not None else 'Missing')
        if len(s)>max_chars: s=s[:max_chars-1]+'…'
        out.append('\n'.join(textwrap.wrap(s,width=width)) if len(s)>width else s)
    return out

def shorten_labels(labels, max_chars=42):
    out=[]
    for lab in labels:
        s=str(lab if lab is not None else 'Missing')
        out.append(s if len(s)<=max_chars else s[:max_chars-1]+'…')
    return out

def prepare_single_axis(title, n_items=10, max_label_len=12, kind='barh', wide=False):
    fig,ax=plt.subplots(figsize=dynamic_size(n_items,max_label_len,kind,wide))
    ax.set_title(wrap_title(title), loc='left', pad=14, fontsize=14, fontweight='bold')
    return fig,ax

def collect_text_bboxes(fig):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[]
    for ax in fig.axes:
        texts=[]
        texts.extend([t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()])
        texts.extend([t for t in ax.get_yticklabels() if t.get_visible() and t.get_text()])
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        leg=ax.get_legend()
        if leg is not None:
            texts.extend([t for t in leg.get_texts() if t.get_visible() and t.get_text()])
            if leg.get_title() is not None: texts.append(leg.get_title())
        for t in texts:
            if not t.get_visible() or not t.get_text(): continue
            try:
                bb=t.get_window_extent(renderer=renderer).expanded(1.01,1.08)
                if bb.width>1 and bb.height>1: boxes.append((bb,t.get_text()))
            except Exception:
                pass
    return boxes

def detect_text_overlaps(fig, max_pairs=10):
    boxes=collect_text_bboxes(fig)
    hits=[]
    for i in range(len(boxes)):
        b1,t1=boxes[i]
        for j in range(i+1,len(boxes)):
            b2,t2=boxes[j]
            if b1.overlaps(b2):
                hits.append((t1,t2))
                if len(hits)>=max_pairs: return hits
    return hits

def finalize_single_figure(fig, logger=None):
    overlap_hits=[]
    for attempt in range(4):
        try:
            fig.tight_layout(pad=2.0)
        except Exception:
            pass
        overlap_hits=detect_text_overlaps(fig)
        if not overlap_hits:
            return 'no_detected_text_overlap', attempt
        # Increase canvas and margins; retry. This is intentionally conservative.
        w,h=fig.get_size_inches()
        fig.set_size_inches(min(w*1.16+0.4, 24), min(h*1.14+0.3, 24), forward=True)
        try:
            fig.subplots_adjust(left=0.22,right=0.94,top=0.88,bottom=0.16)
        except Exception:
            pass
    if logger:
        logger.warning('Possible text overlap remained after retries: %s', overlap_hits[:3])
    return 'possible_text_overlap_after_retries', 4

def sanitize_figure_face(fig):
    for ax in fig.axes:
        ax.set_title(clean_figure_title(ax.get_title()), loc='left')
        # Explicitly remove any accidental panel letters or manuscript numbering from text objects.
        for t in ax.texts:
            val=normalize_space(t.get_text())
            if re.fullmatch(r'[A-Z]', val) or re.match(r'^(Figure|Supplementary Figure|Panel)\b', val, re.I):
                t.set_text('')


def write_caption(path:Path,title:str,caption:str):
    path.write_text(f'{clean_figure_title(title)}\n\n{caption}\n',encoding='utf-8')

def heuristic_score(title,n_categories=0,is_model=False,is_main=False,qa_notes=''):
    leg=10 if n_categories<=10 else (9 if n_categories<=18 else 8)
    vis=10 if 'possible_text_overlap' not in qa_notes else 7
    sci=9 if not qa_notes.lower().startswith('skipped') else 6
    stat=9 if is_model else 8
    ms=10 if is_main else 8
    return {'scientific_validity':sci,'legibility':leg,'manuscript_suitability':ms,'statistical_appropriateness':stat,'visual_quality':vis}

def save_single_figure(fig,state,fig_id,title,section,caption,variables,analysis_type,n,gradient=None,qa_notes='',is_model=False,is_main=False,dpi=600,logger=None):
    title=clean_figure_title(title)
    sanitize_figure_face(fig)
    overlap_status,retries=finalize_single_figure(fig,logger)
    if qa_notes:
        qa_notes=f'{qa_notes}; layout={overlap_status}; retries={retries}'
    else:
        qa_notes=f'layout={overlap_status}; retries={retries}'
    subdir=section_dir(section)
    base=state.outdir/subdir/fig_id
    png=str(base.with_suffix('.png')); pdf=str(base.with_suffix('.pdf')); svg=str(base.with_suffix('.svg'))
    fig.savefig(png,dpi=dpi,facecolor='white',bbox_inches='tight')
    fig.savefig(pdf,facecolor='white',bbox_inches='tight')
    fig.savefig(svg,facecolor='white',bbox_inches='tight')
    cap_path=base.with_suffix('.txt'); write_caption(cap_path,title,caption)
    if gradient is None: gradient=state.choose_gradient()
    scores=heuristic_score(title,n_categories=n if isinstance(n,int) else 0,is_model=is_model,is_main=is_main,qa_notes=qa_notes)
    rec=FigureRecord(figure_id=fig_id,title=title,section=section,base_filename=str(base.name),variables='; '.join(map(str,variables)),analysis_type=analysis_type,gradient=gradient,n=n,caption_file=str(cap_path.relative_to(state.outdir)),png_file=str(Path(png).relative_to(state.outdir)),pdf_file=str(Path(pdf).relative_to(state.outdir)),svg_file=str(Path(svg).relative_to(state.outdir)),recommended_tier=section,qa_notes=qa_notes,**scores)
    state.figure_registry.append(rec)
    if logger: logger.info('Saved single figure: %s', fig_id)
    plt.close(fig); return rec

def write_registry_and_scorecards(state):
    for row in state.coverage_rows:
        if row.get('status')=='planned':
            row['status']='skipped'; row['reason']='Not triggered by available data or not statistically appropriate'
            state.skip_rows.append({'analysis_id':row['analysis_id'],'reason':row['reason']})
    reg_df=pd.DataFrame([asdict(r) for r in state.figure_registry])
    vc=state.outdir/'08_version_control'; qc=state.outdir/'07_quality_control'
    reg_df.to_csv(vc/'figure_registry.csv',index=False)
    reg_df.to_json(vc/'figure_registry.json',orient='records',indent=2)
    cols=['figure_id','title','section','scientific_validity','legibility','manuscript_suitability','statistical_appropriateness','visual_quality','recommended_tier','qa_notes']
    reg_df[cols].to_csv(vc/'figure_scorecard.csv',index=False)
    pd.DataFrame(state.coverage_rows).to_csv(vc/'coverage_matrix.csv',index=False)
    pd.DataFrame(state.skip_rows).drop_duplicates().to_csv(qc/'skipped_analyses.csv',index=False)

def make_gallery(state):
    rows=[]
    for rec in state.figure_registry:
        rows.append(f"<div class='card'><h3>{rec.title}</h3><p><strong>Tag:</strong> {rec.section} | <strong>Type:</strong> {rec.analysis_type} | <strong>N:</strong> {rec.n}</p><a href='../{rec.png_file}'><img src='../{rec.png_file}' alt='{rec.title}'></a><p>{rec.qa_notes}</p><p><a href='../{rec.png_file}'>PNG</a> | <a href='../{rec.pdf_file}'>PDF</a> | <a href='../{rec.svg_file}'>SVG</a> | <a href='../{rec.caption_file}'>Caption</a></p></div>")
    html=f"<html><head><meta charset='utf-8'><title>Poison single-figure gallery</title><style>body{{font-family:Arial,sans-serif;margin:22px;background:#fafbfc;}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px;}}.card{{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 7px rgba(0,0,0,.08);}}img{{width:100%;border:1px solid #e5e7eb;}}h1,h3{{color:#1d3557;}}</style></head><body><h1>Poison single-figure gallery</h1><div class='grid'>{''.join(rows)}</div></body></html>"
    (state.outdir/'06_interactive_html'/'figure_gallery.html').write_text(html,encoding='utf-8')

def save_version_control(input_path:Path, script_path:Path, outdir:Path, run_id:str, metadata:Dict[str,Any], dpi:int=600):
    vc=outdir/'08_version_control'
    data={'run_id':run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'input_file':str(input_path),'input_sha256':sha256_of_file(input_path),'script_file':str(script_path),'script_sha256':sha256_of_file(script_path),'python_version':sys.version,'platform':platform.platform(),'package_versions':{'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':mpl.__version__,**{k:str(v) for k,v in OPTIONAL_IMPORTS.items()}},'analysis_configuration':{'version':'v3_single_figures_only','seed':SEED,'dpi':dpi,'main_sheet':MAIN_SHEET,'study_window':metadata.get('study_window',{}),'pii_patterns':PII_PATTERNS,'single_figure_only':True,'no_panel_letters':True,'no_visible_manuscript_numbering':True},'metadata':metadata}
    (vc/'run_metadata.json').write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
    if yaml is not None: (vc/'analysis_configuration.yaml').write_text(yaml.safe_dump(data['analysis_configuration'],sort_keys=False),encoding='utf-8')
    else: (vc/'analysis_configuration.yaml').write_text('# pyyaml unavailable; JSON configuration written in run_metadata.json\n',encoding='utf-8')

def qa_posthoc(state,clean):
    rows=[]
    forbidden=re.compile(r'^(Figure\s*\d+|Supplementary\s+Figure|Panel\s+[A-Z]|[A-Z]\.\s)',re.I)
    for rec in state.figure_registry:
        notes=[]
        if forbidden.search(rec.title): notes.append('forbidden visible numbering in title')
        if 'possible_text_overlap' in rec.qa_notes: notes.append('review possible text overlap')
        if rec.n is None or (isinstance(rec.n,int) and rec.n<=0): notes.append('missing/zero n annotation')
        rows.append({'figure_id':rec.figure_id,'title':rec.title,'qa_status':'pass' if not notes else 'review','notes':' | '.join(notes)})
    pd.DataFrame(rows).to_csv(state.outdir/'07_quality_control'/'visual_qa_results.csv',index=False)

def save_clean_outputs(clean,raw_header_map,dq,outdir:Path):
    safe=clean.copy()
    safe.to_csv(outdir/'01_clean_data'/'clean_analysis_dataset.csv',index=False)
    dq.to_csv(outdir/'01_clean_data'/'data_quality_summary.csv',index=False)
    pd.DataFrame(raw_header_map).to_csv(outdir/'01_clean_data'/'header_mapping.csv',index=False)
    try:
        if OPTIONAL_IMPORTS['pyarrow']: safe.to_parquet(outdir/'01_clean_data'/'clean_analysis_dataset.parquet',index=False)
    except Exception: pass


def despine_and_tidy(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_axis in {'x','y','both'}:
        ax.grid(True,axis=grid_axis,color='#d9dde3',linewidth=0.5,alpha=0.75)
    else:
        ax.grid(False)
# -----------------------------------------------------------------------------
# Single-figure plotting primitives
# -----------------------------------------------------------------------------
def safe_series_counts(s, top_n=20, include_missing=True):
    vals=s.fillna('Missing') if include_missing else s.dropna()
    vc=vals.value_counts()
    if len(vc)>top_n:
        top=vc.head(top_n-1)
        other=vc.iloc[top_n-1:].sum()
        vc=pd.concat([top,pd.Series({'Other':other})])
    return vc

def save_barh_series(series, state, fig_id, title, section, caption, variables, xlabel='Patients', percent=False, max_items=24, color=None, dpi=600, logger=None):
    vc=series.dropna()
    if vc.empty:
        state.mark_coverage(fig_id,'skipped',reason='No data')
        return None
    if len(vc)>max_items:
        vc=vc.sort_values(ascending=False).head(max_items)
    vc=vc.sort_values(ascending=True)
    max_len=max([len(str(x)) for x in vc.index] or [8])
    fig,ax=prepare_single_axis(title,len(vc),max_len,'barh')
    if color is None:
        _,dark,mid,_=gradient_pair(state); color=mid
    ax.barh(np.arange(len(vc)),vc.values,color=color,edgecolor='white',linewidth=0.5)
    ax.set_yticks(np.arange(len(vc))); ax.set_yticklabels(fit_wrapped_labels(vc.index,width=24,max_chars=70))
    ax.set_xlabel(xlabel)
    if percent: ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%'))
    despine_and_tidy(ax,'x')
    return save_single_figure(fig,state,fig_id,title,section,caption,variables,'horizontal bar chart',int(vc.sum()) if not percent else len(vc),is_main=(section=='main'),dpi=dpi,logger=logger)

def save_vertical_series(series, state, fig_id, title, section, caption, variables, ylabel='Patients', percent=False, max_items=14, color=None, dpi=600, logger=None):
    vc=series.dropna()
    if vc.empty: state.mark_coverage(fig_id,'skipped',reason='No data'); return None
    if len(vc)>max_items: vc=vc.sort_values(ascending=False).head(max_items)
    max_len=max([len(str(x)) for x in vc.index] or [8])
    fig,ax=prepare_single_axis(title,len(vc),max_len,'bar')
    if color is None:
        _,dark,mid,_=gradient_pair(state); color=mid
    ax.bar(np.arange(len(vc)),vc.values,color=color,edgecolor='white',linewidth=0.5)
    ax.set_xticks(np.arange(len(vc))); ax.set_xticklabels(fit_wrapped_labels(vc.index,width=16,max_chars=48),rotation=0,ha='center')
    ax.set_ylabel(ylabel)
    if percent: ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%'))
    despine_and_tidy(ax,'y')
    return save_single_figure(fig,state,fig_id,title,section,caption,variables,'vertical bar chart',int(vc.sum()) if not percent else len(vc),is_main=(section=='main'),dpi=dpi,logger=logger)

def save_stacked_percent_table(tab, state, fig_id, title, section, caption, variables, dpi=600, logger=None, legend_title=''):
    if tab is None or tab.empty:
        state.mark_coverage(fig_id,'skipped',reason='No crosstab data'); return None
    tab=tab.copy().fillna(0)
    # Keep the plot readable: restrict columns and rows with Other already represented where appropriate.
    if tab.shape[1]>8: tab=tab[tab.sum().sort_values(ascending=False).index[:8]]
    if tab.shape[0]>14: tab=tab.loc[tab.sum(axis=1).sort_values(ascending=False).index[:14]]
    max_len=max([len(str(x)) for x in tab.index] + [len(str(x)) for x in tab.columns])
    fig,ax=prepare_single_axis(title,tab.shape[0],max_len,'bar',wide=True)
    bottom=np.zeros(tab.shape[0])
    colors=CATEGORICAL_BASE[:tab.shape[1]]
    for i,col in enumerate(tab.columns):
        vals=tab[col].values
        ax.bar(np.arange(tab.shape[0]),vals,bottom=bottom,color=colors[i%len(colors)],edgecolor='white',linewidth=0.4,label=str(col))
        bottom += vals
    ax.set_xticks(np.arange(tab.shape[0])); ax.set_xticklabels(fit_wrapped_labels(tab.index,width=14,max_chars=46),rotation=0,ha='center')
    ax.set_ylabel('Within-group proportion')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%'))
    leg=ax.legend(title=legend_title if legend_title else None,frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    if leg is not None: leg._legend_box.align='left'
    despine_and_tidy(ax,'y')
    return save_single_figure(fig,state,fig_id,title,section,caption,variables,'stacked percentage bar chart',int(tab.shape[0]),is_main=(section=='main'),dpi=dpi,logger=logger)

def save_heatmap_table(tbl, state, fig_id, title, section, caption, variables, percent=False, cmap_color='#1d3557', dpi=600, logger=None, max_rows=16, max_cols=14):
    if tbl is None or tbl.empty:
        state.mark_coverage(fig_id,'skipped',reason='No heatmap data'); return None
    data=tbl.copy().fillna(0)
    # Split/limit dense heatmaps; continuation figures can be generated by caller if needed.
    if data.shape[0]>max_rows: data=data.loc[data.sum(axis=1).sort_values(ascending=False).index[:max_rows]]
    if data.shape[1]>max_cols: data=data.loc[:, data.sum(axis=0).sort_values(ascending=False).index[:max_cols]]
    max_len=max([len(str(x)) for x in list(data.index)+list(data.columns)] or [8])
    fig,ax=prepare_single_axis(title,max(data.shape),max_len,'heatmap',wide=True)
    cmap=sns.light_palette(cmap_color,as_cmap=True) if sns is not None else 'Blues'
    if sns is not None:
        cbar_fmt=FuncFormatter(lambda x,_:f'{100*x:.0f}%') if percent else None
        sns.heatmap(data,ax=ax,cmap=cmap,cbar=True,cbar_kws={'format':cbar_fmt} if cbar_fmt else None)
    else:
        im=ax.imshow(data.values,aspect='auto',cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set_xticklabels(fit_wrapped_labels(data.columns,width=12,max_chars=36),rotation=45,ha='right')
    ax.set_yticklabels(fit_wrapped_labels(data.index,width=18,max_chars=52),rotation=0)
    ax.set_xlabel(''); ax.set_ylabel('')
    return save_single_figure(fig,state,fig_id,title,section,caption,variables,'heatmap',int(data.size),is_main=(section=='main'),dpi=dpi,logger=logger)

def rate_table(clean, group_col, outcome_col, min_n=30):
    rows=[]
    for grp,sub in clean.dropna(subset=[group_col]).groupby(group_col):
        n=int(sub[outcome_col].notna().sum()) if outcome_col in sub.columns else len(sub)
        if n<min_n: continue
        k=int((sub[outcome_col]==1).sum())
        ph,lo,hi=wilson_ci(k,n)
        rows.append({'group':grp,'n':n,'k':k,'rate':ph,'low':lo,'high':hi})
    return pd.DataFrame(rows).sort_values('rate') if rows else pd.DataFrame(columns=['group','n','k','rate','low','high'])

def save_rate_ci(table, state, fig_id, title, section, caption, variables, dpi=600, logger=None, color='#1d3557'):
    if table is None or table.empty:
        state.mark_coverage(fig_id,'skipped',reason='No estimable rates'); return None
    t=table.copy().sort_values('rate')
    max_len=max([len(str(x)) for x in t['group']] or [8])
    fig,ax=prepare_single_axis(title,len(t),max_len,'rate')
    y=np.arange(len(t)); x=t['rate'].to_numpy(float)
    left=np.maximum(0,x-t['low'].to_numpy(float)); right=np.maximum(0,t['high'].to_numpy(float)-x)
    ax.errorbar(x,y,xerr=[left,right],fmt='o',color=color,ecolor='#7f8c8d',capsize=3,markersize=5)
    ax.set_yticks(y); ax.set_yticklabels(fit_wrapped_labels(t['group'],width=23,max_chars=70))
    ax.set_xlabel('Rate with Wilson 95% CI')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%'))
    despine_and_tidy(ax,'x')
    return save_single_figure(fig,state,fig_id,title,section,caption,variables,'Wilson CI rate plot',int(t['n'].sum()),is_main=(section=='main'),dpi=dpi,logger=logger)

def save_forest(table, state, fig_id, title, section, caption, variables, dpi=600, logger=None):
    t=table if isinstance(table,pd.DataFrame) else pd.DataFrame()
    if t.empty or 'or' not in t.columns:
        fig,ax=prepare_single_axis(title,1,20,'forest')
        ax.axis('off'); ax.text(0.5,0.5,'No estimable effects',ha='center',va='center')
    else:
        max_len=max([len(str(x)) for x in t.get('term_display',t.get('term',pd.Series(['Effect'])))] or [20])
        fig,ax=prepare_single_axis(title,min(len(t),18),max_len,'forest')
        forest_plot(ax,t.head(18),title)
        ax.set_title(wrap_title(title),loc='left',pad=14,fontsize=14,fontweight='bold')
    return save_single_figure(fig,state,fig_id,title,section,caption,variables,'forest plot',int(len(t)),is_main=(section=='main'),is_model=True,dpi=dpi,logger=logger)

# -----------------------------------------------------------------------------
# V3 analysis/figure generation
# -----------------------------------------------------------------------------
def generate_single_main_figures(clean,state,logger,dpi=600):
    results={}
    plot_df=clean.copy()
    major_poison=clean['poison_type'].value_counts().head(10).index.tolist()
    plot_df['poison_type_plot']=plot_df['poison_type'].where(plot_df['poison_type'].isin(major_poison),'Other')

    # Cohort and demographics
    counts={'all':len(clean),'poison_type':int(clean['poison_type'].notna().sum()),'date_valid':int(clean['admission_date_for_plots'].notna().sum()),'outcome_known':int((clean['outcome_category']!='Missing/unknown').sum()),'model_cc':int(clean[[c for c in ['death_flag','age_years','sex','poison_type_major','low_gcs','hypoxia','hypotension'] if c in clean.columns]].dropna().shape[0])}
    fig,ax=prepare_single_axis('Cohort profile and analysis flow',5,28,'alluvial')
    draw_flow_diagram(ax,counts)
    save_single_figure(fig,state,'main_cohort_profile_analysis_flow','Cohort profile and analysis flow','main','Flow diagram showing records available for descriptive, temporal, outcome, and modeling analyses. PII variables are excluded from analytical outputs.', ['study_site','poison_type','admission_date','outcome_category'], 'cohort flow diagram', len(clean), is_main=True, dpi=dpi, logger=logger)
    state.mark_coverage('cohort_flow','generated','main_cohort_profile_analysis_flow')

    vc=safe_series_counts(clean['study_site'],top_n=20)
    save_barh_series(vc,state,'main_study_site_enrollment','Study-site enrollment','main','Enrollment by study site, ranked by number of records.', ['study_site'], dpi=dpi, logger=logger)
    state.mark_coverage('study_site_enrollment','generated','main_study_site_enrollment')

    pyr=clean.dropna(subset=['age_group','sex']).copy()
    if not pyr.empty:
        pyramid=pyr.groupby(['age_group','sex']).size().unstack(fill_value=0).reindex(index=clean['age_group'].cat.categories)
        fig,ax=prepare_single_axis('Age–sex pyramid',len(pyramid),10,'barh')
        males=-pyramid.get('Male',pd.Series(index=pyramid.index,data=0)); females=pyramid.get('Female',pd.Series(index=pyramid.index,data=0)); y=np.arange(len(pyramid.index))
        ax.barh(y,males,color='#457b9d',label='Male'); ax.barh(y,females,color='#e76f51',label='Female')
        ax.set_yticks(y); ax.set_yticklabels(pyramid.index.astype(str)); ax.set_xlabel('Patients')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{abs(int(v))}'))
        ax.legend(frameon=False,loc='lower right'); despine_and_tidy(ax,'x')
        save_single_figure(fig,state,'main_age_sex_pyramid','Age–sex pyramid','main','Age distribution by sex using mirrored horizontal bars.', ['age_group','sex'], 'age-sex pyramid', len(pyr), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('age_sex_pyramid','generated','main_age_sex_pyramid')
    else: state.mark_coverage('age_sex_pyramid','skipped',reason='Insufficient age/sex data')

    comp=pd.DataFrame({'Sex':clean['sex'].fillna('Missing').value_counts(normalize=True),'Residence':clean['living_area'].fillna('Missing').value_counts(normalize=True),'Presentation':clean['presentation_area'].fillna('Missing').value_counts(normalize=True)}).fillna(0)
    save_stacked_percent_table(comp.T,state,'main_demographic_composition','Demographic composition','main','Stacked proportions for sex, living area, and presentation area.', ['sex','living_area','presentation_area'], dpi=dpi, logger=logger, legend_title='Category')
    state.mark_coverage('demographic_composition','generated','main_demographic_composition')

    lab_candidates=[c for c in ['wbc_count_mm3','creatinine_mg_dl','ph','hco3_mmol_l'] if c in clean.columns]
    completeness=pd.Series({'Outcome':clean['outcome_category'].replace('Missing/unknown',np.nan).notna().mean(),'Admission date':clean['admission_date_for_plots'].notna().mean(),'Poison type':clean['poison_type'].notna().mean(),'Component':clean['component'].notna().mean(),'Vitals':clean[[c for c in ['gcs','spo2','sbp'] if c in clean.columns]].notna().mean().mean(),'Labs':clean[lab_candidates].notna().mean().mean() if lab_candidates else np.nan}).dropna().sort_values()
    save_barh_series(completeness,state,'main_key_variable_completeness','Key-variable completeness','main','Completeness of major analytical domains. Missing values are preserved and not treated as observed No.', ['outcome_category','admission_date','poison_type','component','vitals','labs'], xlabel='Completeness', percent=True, dpi=dpi, logger=logger)
    state.mark_coverage('key_variable_completeness','generated','main_key_variable_completeness')

    # Poisoning epidemiology
    save_barh_series(clean['poison_type'].value_counts().head(15),state,'main_ranked_poisoning_types','Ranked poisoning types','main','Most frequent harmonized poisoning-type categories.', ['poison_type'], dpi=dpi, logger=logger)
    state.mark_coverage('ranked_poisoning_types','generated','main_ranked_poisoning_types')
    rare=clean['poison_type'].value_counts().sort_values().head(20)
    save_barh_series(rare,state,'main_rare_poisoning_categories','Rare poisoning categories','main','Less common harmonized poisoning-type categories retained for auditability.', ['poison_type'], dpi=dpi, logger=logger)
    state.mark_coverage('rare_poisoning_categories','generated','main_rare_poisoning_categories')

    age_type=pd.crosstab(plot_df['age_group'],plot_df['poison_type_plot'],normalize='index').fillna(0)
    save_stacked_percent_table(age_type,state,'main_poisoning_type_by_age_group','Poisoning type by age group','main','Within-age-group distribution of major poisoning types.', ['age_group','poison_type'], dpi=dpi, logger=logger, legend_title='Poisoning type')
    state.mark_coverage('poisoning_type_by_age_group','generated','main_poisoning_type_by_age_group')
    sex_type=pd.crosstab(plot_df['sex'],plot_df['poison_type_plot'],normalize='index').fillna(0)
    save_stacked_percent_table(sex_type,state,'main_poisoning_type_by_sex','Poisoning type by sex','main','Within-sex distribution of major poisoning types.', ['sex','poison_type'], dpi=dpi, logger=logger, legend_title='Poisoning type')
    state.mark_coverage('poisoning_type_by_sex','generated','main_poisoning_type_by_sex')
    site_type=pd.crosstab(plot_df['study_site'],plot_df['poison_type_plot'],normalize='index').fillna(0)
    save_heatmap_table(site_type,state,'main_poisoning_type_by_study_site','Poisoning type by study site','main','Site-level distribution of major poisoning types using within-site denominators.', ['study_site','poison_type'], percent=True, cmap_color='#1d3557', dpi=dpi, logger=logger, max_rows=14, max_cols=10)
    state.mark_coverage('poisoning_type_by_study_site','generated','main_poisoning_type_by_study_site')
    save_barh_series(clean['component'].value_counts().head(15),state,'main_top_components','Top components','main','Most frequent harmonized specific component categories.', ['component'], dpi=dpi, logger=logger)
    state.mark_coverage('top_components','generated','main_top_components')
    heat=pd.crosstab(plot_df['poison_type_plot'],plot_df['component_major']).astype(float)
    if heat.shape[0]>=2 and heat.shape[1]>=2:
        heat=heat.loc[heat.sum(axis=1).sort_values(ascending=False).index[:10], heat.sum(axis=0).sort_values(ascending=False).index[:10]]
        row_ord,col_ord=cluster_order(heat)
        heat=heat.iloc[row_ord,col_ord]
        save_heatmap_table(heat,state,'main_poisoning_type_component_heatmap','Poisoning type × component heatmap','main','Clustered cross-tabulation of harmonized poisoning type by specific component, restricted to informative strata for readability.', ['poison_type','component'], cmap_color='#5a189a', dpi=dpi, logger=logger, max_rows=12, max_cols=12)
        state.mark_coverage('type_component_heatmap','generated','main_poisoning_type_component_heatmap')
    else: state.mark_coverage('type_component_heatmap','skipped',reason='Insufficient type/component cross-tab data')

    # Temporal/site patterns
    dt=plot_df.dropna(subset=['admission_date_for_plots']).copy()
    if not dt.empty:
        monthly=dt.groupby(dt['admission_date_for_plots'].dt.to_period('M')).size()
        fig,ax=prepare_single_axis('Monthly admission trend',len(monthly),10,'line',wide=True)
        ax.plot(monthly.index.to_timestamp(),monthly.values,marker='o',color='#1d3557',linewidth=2)
        ax.set_ylabel('Admissions'); ax.set_xlabel('Month'); despine_and_tidy(ax,'y')
        save_single_figure(fig,state,'main_monthly_admission_trend','Monthly admission trend','main','Monthly admissions using valid admission dates within the inferred study window.', ['admission_date_for_plots'], 'line chart', len(dt), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('monthly_admission_trend','generated','main_monthly_admission_trend')
        top_pt=clean['poison_type'].value_counts().head(5).index
        m2=dt[dt['poison_type'].isin(top_pt)].groupby([dt['admission_date_for_plots'].dt.to_period('M'),'poison_type']).size().unstack(fill_value=0)
        fig,ax=prepare_single_axis('Monthly poisoning-type trend',len(m2),18,'line',wide=True)
        for i,c in enumerate(m2.columns): ax.plot(m2.index.to_timestamp(),m2[c],marker='o',linewidth=1.8,label=str(c),color=CATEGORICAL_BASE[i%len(CATEGORICAL_BASE)])
        ax.set_ylabel('Admissions'); ax.set_xlabel('Month'); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(ax,'y')
        save_single_figure(fig,state,'main_monthly_poisoning_type_trend','Monthly poisoning-type trend','main','Monthly trends for the five most frequent poisoning types.', ['admission_date_for_plots','poison_type'], 'multi-line chart', len(dt), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('monthly_poisoning_type_trend','generated','main_monthly_poisoning_type_trend')
        cal=dt.groupby([dt['admission_date_for_plots'].dt.month,dt['admission_date_for_plots'].dt.day]).size().unstack(fill_value=0).reindex(index=range(1,13),fill_value=0)
        fig,ax=prepare_single_axis('Calendar heatmap of admissions',12,10,'heatmap',wide=True)
        sns.heatmap(cal,cmap=sns.light_palette('#1d3557',as_cmap=True),ax=ax,cbar=True)
        ax.set_yticklabels([calendar.month_abbr[i] for i in range(1,13)],rotation=0); ax.set_xlabel('Day of month'); ax.set_ylabel('Month')
        save_single_figure(fig,state,'main_calendar_heatmap','Calendar heatmap of admissions','main','Calendar-style heatmap showing admission counts by month and day of month.', ['admission_date_for_plots'], 'calendar heatmap', len(dt), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('calendar_heatmap','generated','main_calendar_heatmap')
        dow=dt['admission_day_of_week'].value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
        save_vertical_series(dow,state,'main_day_of_week_pattern','Day-of-week pattern','main','Distribution of admissions by day of week.', ['admission_day_of_week'], ylabel='Admissions', dpi=dpi, logger=logger)
        state.mark_coverage('day_of_week_pattern','generated','main_day_of_week_pattern')
        season_tab=pd.crosstab(dt['season'],dt['poison_type_plot'],normalize='index').fillna(0)
        save_stacked_percent_table(season_tab,state,'main_seasonal_pattern','Seasonal pattern by poisoning type','main','Within-season distribution of major poisoning types.', ['season','poison_type'], dpi=dpi, logger=logger, legend_title='Poisoning type')
        state.mark_coverage('seasonal_pattern','generated','main_seasonal_pattern')
        monthly_site=dt.groupby([dt['admission_date_for_plots'].dt.to_period('M'),'study_site']).size().unstack(fill_value=0)
        top_sites=monthly_site.sum().sort_values(ascending=False).index[:6]
        fig,ax=prepare_single_axis('Site-specific temporal trend',len(monthly_site),18,'line',wide=True)
        for i,site in enumerate(top_sites): ax.plot(monthly_site.index.to_timestamp(),monthly_site[site],linewidth=1.6,label=str(site),color=CATEGORICAL_BASE[i%len(CATEGORICAL_BASE)])
        ax.set_ylabel('Admissions'); ax.set_xlabel('Month'); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(ax,'y')
        save_single_figure(fig,state,'main_site_specific_temporal_trend','Site-specific temporal trend','main','Monthly admission trends for the six highest-volume sites.', ['admission_date_for_plots','study_site'], 'multi-line chart', len(dt), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('site_temporal_trend','generated','main_site_specific_temporal_trend')
    else:
        for key in ['monthly_admission_trend','monthly_poisoning_type_trend','calendar_heatmap','day_of_week_pattern','seasonal_pattern','site_temporal_trend']:
            state.mark_coverage(key,'skipped',reason='No valid admission dates')

    # Clinical presentation and severity
    symptom_bin_cols=[c for c in ['sym_vomited_immediately','sym_fever','sym_vomiting','sym_diarrhoea','sym_abdominal_pain','sym_abdominal_distension','sym_cough','sym_shortness_of_breath','sym_heart_burn','sym_oral_ulcers','sym_leg_swelling','sym_reduced_urine','sym_jaundice','sym_unconsciousness','sym_convulsion','sym_chest_pain','sym_bleeding','sym_shock'] if c in clean.columns]
    if symptom_bin_cols:
        sym_prev=clean[symptom_bin_cols].mean().sort_values()
        sym_prev.index=[c.replace('sym_','').replace('_',' ').title() for c in sym_prev.index]
        save_barh_series(sym_prev,state,'main_ranked_symptom_prevalence','Ranked symptom prevalence','main','Prevalence of recorded clinical symptoms. Percentages use observed binary variables with missing handled in the cleaned dataset.', symptom_bin_cols, xlabel='Prevalence', percent=True, dpi=dpi, logger=logger)
        state.mark_coverage('symptom_prevalence','generated','main_ranked_symptom_prevalence')
        heat_rows=plot_df[plot_df['poison_type_plot']!='Other'].copy()
        heat_tbl=heat_rows.groupby('poison_type_plot')[symptom_bin_cols].mean().T
        heat_tbl.index=[c.replace('sym_','').replace('_',' ').title() for c in heat_tbl.index]
        heat_tbl=heat_tbl.loc[heat_tbl.mean(axis=1).sort_values(ascending=False).index[:14],:]
        if heat_tbl.shape[0]>=2 and heat_tbl.shape[1]>=2:
            row_ord,col_ord=cluster_order(heat_tbl); heat_tbl=heat_tbl.iloc[row_ord,col_ord]
            save_heatmap_table(heat_tbl,state,'main_poisoning_type_symptom_heatmap','Poisoning type × symptom heatmap','main','Symptom prevalence across major poisoning types, clustered and restricted to the most frequent symptoms for readability.', symptom_bin_cols+['poison_type'], percent=True, cmap_color='#bc4749', dpi=dpi, logger=logger, max_rows=14, max_cols=10)
            state.mark_coverage('type_symptom_heatmap','generated','main_poisoning_type_symptom_heatmap')
        fig,ax=prepare_single_axis('Top symptom combinations',8,24,'matrix')
        simple_upset_like(ax,clean[symptom_bin_cols],top_n=8)
        save_single_figure(fig,state,'main_top_symptom_combinations','Top symptom combinations','main','Most frequent symptom co-occurrence combinations among recorded symptom variables.', symptom_bin_cols, 'combination matrix', len(clean), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('symptom_combinations','generated','main_top_symptom_combinations')
        fig,ax=prepare_single_axis('Symptom co-occurrence network',len(symptom_bin_cols),24,'alluvial')
        symptom_network_plot(ax,clean[symptom_bin_cols],min_edge=max(50,int(len(clean)*0.02)))
        save_single_figure(fig,state,'main_symptom_cooccurrence_network','Symptom co-occurrence network','main','Network of symptom co-occurrence relationships after thresholding infrequent edges.', symptom_bin_cols, 'network graph', len(clean), is_main=True, dpi=dpi, logger=logger)
        state.mark_coverage('symptom_network','generated','main_symptom_cooccurrence_network')
    else:
        for key in ['symptom_prevalence','type_symptom_heatmap','symptom_combinations','symptom_network']:
            state.mark_coverage(key,'skipped',reason='No symptom columns')

    severity_prev=plot_df.groupby('poison_type_plot')[['low_gcs','hypoxia','hypotension']].mean().drop(index='Other',errors='ignore').sort_values('low_gcs',ascending=False).head(10)
    if not severity_prev.empty:
        save_heatmap_table(severity_prev.rename(columns={'low_gcs':'Low GCS','hypoxia':'Hypoxia','hypotension':'Hypotension'}),state,'main_severity_indicators_by_poisoning_type','Severity indicators by poisoning type','main','Heatmap of key severity indicators across major poisoning types.', ['low_gcs','hypoxia','hypotension','poison_type'], percent=True, cmap_color='#7f1d1d', dpi=dpi, logger=logger, max_rows=12, max_cols=3)
        state.mark_coverage('severity_by_type','generated','main_severity_indicators_by_poisoning_type')
    for col,label,key in [('low_gcs','Low GCS by poisoning type','low_gcs_by_type'),('hypoxia','Hypoxia by poisoning type','hypoxia_by_type'),('hypotension','Hypotension by poisoning type','hypotension_by_type')]:
        tbl=rate_table(plot_df[plot_df['poison_type_plot']!='Other'], 'poison_type_plot', col, min_n=30).rename(columns={'group':'group'})
        save_rate_ci(tbl,state,f'main_{slugify(label)}',label,'main',f'Poisoning-type-specific rate of {label.lower()} with Wilson 95% confidence intervals.', ['poison_type',col], dpi=dpi, logger=logger, color='#7f1d1d')
        state.mark_coverage(key,'generated',f'main_{slugify(label)}')

    out_order=['Survived uncomplicated','Complication','Absconded/DORB','Death','Missing/unknown']
    sb=clean.groupby('outcome_category')['symptom_burden_score'].median().reindex(out_order).dropna().sort_values()
    save_barh_series(sb,state,'main_symptom_burden_by_outcome','Symptom burden by outcome','main','Median symptom burden score by outcome category.', ['symptom_burden_score','outcome_category'], xlabel='Median symptom burden score', dpi=dpi, logger=logger)
    state.mark_coverage('symptom_burden_by_outcome','generated','main_symptom_burden_by_outcome')

    # Treatment and outcomes
    treat_cols=[c for c in ['oxygen_any','ng_suction','dialysis_any','ventilation_support','operation_support'] if c in clean.columns]
    if treat_cols:
        treat_heat=plot_df.groupby('poison_type_plot')[treat_cols].mean().drop(index='Other',errors='ignore').head(10)
        save_heatmap_table(treat_heat.rename(columns={c:c.replace('_',' ').title() for c in treat_cols}),state,'main_treatment_support_heatmap_by_poisoning_type','Treatment/support heatmap by poisoning type','main','Supportive-treatment proportions by major poisoning type.', treat_cols+['poison_type'], percent=True, cmap_color='#1b4332', dpi=dpi, logger=logger, max_rows=12, max_cols=6)
        state.mark_coverage('treatment_heatmap','generated','main_treatment_support_heatmap_by_poisoning_type')
        inter=clean[treat_cols].mean().sort_values()
        inter.index=[c.replace('_',' ').title() for c in inter.index]
        save_barh_series(inter,state,'main_supportive_intervention_rates','Supportive intervention rates','main','Overall supportive intervention rates across the cohort.', treat_cols, xlabel='Rate', percent=True, dpi=dpi, logger=logger)
        state.mark_coverage('intervention_rates','generated','main_supportive_intervention_rates')
    out_counts=clean['outcome_category'].value_counts().reindex(out_order).dropna(); out_props=out_counts/out_counts.sum()
    save_vertical_series(out_props,state,'main_outcome_distribution','Outcome distribution','main','Outcome distribution using the full analysis denominator.', ['outcome_category'], ylabel='Proportion', percent=True, dpi=dpi, logger=logger)
    state.mark_coverage('outcome_distribution','generated','main_outcome_distribution')
    death_tbl=rate_table(plot_df[plot_df['poison_type_plot']!='Other'], 'poison_type_plot','death_flag',min_n=30)
    save_rate_ci(death_tbl,state,'main_death_rate_by_poisoning_type','Death rate by poisoning type','main','Death rate by major poisoning type using observed denominators and Wilson 95% confidence intervals.', ['poison_type','death_flag'], dpi=dpi, logger=logger, color='#bc4749')
    state.mark_coverage('death_rate_by_type','generated','main_death_rate_by_poisoning_type')
    severe_tbl=rate_table(plot_df[plot_df['poison_type_plot']!='Other'], 'poison_type_plot','severe_outcome',min_n=30)
    save_rate_ci(severe_tbl,state,'main_severe_outcome_rate_by_poisoning_type','Severe outcome rate by poisoning type','main','Composite severe-outcome rate by major poisoning type using Wilson 95% confidence intervals.', ['poison_type','severe_outcome'], dpi=dpi, logger=logger, color='#7f1d1d')
    state.mark_coverage('severe_rate_by_type','generated','main_severe_outcome_rate_by_poisoning_type')
    ti=clean.groupby('outcome_category')['treatment_intensity_score'].median().reindex(out_order).dropna().sort_values()
    save_barh_series(ti,state,'main_treatment_intensity_by_outcome','Treatment intensity by outcome','main','Median treatment-intensity score by outcome category.', ['treatment_intensity_score','outcome_category'], xlabel='Median treatment intensity score', dpi=dpi, logger=logger)
    state.mark_coverage('treatment_intensity_by_outcome','generated','main_treatment_intensity_by_outcome')
    follow=clean['followup_status'].fillna('Missing/unknown').value_counts(normalize=True).sort_values()
    save_barh_series(follow,state,'main_followup_status_summary','Follow-up status summary','main','Distribution of recorded follow-up status.', ['followup_status'], xlabel='Proportion', percent=True, dpi=dpi, logger=logger)
    state.mark_coverage('followup_summary','generated','main_followup_status_summary')
    ftab=pd.crosstab(clean['poison_type_major'],clean['followup_status'],normalize='index').fillna(0)
    save_heatmap_table(ftab,state,'main_followup_by_poisoning_type','Follow-up by poisoning type','main','Follow-up status distribution by major poisoning type.', ['poison_type_major','followup_status'], percent=True, cmap_color='#264653', dpi=dpi, logger=logger, max_rows=12, max_cols=8)
    state.mark_coverage('followup_by_type','generated','main_followup_by_poisoning_type')

    # Modeling and association analyses
    death_uni=univariable_or_table(clean,'death_flag',logger); death_adj_cc,death_meta_cc,death_model_cc,death_X_cc,death_y_cc=fit_adjusted_logistic(clean,'death_flag',impute=False,logger=logger); death_adj_imp,death_meta_imp,death_model_imp,death_X_imp,death_y_imp=fit_adjusted_logistic(clean,'death_flag',impute=True,logger=logger)
    severe_uni=univariable_or_table(clean,'severe_outcome',logger); severe_adj_cc,severe_meta_cc,severe_model_cc,severe_X_cc,severe_y_cc=fit_adjusted_logistic(clean,'severe_outcome',impute=False,logger=logger); severe_adj_imp,severe_meta_imp,severe_model_imp,severe_X_imp,severe_y_imp=fit_adjusted_logistic(clean,'severe_outcome',impute=True,logger=logger)
    results.update({'death_uni':death_uni,'death_adj_cc':death_adj_cc,'death_adj_imp':death_adj_imp,'severe_uni':severe_uni,'severe_adj_cc':severe_adj_cc,'severe_adj_imp':severe_adj_imp,'death_meta_cc':death_meta_cc,'death_meta_imp':death_meta_imp,'severe_meta_cc':severe_meta_cc,'severe_meta_imp':severe_meta_imp,'death_X_imp':death_X_imp,'death_y_imp':death_y_imp})
    save_forest(death_uni,state,'main_univariable_odds_ratios_for_death','Univariable odds ratios for death','main','Univariable odds ratios for death. Effects are observational and should not be interpreted causally.', ['death_flag'], dpi=dpi, logger=logger); state.mark_coverage('univariable_death_or','generated','main_univariable_odds_ratios_for_death')
    save_forest(death_adj_cc,state,'main_adjusted_odds_ratios_for_death','Adjusted odds ratios for death','main','Complete-case adjusted odds ratios for death. Penalized estimates are used where classical models are unstable.', ['death_flag'], dpi=dpi, logger=logger); state.mark_coverage('adjusted_death_or','generated','main_adjusted_odds_ratios_for_death')
    save_forest(severe_uni,state,'main_univariable_odds_ratios_for_severe_outcome','Univariable odds ratios for severe outcome','main','Univariable odds ratios for the composite severe outcome.', ['severe_outcome'], dpi=dpi, logger=logger); state.mark_coverage('univariable_severe_or','generated','main_univariable_odds_ratios_for_severe_outcome')
    save_forest(severe_adj_cc,state,'main_adjusted_odds_ratios_for_severe_outcome','Adjusted odds ratios for severe outcome','main','Complete-case adjusted odds ratios for the composite severe outcome.', ['severe_outcome'], dpi=dpi, logger=logger); state.mark_coverage('adjusted_severe_or','generated','main_adjusted_odds_ratios_for_severe_outcome')
    if not death_adj_cc.empty and not death_adj_imp.empty:
        sens=death_adj_cc[['term','or']].merge(death_adj_imp[['term','or']],on='term',suffixes=('_cc','_imp'),how='inner').head(12)
        if not sens.empty:
            fig,ax=prepare_single_axis('Sensitivity analysis: complete-case vs imputed',len(sens),30,'line')
            ax.scatter(sens['or_cc'],sens['or_imp'],color='#1d3557',s=34)
            maxv=np.nanmax(np.r_[sens['or_cc'].values,sens['or_imp'].values]) if np.isfinite(np.r_[sens['or_cc'].values,sens['or_imp'].values]).any() else 2
            ax.plot([0,maxv],[0,maxv],linestyle='--',color='#bc4749',linewidth=1)
            for _,r in sens.iterrows(): ax.annotate(str(r['term']).replace('_',' '),(r['or_cc'],r['or_imp']),fontsize=7,xytext=(4,4),textcoords='offset points')
            ax.set_xlabel('Odds ratio: complete-case model'); ax.set_ylabel('Odds ratio: imputed model'); despine_and_tidy(ax,'both')
            save_single_figure(fig,state,'main_death_model_sensitivity_complete_case_vs_imputed','Sensitivity analysis: complete-case vs imputed','main','Comparison of death-model odds ratios from complete-case and imputed analyses.', ['death_flag'], 'sensitivity scatter plot', len(sens), is_main=True, is_model=True, dpi=dpi, logger=logger)
            state.mark_coverage('model_sensitivity','generated','main_death_model_sensitivity_complete_case_vs_imputed')
    else: state.mark_coverage('model_sensitivity','skipped',reason='Death model sensitivity unavailable')
    roc_info,calib=cross_validated_roc_and_calibration(death_X_imp,death_y_imp,logger) if death_X_imp is not None and death_y_imp is not None else (None,None)
    if roc_info is not None:
        fig,ax=prepare_single_axis('Exploratory ROC curve for death model',10,20,'line')
        ax.plot(roc_info['diag']['fpr'],roc_info['diag']['tpr'],color='#1d3557',lw=2,label=f"CV AUC = {roc_info['auc']:.2f}")
        ax.plot([0,1],[0,1],linestyle='--',color='#adb5bd'); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.legend(frameon=False,loc='lower right'); despine_and_tidy(ax,'both')
        save_single_figure(fig,state,'main_exploratory_roc_curve_for_death_model','Exploratory ROC curve for death model','main','Internally cross-validated ROC curve for the death model, shown as exploratory model diagnostics only.', ['death_flag'], 'ROC curve', len(death_y_imp), is_main=True, is_model=True, dpi=dpi, logger=logger)
        state.mark_coverage('roc_curve','generated','main_exploratory_roc_curve_for_death_model')
        fig,ax=prepare_single_axis('Exploratory calibration plot for death model',10,20,'line')
        ax.plot(calib['mean_pred'],calib['frac_pos'],'o-',color='#bc4749'); ax.plot([0,1],[0,1],'--',color='#adb5bd')
        ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed fraction'); despine_and_tidy(ax,'both')
        save_single_figure(fig,state,'main_exploratory_calibration_plot_for_death_model','Exploratory calibration plot for death model','main','Exploratory calibration curve for the death model.', ['death_flag'], 'calibration plot', len(death_y_imp), is_main=True, is_model=True, dpi=dpi, logger=logger)
        state.mark_coverage('calibration_plot','generated','main_exploratory_calibration_plot_for_death_model')
    else:
        state.mark_coverage('roc_curve','skipped',reason='ROC/calibration not statistically appropriate or unavailable')
        state.mark_coverage('calibration_plot','skipped',reason='ROC/calibration not statistically appropriate or unavailable')
    return results

def generate_single_supplementary_figures(clean,dq,metadata,model_results,state,logger,dpi=600):
    # Missingness matrix
    vars_show=[c for c in ['study_site','sex','age_years','poison_type','component','admission_date_for_plots','time_to_presentation_hrs','gcs','spo2','sbp','creatinine_mg_dl','ph','outcome_category','severe_outcome'] if c in clean.columns]
    if vars_show:
        fig,ax=prepare_single_axis('Missingness matrix',len(vars_show),24,'matrix')
        miss=clean[vars_show].isna().astype(int).sample(min(len(clean),400),random_state=SEED)
        sns.heatmap(miss.T,cmap=sns.color_palette(['#f8f9fa','#1d3557'],as_cmap=True),cbar=False,ax=ax)
        ax.set_xlabel('Record sample'); ax.set_ylabel('Variable'); ax.set_yticklabels(fit_wrapped_labels(vars_show,width=20,max_chars=48),rotation=0); ax.set_xticks([])
        save_single_figure(fig,state,'supp_missingness_matrix','Missingness matrix','supplementary','Missingness matrix for selected high-value analysis variables; rows are variables and columns are a random sample of records for display readability.', vars_show, 'missingness heatmap', len(clean), dpi=dpi, logger=logger)
        state.mark_coverage('missingness_matrix','generated','supp_missingness_matrix')
    # Data quality split into single figures
    date_flags=clean['admission_date_quality_flag'].fillna('missing').value_counts().sort_values()
    save_barh_series(date_flags,state,'supp_date_quality_flags','Date-quality flags','supplementary','Admission-date parsing and quality flags.', ['admission_date_quality_flag'], dpi=dpi, logger=logger); state.mark_coverage('date_quality_flags','generated','supp_date_quality_flags')
    implausible_counts={c.replace('_flag',''):int((clean[c]=='implausible').sum()) for c in clean.columns if c.endswith('_flag') and c not in {'admission_date_quality_flag','ingestion_date_quality_flag'}}
    implausible_counts=pd.Series(implausible_counts).sort_values().tail(20)
    save_barh_series(implausible_counts,state,'supp_numeric_quality_flags','Implausible numeric values flagged','supplementary','Counts of values outside clinically plausible ranges by variable.', list(implausible_counts.index), dpi=dpi, logger=logger); state.mark_coverage('numeric_quality_flags','generated','supp_numeric_quality_flags')
    top_missing=dq.sort_values('missing_pct',ascending=False).head(20).set_index('variable')['missing_pct']/100
    save_barh_series(top_missing.sort_values(),state,'supp_highest_missingness_variables','Highest missingness variables','supplementary','Variables with the highest missingness proportions in the cleaned dataset.', ['missing_pct'], xlabel='Missingness', percent=True, dpi=dpi, logger=logger); state.mark_coverage('highest_missingness','generated','supp_highest_missingness_variables')
    # Harmonization
    pm=clean[['poison_type_raw','poison_type']].dropna().value_counts().reset_index(name='n').head(18)
    if not pm.empty:
        s=pd.Series(pm['n'].values,index=[f"{a} → {b}" for a,b in zip(pm['poison_type_raw'],pm['poison_type'])])
        save_barh_series(s.sort_values(),state,'supp_poisoning_type_harmonization','Poisoning-type harmonization summary','supplementary','Most frequent raw-to-clean poisoning-type mapping examples.', ['poison_type_raw','poison_type'], dpi=dpi, logger=logger)
        state.mark_coverage('harmonization_poison','generated','supp_poisoning_type_harmonization')
    cm=clean[['component_raw','component']].dropna().value_counts().reset_index(name='n').head(18)
    if not cm.empty:
        s=pd.Series(cm['n'].values,index=[f"{a} → {b}" for a,b in zip(cm['component_raw'],cm['component'])])
        save_barh_series(s.sort_values(),state,'supp_component_harmonization','Component harmonization summary','supplementary','Most frequent raw-to-clean component mapping examples.', ['component_raw','component'], dpi=dpi, logger=logger)
        state.mark_coverage('harmonization_component','generated','supp_component_harmonization')
    # Context stratified plots
    for col,title,key in [('occupation','Occupation distribution','context_occupation'),('living_area','Living-area distribution','context_living_area'),('presentation_area','Presentation-area distribution','context_presentation_area')]:
        if col in clean.columns:
            vc=safe_series_counts(clean[col],top_n=16)
            save_barh_series(vc.sort_values(),state,f'supp_{slugify(title)}',title,'supplementary',f'Distribution of {title.lower()}.', [col], dpi=dpi, logger=logger)
            state.mark_coverage(key,'generated',f'supp_{slugify(title)}')
    # Labs and correlations
    lab_cols=[c for c in ['wbc_count_mm3','creatinine_mg_dl','ph','hco3_mmol_l','sodium_mmol_l','potassium_mmol_l','hemoglobin_g_dl','bilirubin_mg_dl','sgpt_u_l','sgot_u_l'] if c in clean.columns and clean[c].notna().sum()>=20]
    if lab_cols:
        completeness=clean[lab_cols].notna().mean().sort_values()
        save_barh_series(completeness,state,'supp_lab_completeness','Laboratory completeness','supplementary','Completeness of selected laboratory variables.', lab_cols, xlabel='Completeness', percent=True, dpi=dpi, logger=logger); state.mark_coverage('lab_completeness','generated','supp_lab_completeness')
        abnormal=[]
        ref={'creatinine_mg_dl':('Creatinine >1.2 mg/dL',lambda s:s>1.2),'ph':('pH <7.35',lambda s:s<7.35),'hco3_mmol_l':('HCO3 <22 mmol/L',lambda s:s<22),'wbc_count_mm3':('WBC >11000/mm3',lambda s:s>11000),'potassium_mmol_l':('Potassium >5.2 mmol/L',lambda s:s>5.2)}
        for col,(lab,fn) in ref.items():
            if col in clean.columns and clean[col].notna().sum()>=20: abnormal.append({'lab':lab,'pct':fn(clean[col].dropna()).mean()})
        if abnormal:
            ab=pd.DataFrame(abnormal).set_index('lab')['pct'].sort_values()
            save_barh_series(ab,state,'supp_selected_lab_abnormalities','Selected laboratory abnormality prevalence','supplementary','Prevalence of selected laboratory abnormalities among patients with observed values.', list(ref.keys()), xlabel='Prevalence', percent=True, dpi=dpi, logger=logger); state.mark_coverage('lab_abnormality','generated','supp_selected_lab_abnormalities')
        for lab in lab_cols[:6]:
            if clean[lab].notna().sum()>=30:
                fig,ax=prepare_single_axis(f'{lab.replace("_"," ").title()} by outcome',5,22,'bar')
                sns.boxplot(data=clean,x='outcome_category',y=lab,ax=ax,showfliers=False,color='#d7e3fc')
                ax.set_xlabel('Outcome'); ax.set_ylabel(lab.replace('_',' ')); ax.set_xticklabels(fit_wrapped_labels([t.get_text() for t in ax.get_xticklabels()],width=14,max_chars=40),rotation=0)
                despine_and_tidy(ax,'y')
                save_single_figure(fig,state,f'supp_lab_{slugify(lab)}_by_outcome',f'{lab.replace("_"," ").title()} by outcome','supplementary',f'Observed {lab} distribution by outcome category; extreme fliers are suppressed for readability.', [lab,'outcome_category'], 'boxplot', int(clean[lab].notna().sum()), dpi=dpi, logger=logger)
    else:
        state.mark_coverage('lab_completeness','skipped',reason='Insufficient laboratory data')
        state.mark_coverage('lab_abnormality','skipped',reason='Insufficient laboratory data')
    corr_cols=[c for c in ['gcs','spo2','sbp','dbp','pulse_bpm','respiratory_rate','creatinine_mg_dl','ph','hco3_mmol_l','glucose_mmol_l'] if c in clean.columns and clean[c].notna().sum()>=30]
    if len(corr_cols)>=4:
        corr=clean[corr_cols].corr(method='spearman')
        save_heatmap_table(corr,state,'supp_vital_lab_correlation_heatmap','Vital/lab correlation heatmap','supplementary','Spearman correlation matrix for sufficiently complete vital-sign and laboratory variables.', corr_cols, cmap_color='#1d3557', dpi=dpi, logger=logger, max_rows=12, max_cols=12)
        state.mark_coverage('vital_lab_correlation','generated','supp_vital_lab_correlation_heatmap')
    else: state.mark_coverage('vital_lab_correlation','skipped',reason='Fewer than four sufficiently complete numeric variables')
    # Alluvial/pathway figures as single visual analyses
    alluvial_specs=[('alluvial_site_type_outcome','supp_alluvial_site_type_outcome',['study_site','poison_type_major','outcome_category'],'Site → poisoning type → outcome'),('alluvial_age_type_outcome','supp_alluvial_age_type_outcome',['age_group','poison_type_major','outcome_category'],'Age group → poisoning type → outcome'),('alluvial_type_treatment_outcome','supp_alluvial_type_treatment_outcome',['poison_type_major','outcome_category','followup_status'],'Poisoning type → outcome → follow-up status'),('followup_pathway','supp_followup_pathway',['poison_type_major','outcome_category','followup_status'],'Follow-up pathway')]
    for key,fig_id,cols,title in alluvial_specs:
        fig,ax=prepare_single_axis(title,6,22,'alluvial')
        ok=custom_alluvial_three_stage(ax,clean,cols,top_n=6,title=title)
        if ok:
            ax.set_title(wrap_title(title),loc='left',pad=14,fontsize=14,fontweight='bold')
            save_single_figure(fig,state,fig_id,title,'supplementary',f'Static alluvial summary of {title.lower()}; categories are truncated to frequent levels for readability.', cols, 'static alluvial diagram', len(clean), dpi=dpi, logger=logger)
            state.mark_coverage(key,'generated',fig_id)
        else:
            plt.close(fig); state.mark_coverage(key,'skipped',reason='Insufficient complete data')
    # Delay/amount and heterogeneity/sensitivity
    tab=pd.crosstab(clean['presentation_time_category'],clean['severe_outcome'],normalize='index').fillna(0)
    if not tab.empty:
        tab=tab.rename(columns={0:'No severe outcome',1:'Severe outcome'})
        save_stacked_percent_table(tab,state,'supp_delay_vs_severity','Presentation delay versus severe outcome','supplementary','Severe outcome proportion by presentation-delay category.', ['presentation_time_category','severe_outcome'], dpi=dpi, logger=logger, legend_title='Outcome')
        state.mark_coverage('delay_vs_severity','generated','supp_delay_vs_severity')
    if 'amount_ingested_ml' in clean.columns and clean['amount_ingested_ml'].notna().sum()>=30:
        fig,ax=prepare_single_axis('Amount ingested by outcome',5,24,'bar')
        sns.boxplot(data=clean,x='outcome_category',y='amount_ingested_ml',ax=ax,showfliers=False,color='#d7e3fc')
        ax.set_xlabel('Outcome'); ax.set_ylabel('Amount ingested (mL)'); ax.set_xticklabels(fit_wrapped_labels([t.get_text() for t in ax.get_xticklabels()],width=14,max_chars=40),rotation=0); despine_and_tidy(ax,'y')
        save_single_figure(fig,state,'supp_amount_ingested_by_outcome','Amount ingested by outcome','supplementary','Observed amount ingested by outcome category; extreme fliers are suppressed for readability.', ['amount_ingested_ml','outcome_category'], 'boxplot', int(clean['amount_ingested_ml'].notna().sum()), dpi=dpi, logger=logger)
        state.mark_coverage('amount_vs_outcome','generated','supp_amount_ingested_by_outcome')
    site_tbl=rate_table(clean,'study_site','death_flag',min_n=20)
    save_rate_ci(site_tbl,state,'supp_site_heterogeneity_death_rate','Site heterogeneity in death rate','supplementary','Site-level death-rate heterogeneity with Wilson 95% confidence intervals.', ['study_site','death_flag'], dpi=dpi, logger=logger, color='#bc4749'); state.mark_coverage('site_heterogeneity','generated','supp_site_heterogeneity_death_rate')
    full=clean.groupby('poison_type_major')['severe_outcome'].mean().rename('Main collapsed grouping')
    sparse=clean['poison_type'].value_counts(); sparse_keep=sparse[sparse>=30].index
    filt=clean[clean['poison_type'].isin(sparse_keep)].groupby('poison_type')['severe_outcome'].mean().rename('Excluding sparse categories')
    comp=pd.concat([full,filt],axis=1).dropna(how='all').head(14).sort_values('Main collapsed grouping')
    if not comp.empty:
        fig,ax=prepare_single_axis('Sparse-category sensitivity',len(comp),30,'barh')
        y=np.arange(len(comp)); ax.scatter(comp['Main collapsed grouping'],y,color='#4e79a7',label='Main collapsed grouping')
        if 'Excluding sparse categories' in comp: ax.scatter(comp['Excluding sparse categories'],y,color='#bc4749',label='Excluding sparse categories')
        ax.set_yticks(y); ax.set_yticklabels(fit_wrapped_labels(comp.index,width=22,max_chars=60)); ax.set_xlabel('Severe outcome rate'); ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(ax,'x')
        save_single_figure(fig,state,'supp_sparse_category_sensitivity','Sparse-category sensitivity','supplementary','Comparison of severe-outcome rates under main collapsed grouping and sensitivity excluding sparse categories.', ['poison_type','poison_type_major','severe_outcome'], 'sensitivity dot plot', len(comp), dpi=dpi, logger=logger)
        state.mark_coverage('sparse_category_sensitivity','generated','supp_sparse_category_sensitivity')
    impl_cols=[c for c in clean.columns if c.endswith('_flag') and c not in {'admission_date_quality_flag','ingestion_date_quality_flag'}]
    impl_mask=pd.DataFrame({c:clean[c]=='implausible' for c in impl_cols if clean[c].dtype=='object'}) if impl_cols else pd.DataFrame(index=clean.index)
    if not impl_mask.empty:
        clean_impl=clean.loc[~impl_mask.any(axis=1)]
        a=clean.groupby('poison_type_major')['severe_outcome'].mean().rename('All records')
        b=clean_impl.groupby('poison_type_major')['severe_outcome'].mean().rename('Excluding implausible numeric values')
        sens=pd.concat([a,b],axis=1).dropna(how='all').head(14).sort_values('All records')
        fig,ax=prepare_single_axis('Implausible numeric sensitivity',len(sens),32,'barh')
        y=np.arange(len(sens)); ax.scatter(sens['All records'],y,color='#6c757d',label='All records'); ax.scatter(sens['Excluding implausible numeric values'],y,color='#1b4332',label='Excluding implausible numeric values')
        ax.set_yticks(y); ax.set_yticklabels(fit_wrapped_labels(sens.index,width=22,max_chars=60)); ax.set_xlabel('Severe outcome rate'); ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(ax,'x')
        save_single_figure(fig,state,'supp_implausible_numeric_sensitivity','Implausible numeric sensitivity','supplementary','Sensitivity analysis comparing severe-outcome rates before and after excluding records with any implausible numeric flag.', ['severe_outcome']+impl_cols, 'sensitivity dot plot', len(sens), dpi=dpi, logger=logger)
        state.mark_coverage('implausible_numeric_sensitivity','generated','supp_implausible_numeric_sensitivity')
    else: state.mark_coverage('implausible_numeric_sensitivity','skipped',reason='No implausible numeric flags')
    # PCA
    if OPTIONAL_IMPORTS['sklearn'] and len(lab_cols)>=5 and clean[lab_cols].dropna().shape[0]>=50:
        X=clean[lab_cols].dropna(); comps=PCA(n_components=2,random_state=SEED).fit_transform(StandardScaler().fit_transform(X)); pca_df=pd.DataFrame({'PC1':comps[:,0],'PC2':comps[:,1],'Outcome':clean.loc[X.index,'outcome_category'].fillna('Missing')})
        fig,ax=prepare_single_axis('PCA of laboratory data',6,25,'line')
        for i,(lab,sub) in enumerate(pca_df.groupby('Outcome')): ax.scatter(sub['PC1'],sub['PC2'],s=22,alpha=0.75,label=lab,color=CATEGORICAL_BASE[i%len(CATEGORICAL_BASE)])
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(ax,'both')
        save_single_figure(fig,state,'supp_pca_laboratory_data','PCA of laboratory data','supplementary','Exploratory PCA using complete laboratory cases only.', lab_cols+['outcome_category'], 'PCA scatter plot', len(X), dpi=dpi, logger=logger)
        state.mark_coverage('pca_labs','generated','supp_pca_laboratory_data')
    else: state.mark_coverage('pca_labs','skipped',reason='Insufficient complete laboratory data')

    # Exploratory
    exp_tbl=pd.crosstab(clean['component_major'],clean['outcome_category'])
    if not exp_tbl.empty:
        exp_tbl=exp_tbl.loc[exp_tbl.sum(axis=1).sort_values(ascending=False).index[:18]]
        save_heatmap_table(exp_tbl,state,'exp_component_outcome_heatmap','Component × outcome heatmap','exploratory','Exploratory cross-tabulation of major component groups against outcome categories.', ['component_major','outcome_category'], cmap_color='#2a9d8f', dpi=dpi, logger=logger, max_rows=18, max_cols=8)
        state.mark_coverage('component_outcome_heatmap','generated','exp_component_outcome_heatmap')
    obs=clean.groupby('admission_date_quality_flag').size().sort_values()
    save_barh_series(obs,state,'exp_temporal_quality_flags','Temporal quality flags','exploratory','Exploratory audit of admission-date parsing and filtering flags.', ['admission_date_quality_flag'], dpi=dpi, logger=logger)
    state.mark_coverage('temporal_qc','generated','exp_temporal_quality_flags')

    # Predictor importance: only if a penalized model exists with coefficients
    imp=model_results.get('death_adj_imp')
    if isinstance(imp,pd.DataFrame) and not imp.empty and 'or' in imp.columns:
        tmp=imp.copy(); tmp['importance']=np.abs(np.log(pd.to_numeric(tmp['or'],errors='coerce'))); tmp=tmp.dropna(subset=['importance']).sort_values('importance').tail(14)
        if not tmp.empty:
            s=pd.Series(tmp['importance'].values,index=tmp.get('term_display',tmp['term']))
            save_barh_series(s,state,'exp_predictor_importance_death_model','Exploratory predictor importance for death model','exploratory','Absolute log-odds coefficient magnitude from the penalized death model; this is exploratory and not causal.', list(tmp.get('term',[])), xlabel='Absolute log odds coefficient', dpi=dpi, logger=logger)
            state.mark_coverage('predictor_importance','generated','exp_predictor_importance_death_model')
    else: state.mark_coverage('predictor_importance','skipped',reason='Adjusted/imputed model unavailable')


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--dpi',type=int,default=600,help='Raster export DPI; default 600 for publication output')
    args=ap.parse_args(argv)
    input_path=Path(args.input).resolve(); outdir=Path(args.outdir).resolve(); script_path=Path(__file__).resolve(); run_id=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
    initialise_output_tree(outdir,args.overwrite)
    logger=init_logging(outdir)
    logger.info('Starting poison publication figure pipeline v3: single standalone figures only')
    state=AnalysisState(outdir,run_id)
    try:
        raw,unique_names,dropdowns,header_map=load_workbook_sheets(input_path,logger)
        clean,dq,metadata=build_analysis_dataset(raw,unique_names,dropdowns,logger)
        save_clean_outputs(clean,header_map,dq,outdir)
        save_summary_tables(clean,dq,state)
        model_results=generate_single_main_figures(clean,state,logger,dpi=args.dpi)
        generate_single_supplementary_figures(clean,dq,metadata,model_results,state,logger,dpi=args.dpi)
        save_version_control(input_path,script_path,outdir,run_id,metadata,dpi=args.dpi)
        write_registry_and_scorecards(state); make_gallery(state); qa_posthoc(state,clean)
        logger.info('Pipeline completed successfully. Generated %s single standalone figures.',len(state.figure_registry))
        return 0
    except Exception as e:
        logger.error('Pipeline failed: %s',e)
        logger.error(traceback.format_exc())
        raise

if __name__=='__main__': raise SystemExit(main())
