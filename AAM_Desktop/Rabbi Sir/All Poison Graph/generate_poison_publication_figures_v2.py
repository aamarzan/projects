#!/usr/bin/env python3
# PATCH VERSION: 20260614_FINAL_SAFE4_MAINFOLDER
from __future__ import annotations
import argparse, calendar, hashlib, json, logging, math, os, platform, re, subprocess, sys, textwrap, traceback, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np, pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='Data Validation extension is not supported*')
warnings.filterwarnings('ignore', category=FutureWarning)

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
from matplotlib.gridspec import GridSpec
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
OUTPUT_SUBDIRS=['00_logs','01_clean_data','02_tables','03_main_manuscript_figures','04_supplementary_figures','05_exploratory_figures','06_interactive_html','07_multiplanel_assembled_figures','08_quality_control','09_version_control']
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
    if pd.isna(x): return None
    s=normalize_space(str(x).replace('\u00a0',' '))
    if s=='' or s.lower() in {'nan','none','n/a','na','#ref!','#value!','#name?','ref'}: return None
    return s
def normalize_text_lower(x:Any)->Optional[str]:
    s=normalize_text(x); return s.lower() if s is not None else None
wrap=lambda s,width=26:'\n'.join(textwrap.wrap(str(s if s is not None else 'Missing'),width=width))
def slugify(s:str)->str: return re.sub(r'_+','_',re.sub(r'[^a-zA-Z0-9]+','_',s.strip().lower())).strip('_')

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

def categorical_clean(s:pd.Series)->pd.Series: return s.map(normalize_text).astype('object')
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
    raw_poison=categorical_clean(raw.get('Types of poisoning')); clean['poison_type_raw']=raw_poison; clean['poison_type']=raw_poison.map(lambda x: poison_map.get(safe_lower(x),rule_based_poison_type(x)) if safe_lower(x) else 'Unknown')
    raw_component=categorical_clean(raw.get('Name of the specific component')); clean['component_raw']=raw_component; clean['component']=raw_component.map(lambda x: comp_map.get(safe_lower(x),rule_based_component(x)) if safe_lower(x) else 'Unknown/Other')
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

def initialise_output_tree(outdir:Path, overwrite:bool):
    if outdir.exists() and any(outdir.iterdir()) and not overwrite: raise FileExistsError(f'Output directory already exists and is not empty: {outdir}')
    ensure_dir(outdir); [ensure_dir(outdir/d) for d in OUTPUT_SUBDIRS]

def init_logging(outdir:Path):
    logger=logging.getLogger('poison_v2'); logger.setLevel(logging.INFO); logger.handlers.clear(); fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh=logging.FileHandler(outdir/'00_logs'/'run.log',mode='w',encoding='utf-8'); fh.setFormatter(fmt); sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(sh); return logger

def add_panel_letter(ax, letter): ax.text(-0.08,1.08,letter,transform=ax.transAxes,fontsize=14,fontweight='bold',va='top',ha='left')
def despine_and_tidy(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True,axis=grid_axis,color='#d9dde3',linewidth=0.5) if grid_axis in {'x','y','both'} else ax.grid(False)

def write_caption(path:Path,title:str,caption:str): path.write_text(f'{title}\n\n{caption}\n',encoding='utf-8')
def heuristic_score(title,n_categories=0,is_model=False,is_main=False,qa_notes=''):
    leg=9 if n_categories<=12 else (8 if n_categories<=18 else 6); vis=9 if 'artifact' not in qa_notes.lower() else 5; sci=9 if not qa_notes else 8; stat=9 if is_model else 8; ms=9 if is_main else 7
    return {'scientific_validity':sci,'legibility':leg,'manuscript_suitability':ms,'statistical_appropriateness':stat,'visual_quality':vis}

def save_figure(fig,state,fig_id,title,section,subdir,caption,variables,analysis_type,n,gradient=None,qa_notes='',is_model=False,is_main=False,dpi=600):
    # Main manuscript figures should be easy to find.
    # Save them primarily in 03_main_manuscript_figures, and also mirror copies
    # in 07_multiplanel_assembled_figures for backward compatibility with v2 outputs.
    primary_subdir='03_main_manuscript_figures' if section=='main' else subdir
    base=state.outdir/primary_subdir/fig_id
    png=str(base.with_suffix('.png')); pdf=str(base.with_suffix('.pdf')); svg=str(base.with_suffix('.svg'))
    fig.savefig(png,dpi=dpi,facecolor='white'); fig.savefig(pdf,facecolor='white'); fig.savefig(svg,facecolor='white')
    cap_path=base.with_suffix('.txt'); write_caption(cap_path,title,caption)
    if section=='main':
        mirror_dir=state.outdir/'07_multiplanel_assembled_figures'
        ensure_dir(mirror_dir)
        for source_path in [Path(png),Path(pdf),Path(svg),cap_path]:
            mirror_path=mirror_dir/source_path.name
            mirror_path.write_bytes(source_path.read_bytes())
    if gradient is None: gradient=state.choose_gradient()
    scores=heuristic_score(title,is_model=is_model,is_main=is_main,qa_notes=qa_notes)
    rec=FigureRecord(figure_id=fig_id,title=title,section=section,base_filename=str(base.name),variables='; '.join(variables),analysis_type=analysis_type,gradient=gradient,n=n,caption_file=str(cap_path.relative_to(state.outdir)),png_file=str(Path(png).relative_to(state.outdir)),pdf_file=str(Path(pdf).relative_to(state.outdir)),svg_file=str(Path(svg).relative_to(state.outdir)),recommended_tier=section,qa_notes=qa_notes,**scores)
    state.figure_registry.append(rec); plt.close(fig); return rec

def write_registry_and_scorecards(state):
    for row in state.coverage_rows:
        if row.get('status')=='planned':
            row['status']='skipped'; row['reason']='Optional or data-driven analysis not triggered in this run'
            state.skip_rows.append({'analysis_id':row['analysis_id'],'reason':row['reason']})
    reg_df=pd.DataFrame([asdict(r) for r in state.figure_registry]); reg_df.to_csv(state.outdir/'09_version_control'/'figure_registry.csv',index=False); reg_df.to_json(state.outdir/'09_version_control'/'figure_registry.json',orient='records',indent=2)
    reg_df[['figure_id','title','scientific_validity','legibility','manuscript_suitability','statistical_appropriateness','visual_quality','recommended_tier','qa_notes']].to_csv(state.outdir/'09_version_control'/'figure_scorecard.csv',index=False)
    pd.DataFrame(state.coverage_rows).to_csv(state.outdir/'09_version_control'/'coverage_matrix.csv',index=False); pd.DataFrame(state.skip_rows).drop_duplicates().to_csv(state.outdir/'08_quality_control'/'skipped_analyses.csv',index=False)

def make_gallery(state):
    rows=[]
    for rec in state.figure_registry:
        rows.append(f"<div class='card'><h3>{rec.figure_id}: {rec.title}</h3><p><strong>Section:</strong> {rec.section} | <strong>Type:</strong> {rec.analysis_type} | <strong>N:</strong> {rec.n}</p><a href='../{rec.png_file}'><img src='../{rec.png_file}' alt='{rec.title}'></a><p>{rec.qa_notes}</p><p><a href='../{rec.png_file}'>PNG</a> | <a href='../{rec.pdf_file}'>PDF</a> | <a href='../{rec.svg_file}'>SVG</a> | <a href='../{rec.caption_file}'>Caption</a></p></div>")
    html=f"<html><head><meta charset='utf-8'><title>Poison figure gallery</title><style>body{{font-family:Arial,sans-serif;margin:20px;background:#fafbfc;}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;}}.card{{background:white;padding:12px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.08);}}img{{width:100%;border:1px solid #e5e7eb;}}h1,h3{{color:#1d3557;}}</style></head><body><h1>Poison figure gallery</h1><div class='grid'>{''.join(rows)}</div></body></html>"
    (state.outdir/'06_interactive_html'/'figure_gallery.html').write_text(html,encoding='utf-8')

def draw_flow_diagram(ax, counts):
    ax.axis('off'); boxes=[(0.05,0.70,0.28,0.18,f"All records\nN = {counts['all']:,}"),(0.38,0.70,0.25,0.18,f"Non-missing poisoning type\nN = {counts['poison_type']:,}"),(0.68,0.70,0.25,0.18,f"Temporal plot eligible\nN = {counts['date_valid']:,}"),(0.20,0.35,0.25,0.18,f"Outcome classified\nN = {counts['outcome_known']:,}"),(0.55,0.35,0.25,0.18,f"Model complete-case\nN = {counts['model_cc']:,}")]
    for x,y,w,h,txt in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.02',facecolor='#f8fafc',edgecolor='#4e79a7',linewidth=1.2)); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10)
    for start,end in [((0.33,0.79),(0.38,0.79)),((0.63,0.79),(0.68,0.79)),((0.50,0.70),(0.32,0.53)),((0.75,0.70),(0.67,0.53))]: ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,linewidth=1.0,color='#4e79a7'))

def plot_rate_with_ci(ax, table, rate_col, low_col, high_col, label_col, color='#1d3557'):
    if table is None or len(table)==0 or rate_col not in table.columns:
        ax.axis('off')
        ax.text(0.5,0.5,'No estimable rates',ha='center',va='center')
        return
    t=table.copy()
    for c in [rate_col,low_col,high_col]:
        t[c]=pd.to_numeric(t[c],errors='coerce')
    t=t.dropna(subset=[rate_col,low_col,high_col,label_col])
    if t.empty:
        ax.axis('off')
        ax.text(0.5,0.5,'No estimable rates',ha='center',va='center')
        return
    y=np.arange(len(t))
    x=t[rate_col].to_numpy(dtype=float)
    left=np.maximum(0, x-t[low_col].to_numpy(dtype=float))
    right=np.maximum(0, t[high_col].to_numpy(dtype=float)-x)
    ax.errorbar(x,y,xerr=[left,right],fmt='o',color=color,ecolor='#7f8c8d',capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([wrap(v,22) for v in t[label_col]])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%'))
    despine_and_tidy(ax,'x')

def forest_plot(ax, tbl, title):
    # Fully defensive forest plot: empty/unestimable model tables must not stop the pipeline.
    if not isinstance(tbl, pd.DataFrame) or tbl.empty or 'or' not in tbl.columns:
        ax.axis('off')
        ax.text(0.5,0.5,'No estimable effects',ha='center',va='center')
        ax.set_title(title)
        return
    t=tbl.copy()
    for col in ['or','low','high']:
        if col not in t.columns:
            t[col]=np.nan
        t[col]=pd.to_numeric(t[col],errors='coerce')
    if 'term_display' not in t.columns:
        t['term_display']=t.get('term',pd.Series(['Effect']*len(t),index=t.index)).astype(str)
    t=t.dropna(subset=['or'])
    if t.empty:
        ax.axis('off')
        ax.text(0.5,0.5,'No estimable effects',ha='center',va='center')
        ax.set_title(title)
        return
    # Penalized models do not have confidence intervals; draw point estimates only in that case.
    y=np.arange(len(t))
    low=t['low'].where(t['low'].notna(),t['or'])
    high=t['high'].where(t['high'].notna(),t['or'])
    x=t['or'].to_numpy(dtype=float)
    left=np.maximum(0, x-low.to_numpy(dtype=float))
    right=np.maximum(0, high.to_numpy(dtype=float)-x)
    ax.errorbar(x,y,xerr=[left,right],fmt='o',color='#1d3557',ecolor='#6c757d',capsize=3)
    ax.axvline(1,color='#bc4749',linestyle='--',linewidth=1)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(v,26) for v in t['term_display']])
    ax.set_xlabel('Odds ratio (log scale)')
    ax.set_title(title)
    despine_and_tidy(ax,'x')

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
    rows=[]; preds={'Male sex':(clean['sex']=='Male').astype(float),'Rural residence':(clean['living_area']=='Rural').astype(float),'Delayed presentation >6 h':clean.get('delayed_presentation',pd.Series(index=clean.index,dtype=float)),'Low GCS <13':clean.get('low_gcs',pd.Series(index=clean.index,dtype=float)),'Hypoxia (SpO2 <94%)':clean.get('hypoxia',pd.Series(index=clean.index,dtype=float)),'Hypotension (SBP <90)':clean.get('hypotension',pd.Series(index=clean.index,dtype=float)),'Shock':clean.get('shock_binary',pd.Series(index=clean.index,dtype=float)),'Convulsion':clean.get('convulsion_binary',pd.Series(index=clean.index,dtype=float)),'High-risk poison group':clean.get('high_risk_poison_group',pd.Series(index=clean.index,dtype=float))}
    for name,series in preds.items():
        tmp=pd.DataFrame({'x':series,'y':clean[outcome_col]}).dropna();
        if tmp['x'].nunique()<2 or tmp['y'].sum()<5: continue
        try:
            if OPTIONAL_IMPORTS['statsmodels']:
                model=sm.Logit(tmp['y'],sm.add_constant(tmp[['x']],has_constant='add')).fit(disp=False); coef=model.params['x']; se=model.bse['x']
                rows.append({'term':name,'term_display':name,'or':math.exp(coef),'low':math.exp(coef-1.96*se),'high':math.exp(coef+1.96*se),'pvalue':float(model.pvalues['x']),'n':int(tmp.shape[0])})
        except Exception as e: logger.warning(f'Univariable model failed for {name}: {e}')
    return pd.DataFrame(rows,columns=expected_cols)

def fit_adjusted_logistic(clean,outcome_col,impute,logger):
    X_raw,y,predictors,meta=build_model_matrix(clean,outcome_col)
    # The outcome itself must never be imputed. Drop rows with unknown outcome first.
    base_df=pd.concat([X_raw,y.rename(outcome_col)],axis=1)
    base_df=base_df[pd.to_numeric(base_df[outcome_col],errors='coerce').notna()].copy()
    base_df[outcome_col]=pd.to_numeric(base_df[outcome_col],errors='coerce').astype(float)
    event_n=int(base_df[outcome_col].sum()) if not base_df.empty else 0
    meta['event_n']=event_n

    def split_model_columns(frame):
        """Strictly separate numeric and categorical predictors.
        A predictor is numeric only when all observed values can be converted to numbers.
        This prevents fields such as sex/site/poison type from entering median imputation.
        """
        numeric=[]; categorical=[]
        for col in frame.columns:
            observed=frame[col].notna().sum()
            coerced=pd.to_numeric(frame[col],errors='coerce')
            numeric_observed=coerced.notna().sum()
            if observed>0 and numeric_observed==observed:
                numeric.append(col)
            else:
                categorical.append(col)
        return numeric,categorical

    if base_df.empty:
        meta['status']='no_observed_outcome'
        return pd.DataFrame(),meta,None,None,None
    if impute:
        model_df=base_df.copy()
    else:
        model_df=base_df.dropna().copy()
    y_model=model_df[outcome_col].astype(float)
    X_model=model_df.drop(columns=[outcome_col])
    num_cols,cat_cols=split_model_columns(X_model)
    dummy_counts=sum(max(1,X_model[c].nunique(dropna=True)-1) for c in cat_cols if c in X_model.columns)
    est_terms=len([c for c in num_cols if c in X_model.columns])+dummy_counts
    # Keep adjusted models stable and fast; this is a descriptive manuscript model, not a prediction contest.
    max_terms=min(10,max(2,event_n//12))
    if est_terms>max_terms:
        keep=[p for p in ['age_years','sex','poison_type_major','delayed_presentation','low_gcs','hypoxia','hypotension','shock_binary','high_risk_poison_group'] if p in X_model.columns][:max_terms]
        X_model=X_model[keep]
        num_cols,cat_cols=split_model_columns(X_model)
        meta['reduced_predictors']=keep
    if impute and not X_model.empty and OPTIONAL_IMPORTS['sklearn']:
        Xi=X_model.copy()
        for c in cat_cols:
            Xi[c]=Xi[c].map(normalize_text).fillna('Missing').astype(str)
        if num_cols:
            Xi[num_cols]=Xi[num_cols].apply(pd.to_numeric,errors='coerce')
            Xi[num_cols]=SimpleImputer(strategy='median').fit_transform(Xi[num_cols])
        X_model=Xi
    else:
        X_model=X_model.dropna().copy()
        y_model=y_model.loc[X_model.index]
        num_cols,cat_cols=split_model_columns(X_model)
        if num_cols:
            X_model[num_cols]=X_model[num_cols].apply(pd.to_numeric,errors='coerce')
        for c in cat_cols:
            X_model[c]=X_model[c].map(normalize_text).fillna('Missing').astype(str)
    if y_model.sum()<10 or y_model.nunique()<2:
        meta['status']='insufficient_events'
        return pd.DataFrame(),meta,None,None,None
    X_design=pd.get_dummies(X_model,drop_first=True,dtype=float)
    X_design=X_design.replace([np.inf,-np.inf],np.nan).dropna(axis=1,how='any')
    sparse_ok=[c for c in X_design.columns if X_design[c].sum()>=10 or (X_design[c]==0).sum()>=10]
    X_design=X_design[sparse_ok]
    meta['design_terms']=X_design.columns.tolist()
    if X_design.empty:
        meta['status']='empty_design'
        return pd.DataFrame(),meta,None,None,None
    # Try statsmodels only for small, well-conditioned designs. Otherwise use penalized logistic regression.
    try_statsmodels = OPTIONAL_IMPORTS['statsmodels'] and X_design.shape[1] <= 12 and X_design.shape[0] <= 5000
    if try_statsmodels:
        try:
            res=sm.Logit(y_model,sm.add_constant(X_design,has_constant='add')).fit(disp=False,maxiter=100)
            conf=res.conf_int(); rows=[]
            for term in X_design.columns:
                coef=res.params[term]; low,high=conf.loc[term]
                rows.append({'term':term,'term_display':term.replace('_',' '),'or':math.exp(coef),'low':math.exp(low),'high':math.exp(high),'pvalue':float(res.pvalues.get(term,np.nan))})
            meta['status']='fitted_statsmodels'
            return pd.DataFrame(rows),meta,res,X_design,y_model
        except Exception as e:
            meta['statsmodels_failure']=str(e)
            logger.warning(f"Statsmodels logistic failed for {outcome_col} ({'imputed' if impute else 'complete-case'}): {e}")
    try:
        if OPTIONAL_IMPORTS['sklearn']:
            clf=LogisticRegression(max_iter=500,penalty='l2',solver='liblinear')
            clf.fit(X_design,y_model)
            rows=[{'term':term,'term_display':term.replace('_',' '),'or':math.exp(c),'low':np.nan,'high':np.nan,'pvalue':np.nan} for term,c in zip(X_design.columns,clf.coef_[0])]
            meta['status']='fitted_sklearn_penalized'
            return pd.DataFrame(rows),meta,clf,X_design,y_model
    except Exception as e:
        meta['sklearn_failure']=str(e)
        logger.warning(f'Penalized logistic fallback failed for {outcome_col}: {e}')
    meta['status']='failed'
    return pd.DataFrame(),meta,None,None,None

def cross_validated_roc_and_calibration(model_X, model_y, logger):
    if not OPTIONAL_IMPORTS['sklearn'] or model_X is None or model_y is None or model_y.sum()<30 or model_y.nunique()<2 or model_X.shape[0]<200: return None,None
    try:
        clf=LogisticRegression(max_iter=500,solver='liblinear'); cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); probs=cross_val_predict(clf,model_X,model_y,cv=cv,method='predict_proba')[:,1]; auc=roc_auc_score(model_y,probs); fpr,tpr,_=roc_curve(model_y,probs); frac_pos,mean_pred=calibration_curve(model_y,probs,n_bins=10)
        return {'auc':auc,'diag':pd.DataFrame({'fpr':fpr,'tpr':tpr}),'calib':pd.DataFrame({'mean_pred':mean_pred,'frac_pos':frac_pos}),'probs':probs}, pd.DataFrame({'mean_pred':mean_pred,'frac_pos':frac_pos})
    except Exception as e: logger.warning(f'Cross-validated ROC/calibration failed: {e}'); return None,None

def generate_main_figures(clean,state,logger):
    results={}; plot_df=clean.copy(); major_poison=clean['poison_type'].value_counts().head(10).index.tolist(); plot_df['poison_type_plot']=plot_df['poison_type'].where(plot_df['poison_type'].isin(major_poison),'Other')
    # Figure 1
    fig=plt.figure(figsize=(16,11),constrained_layout=True); gs=GridSpec(3,2,figure=fig,height_ratios=[1.05,1,1]); fig.suptitle('Figure 1. Cohort profile and demographics',x=0.01,ha='left',fontsize=18,fontweight='bold')
    axA=fig.add_subplot(gs[0,0]); counts={'all':len(clean),'poison_type':int(clean['poison_type'].notna().sum()),'date_valid':int(clean['admission_date_for_plots'].notna().sum()),'outcome_known':int((clean['outcome_category']!='Missing/unknown').sum()),'model_cc':int(clean[[c for c in ['death_flag','age_years','sex','poison_type_major','low_gcs','hypoxia','hypotension'] if c in clean.columns]].dropna().shape[0])}; draw_flow_diagram(axA,counts); add_panel_letter(axA,'A'); axA.set_title('STROBE-style cohort overview',loc='left')
    axB=fig.add_subplot(gs[0,1]); site_counts=clean['study_site'].fillna('Missing').value_counts().sort_values(ascending=False); axB.barh(range(len(site_counts)),site_counts.values,color='#1d3557'); axB.set_yticks(range(len(site_counts))); axB.set_yticklabels([wrap(v,12) for v in site_counts.index]); axB.invert_yaxis(); axB.set_xlabel('Patients'); axB.set_title('Study-site enrollment',loc='left'); despine_and_tidy(axB,'x'); add_panel_letter(axB,'B')
    axC=fig.add_subplot(gs[1:,0]); pyr=clean.dropna(subset=['age_group','sex']).copy(); pyramid=pyr.groupby(['age_group','sex']).size().unstack(fill_value=0).reindex(index=clean['age_group'].cat.categories); males=-pyramid.get('Male',pd.Series(index=pyramid.index,data=0)); females=pyramid.get('Female',pd.Series(index=pyramid.index,data=0)); y=np.arange(len(pyramid.index)); axC.barh(y,males,color='#457b9d',label='Male'); axC.barh(y,females,color='#e76f51',label='Female'); axC.set_yticks(y); axC.set_yticklabels(pyramid.index.astype(str)); axC.set_xlabel('Patients'); axC.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{abs(int(v))}')); axC.set_title('Age–sex pyramid',loc='left'); axC.legend(frameon=False,loc='lower right'); despine_and_tidy(axC,'x'); add_panel_letter(axC,'C')
    axD=fig.add_subplot(gs[1,1]); comp=pd.DataFrame({'Sex':clean['sex'].fillna('Missing').value_counts(normalize=True),'Residence':clean['living_area'].fillna('Missing').value_counts(normalize=True),'Presentation':clean['presentation_area'].fillna('Missing').value_counts(normalize=True)}).fillna(0); bottom=np.zeros(len(comp.columns))
    for i,cat in enumerate(comp.index): vals=comp.loc[cat].values; axD.bar(comp.columns,vals,bottom=bottom,color=CATEGORICAL_BASE[i%len(CATEGORICAL_BASE)],label=wrap(cat,16)); bottom+=vals
    axD.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axD.set_title('Demographic composition',loc='left'); axD.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(axD,'y'); add_panel_letter(axD,'D')
    axE=fig.add_subplot(gs[2,1]); lab_candidates=[c for c in ['wbc_count_mm3','creatinine_mg_dl','ph','hco3_mmol_l'] if c in clean.columns]; completeness=pd.Series({'Outcome':clean['outcome_category'].replace('Missing/unknown',np.nan).notna().mean(),'Admission date':clean['admission_date_for_plots'].notna().mean(),'Poison type':clean['poison_type'].notna().mean(),'Component':clean['component'].notna().mean(),'Vitals':clean[[c for c in ['gcs','spo2','sbp'] if c in clean.columns]].notna().mean().mean(),'Labs':clean[lab_candidates].notna().mean().mean() if lab_candidates else np.nan}); axE.barh(range(len(completeness)),completeness.values,color='#6c757d'); axE.set_yticks(range(len(completeness))); axE.set_yticklabels(completeness.index); axE.set_xlim(0,1); axE.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axE.set_title('Key-variable completeness',loc='left'); despine_and_tidy(axE,'x'); add_panel_letter(axE,'E')
    save_figure(fig,state,'figure_1_cohort_demographics','Figure 1. Cohort profile and demographics','main','07_multiplanel_assembled_figures','A, STROBE-style cohort overview showing records available for descriptive, temporal, outcome, and modeling analyses. B, study-site enrollment ranked by sample size. C, age–sex pyramid. D, compositional distributions for sex, residence, and presentation area. E, mini-panel showing completeness of major analytical domains. Missing values are retained in the cleaned dataset but are not conflated with observed No.',variables=['study_site','age_group','sex','living_area','presentation_area','outcome_category'],analysis_type='multiplanel descriptive',n=len(clean),is_main=True); state.mark_coverage('fig1_cohort_demographics','generated','figure_1_cohort_demographics')
    # Figure 2
    fig=plt.figure(figsize=(16,12),constrained_layout=True); gs=GridSpec(3,2,figure=fig); fig.suptitle('Figure 2. Poisoning epidemiology',x=0.01,ha='left',fontsize=18,fontweight='bold')
    axA=fig.add_subplot(gs[0,0]); pt=clean['poison_type'].value_counts().head(12).sort_values(ascending=True); axA.barh(range(len(pt)),pt.values,color='#1d3557'); axA.set_yticks(range(len(pt))); axA.set_yticklabels([wrap(i,18) for i in pt.index]); axA.set_xlabel('Patients'); axA.set_title('Ranked poisoning types',loc='left'); despine_and_tidy(axA,'x'); add_panel_letter(axA,'A')
    axB=fig.add_subplot(gs[0,1]); age_type=pd.crosstab(plot_df['age_group'],plot_df['poison_type_plot'],normalize='index').fillna(0); age_type=age_type[[c for c in age_type.columns if c in age_type.sum().sort_values(ascending=False).index[:8]]]; age_type.plot(kind='bar',stacked=True,ax=axB,color=CATEGORICAL_BASE[:len(age_type.columns)],width=0.85); axB.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axB.set_xlabel(''); axB.set_ylabel('Within-age-group proportion'); axB.set_title('Poisoning type by age group',loc='left'); axB.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(axB,'y'); add_panel_letter(axB,'B')
    axC=fig.add_subplot(gs[1,0]); sex_type=pd.crosstab(plot_df['sex'],plot_df['poison_type_plot'],normalize='index').fillna(0); sex_type.plot(kind='bar',stacked=True,ax=axC,color=CATEGORICAL_BASE[:len(sex_type.columns)]); axC.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axC.set_xlabel(''); axC.set_ylabel('Within-sex proportion'); axC.set_title('Poisoning type by sex',loc='left'); axC.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(axC,'y'); add_panel_letter(axC,'C')
    axD=fig.add_subplot(gs[1,1]); site_type=pd.crosstab(plot_df['study_site'],plot_df['poison_type_plot'],normalize='index').fillna(0); site_type=site_type[site_type.sum().sort_values(ascending=False).index[:8]] if site_type.shape[1]>8 else site_type; sns.heatmap(site_type,cmap=sns.light_palette('#1d3557',as_cmap=True),ax=axD,cbar_kws={'format':FuncFormatter(lambda x,_:f'{100*x:.0f}%')}); axD.set_title('Poisoning type by study site',loc='left'); add_panel_letter(axD,'D')
    axE=fig.add_subplot(gs[2,0]); comp=clean['component'].value_counts().head(12).sort_values(ascending=True); axE.barh(range(len(comp)),comp.values,color='#4e79a7'); axE.set_yticks(range(len(comp))); axE.set_yticklabels([wrap(i,18) for i in comp.index]); axE.set_xlabel('Patients'); axE.set_title('Top specific components',loc='left'); despine_and_tidy(axE,'x'); add_panel_letter(axE,'E')
    axF=fig.add_subplot(gs[2,1]); heat=pd.crosstab(plot_df['poison_type_plot'],plot_df['component_major']).astype(float); heat=heat.loc[heat.sum(axis=1).sort_values(ascending=False).index[:10],heat.sum(axis=0).sort_values(ascending=False).index[:10]]; row_ord,col_ord=cluster_order(heat); sns.heatmap(heat.iloc[row_ord,col_ord],cmap=sns.light_palette('#5a189a',as_cmap=True),ax=axF); axF.set_title('Poisoning type × component heatmap',loc='left'); axF.set_xlabel(''); axF.set_ylabel(''); add_panel_letter(axF,'F')
    save_figure(fig,state,'figure_2_poisoning_epidemiology','Figure 2. Poisoning epidemiology','main','07_multiplanel_assembled_figures','Panel set summarising poisoning epidemiology. A, ranked poisoning types. B–D, denominator-aware distributions of poisoning type across age groups, sex, and study sites. E, top harmonised component categories. F, clustered cross-tabulation of harmonised poisoning type by specific component, filtered to the most informative strata for readability.',variables=['poison_type','age_group','sex','study_site','component'],analysis_type='multiplanel descriptive',n=len(clean),is_main=True); state.mark_coverage('fig2_poisoning_epidemiology','generated','figure_2_poisoning_epidemiology')
    # Figure 3
    fig=plt.figure(figsize=(17,12),constrained_layout=True); gs=GridSpec(3,2,figure=fig); fig.suptitle('Figure 3. Temporal and site patterns',x=0.01,ha='left',fontsize=18,fontweight='bold'); dt=plot_df.dropna(subset=['admission_date_for_plots']).copy()
    axA=fig.add_subplot(gs[0,0]); monthly=dt.groupby(dt['admission_date_for_plots'].dt.to_period('M')).size(); axA.plot(monthly.index.to_timestamp(),monthly.values,marker='o',color='#1d3557',linewidth=2); axA.set_title('Monthly admission trend',loc='left'); axA.set_ylabel('Admissions'); axA.annotate(f"Invalid/implausible admission dates excluded from plotting: {(clean['admission_date_for_plots'].isna() & clean['admission_date'].notna()).sum()}",xy=(0.01,0.02),xycoords='axes fraction',fontsize=8); despine_and_tidy(axA,'y'); add_panel_letter(axA,'A')
    axB=fig.add_subplot(gs[0,1]); top_pt=clean['poison_type'].value_counts().head(5).index; m2=dt[dt['poison_type'].isin(top_pt)].groupby([dt['admission_date_for_plots'].dt.to_period('M'),'poison_type']).size().unstack(fill_value=0)
    for i,c in enumerate(m2.columns): axB.plot(m2.index.to_timestamp(),m2[c],marker='o',linewidth=1.8,label=c,color=CATEGORICAL_BASE[i%len(CATEGORICAL_BASE)])
    axB.set_title('Monthly trend by major poisoning type',loc='left'); axB.legend(frameon=False,ncol=2,fontsize=8); despine_and_tidy(axB,'y'); add_panel_letter(axB,'B')
    axC=fig.add_subplot(gs[1,0]); cal=dt.groupby([dt['admission_date_for_plots'].dt.month,dt['admission_date_for_plots'].dt.day]).size().unstack(fill_value=0).reindex(index=range(1,13),fill_value=0); sns.heatmap(cal,cmap=sns.light_palette('#1d3557',as_cmap=True),ax=axC,cbar=False); axC.set_yticklabels([calendar.month_abbr[i] for i in range(1,13)],rotation=0); axC.set_title('Calendar heatmap of admissions',loc='left'); axC.set_xlabel('Day of month'); axC.set_ylabel(''); add_panel_letter(axC,'C')
    axD=fig.add_subplot(gs[1,1]); dow=dt['admission_day_of_week'].value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']); axD.bar(range(len(dow)),dow.values,color='#4e79a7'); axD.set_xticks(range(len(dow))); axD.set_xticklabels([d[:3] for d in dow.index],rotation=0); axD.set_title('Day-of-week pattern',loc='left'); axD.set_ylabel('Admissions'); despine_and_tidy(axD,'y'); add_panel_letter(axD,'D')
    axE=fig.add_subplot(gs[2,0]); season_tab=pd.crosstab(dt['season'],dt['poison_type_plot'],normalize='index').fillna(0); season_tab=season_tab[season_tab.sum().sort_values(ascending=False).index[:6]] if season_tab.shape[1]>6 else season_tab; season_tab.plot(kind='bar',stacked=True,ax=axE,color=CATEGORICAL_BASE[:season_tab.shape[1]]); axE.set_title('Seasonality by poisoning type',loc='left'); axE.set_xlabel(''); axE.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axE.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(axE,'y'); add_panel_letter(axE,'E')
    axF=fig.add_subplot(gs[2,1]); monthly_site=dt.groupby([dt['admission_date_for_plots'].dt.to_period('M'),'study_site']).size().unstack(fill_value=0); top_sites=monthly_site.sum().sort_values(ascending=False).index[:6]
    for i,site in enumerate(top_sites): axF.plot(monthly_site.index.to_timestamp(),monthly_site[site],linewidth=1.4,label=site,color=CATEGORICAL_BASE[i])
    axF.set_title('Site-specific temporal trends',loc='left'); axF.legend(frameon=False,ncol=2,fontsize=8); despine_and_tidy(axF,'y'); add_panel_letter(axF,'F')
    caption=f"Temporal analyses use admission dates within the inferred plausible study window ({clean['admission_date_for_plots'].min().date()} to {clean['admission_date_for_plots'].max().date()}); implausible year-1900 artifacts and dates outside the plausible window are flagged and excluded from plotting only. Panels summarise overall monthly admissions, monthly trends by major poisoning type, a calendar heatmap, weekday distribution, seasonal type distribution, and site-specific temporal profiles."
    save_figure(fig,state,'figure_3_temporal_site_patterns','Figure 3. Temporal and site patterns','main','07_multiplanel_assembled_figures',caption,variables=['admission_date_for_plots','poison_type','study_site','season'],analysis_type='multiplanel temporal',n=len(dt),is_main=True); state.mark_coverage('fig3_temporal_site','generated','figure_3_temporal_site_patterns')
    # Figure 4
    fig=plt.figure(figsize=(17,12),constrained_layout=True); gs=GridSpec(3,2,figure=fig); fig.suptitle('Figure 4. Clinical presentation and severity',x=0.01,ha='left',fontsize=18,fontweight='bold'); symptom_bin_cols=[c for c in ['sym_vomited_immediately','sym_fever','sym_vomiting','sym_diarrhoea','sym_abdominal_pain','sym_abdominal_distension','sym_cough','sym_shortness_of_breath','sym_heart_burn','sym_oral_ulcers','sym_leg_swelling','sym_reduced_urine','sym_jaundice','sym_unconsciousness','sym_convulsion','sym_chest_pain','sym_bleeding','sym_shock'] if c in clean.columns]
    axA=fig.add_subplot(gs[0,0]); sym_prev=clean[symptom_bin_cols].mean().sort_values().dropna(); axA.barh(range(len(sym_prev)),sym_prev.values,color='#1d3557'); axA.set_yticks(range(len(sym_prev))); axA.set_yticklabels([wrap(c.replace('sym_','').replace('_',' '),20) for c in sym_prev.index]); axA.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axA.set_title('Ranked symptom prevalence',loc='left'); despine_and_tidy(axA,'x'); add_panel_letter(axA,'A')
    axB=fig.add_subplot(gs[0,1]); heat_rows=plot_df[plot_df['poison_type_plot']!='Other'].copy(); heat_tbl=heat_rows.groupby('poison_type_plot')[symptom_bin_cols].mean().T; heat_tbl=heat_tbl.loc[heat_tbl.mean(axis=1).sort_values(ascending=False).index[:12],:]; row_ord,col_ord=cluster_order(heat_tbl); sns.heatmap(heat_tbl.iloc[row_ord,col_ord],cmap=sns.light_palette('#bc4749',as_cmap=True),ax=axB,cbar_kws={'format':FuncFormatter(lambda x,_:f'{100*x:.0f}%')}); axB.set_title('Poisoning type × symptom heatmap',loc='left'); add_panel_letter(axB,'B')
    axC=fig.add_subplot(gs[1,0]); simple_upset_like(axC,clean[symptom_bin_cols],top_n=8); add_panel_letter(axC,'C')
    axD=fig.add_subplot(gs[1,1]); symptom_network_plot(axD,clean[symptom_bin_cols]); add_panel_letter(axD,'D')
    axE=fig.add_subplot(gs[2,0]); severity_prev=plot_df.groupby('poison_type_plot')[['low_gcs','hypoxia','hypotension']].mean().sort_values('low_gcs',ascending=False); severity_prev=severity_prev.loc[[i for i in severity_prev.index if i!='Other'][:8]]; width=0.25; x=np.arange(len(severity_prev.index))
    for i,col in enumerate(['low_gcs','hypoxia','hypotension']): axE.bar(x+(i-1)*width,severity_prev[col].values,width=width,label=col.replace('_',' '),color=CATEGORICAL_BASE[i])
    axE.set_xticks(x); axE.set_xticklabels([wrap(v,14) for v in severity_prev.index],rotation=0); axE.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axE.set_title('Severity indicators by poisoning type',loc='left'); axE.legend(frameon=False); despine_and_tidy(axE,'y'); add_panel_letter(axE,'E')
    axF=fig.add_subplot(gs[2,1]); out_order=['Survived uncomplicated','Complication','Absconded/DORB','Death','Missing/unknown']; sb=clean.groupby('outcome_category')['symptom_burden_score'].median().reindex(out_order); iqr1=clean.groupby('outcome_category')['symptom_burden_score'].quantile(0.25).reindex(out_order); iqr3=clean.groupby('outcome_category')['symptom_burden_score'].quantile(0.75).reindex(out_order); axF.errorbar(sb.values,range(len(sb)),xerr=[sb.values-iqr1.values,iqr3.values-sb.values],fmt='o',color='#1d3557'); axF.set_yticks(range(len(sb))); axF.set_yticklabels([wrap(v,18) for v in sb.index]); axF.set_xlabel('Median symptom burden (IQR)'); axF.set_title('Symptom burden by outcome',loc='left'); despine_and_tidy(axF,'x'); add_panel_letter(axF,'F')
    save_figure(fig,state,'figure_4_clinical_severity','Figure 4. Clinical presentation and severity','main','07_multiplanel_assembled_figures','Clinical presentation panel set including ranked symptom prevalence, clustered symptom-pattern heatmaps across poisoning types, top symptom combinations, symptom co-occurrence network, key severity indicators by poisoning type, and symptom burden across outcome strata. Symptom percentages use non-missing denominators for each feature.',variables=symptom_bin_cols+['low_gcs','hypoxia','hypotension','outcome_category'],analysis_type='multiplanel clinical',n=len(clean),is_main=True); state.mark_coverage('fig4_clinical_severity','generated','figure_4_clinical_severity')
    # Figure 5
    fig=plt.figure(figsize=(17,12),constrained_layout=True); gs=GridSpec(3,2,figure=fig); fig.suptitle('Figure 5. Treatment and outcomes',x=0.01,ha='left',fontsize=18,fontweight='bold'); treat_cols=[c for c in ['oxygen_any','ng_suction','dialysis_any','ventilation_support','operation_support'] if c in clean.columns]
    axA=fig.add_subplot(gs[0,0]); treat_heat=plot_df.groupby('poison_type_plot')[treat_cols].mean().loc[[i for i in plot_df['poison_type_plot'].value_counts().index if i!='Other'][:10]]; sns.heatmap(treat_heat,cmap=sns.light_palette('#1b4332',as_cmap=True),ax=axA,cbar_kws={'format':FuncFormatter(lambda x,_:f'{100*x:.0f}%')}); axA.set_title('Treatment/support heatmap by poisoning type',loc='left'); add_panel_letter(axA,'A')
    axB=fig.add_subplot(gs[0,1]); inter=treat_heat.T; inter.plot(kind='bar',ax=axB,color=CATEGORICAL_BASE[:inter.shape[1]]); axB.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axB.set_title('Supportive intervention rates',loc='left'); axB.set_xlabel(''); axB.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left',fontsize=7); despine_and_tidy(axB,'y'); add_panel_letter(axB,'B')
    axC=fig.add_subplot(gs[1,0]); out_counts=clean['outcome_category'].value_counts().reindex(['Survived uncomplicated','Complication','Absconded/DORB','Death','Missing/unknown']).fillna(0); out_props=out_counts/out_counts.sum(); axC.bar(range(len(out_props)),out_props.values,color=CATEGORICAL_BASE[:len(out_props)]); axC.set_xticks(range(len(out_props))); axC.set_xticklabels([wrap(v,18) for v in out_props.index],rotation=0); axC.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axC.set_title('Outcome distribution (overall denominator)',loc='left'); [axC.text(i,prop+0.01,f'n={int(cnt)}',ha='center',va='bottom',fontsize=8) for i,(cnt,prop) in enumerate(zip(out_counts.values,out_props.values))]; despine_and_tidy(axC,'y'); add_panel_letter(axC,'C')
    axD=fig.add_subplot(gs[1,1]); death_tbl=[]
    for p,sub in plot_df.groupby('poison_type_plot'):
        if p=='Other' or len(sub)<30: continue
        k=int((sub['death_flag']==1).sum()); n=int(sub['death_flag'].notna().sum()); ph,lo,hi=wilson_ci(k,n); death_tbl.append({'poison_type':p,'n':n,'k':k,'rate':ph,'low':lo,'high':hi})
    death_tbl=pd.DataFrame(death_tbl).sort_values('rate'); plot_rate_with_ci(axD,death_tbl,'rate','low','high','poison_type'); axD.set_title('Death rate by poisoning type',loc='left'); add_panel_letter(axD,'D')
    axE=fig.add_subplot(gs[2,0]); severe_tbl=[]
    for p,sub in plot_df.groupby('poison_type_plot'):
        if p=='Other' or len(sub)<30: continue
        k=int(sub['severe_outcome'].sum()); n=len(sub); ph,lo,hi=wilson_ci(k,n); severe_tbl.append({'poison_type':p,'rate':ph,'low':lo,'high':hi})
    severe_tbl=pd.DataFrame(severe_tbl).sort_values('rate'); plot_rate_with_ci(axE,severe_tbl,'rate','low','high','poison_type',color='#bc4749'); axE.set_title('Severe outcome rate by poisoning type',loc='left'); add_panel_letter(axE,'E')
    axF=fig.add_subplot(gs[2,1]); ti=clean.groupby('outcome_category')['treatment_intensity_score'].median().reindex(out_order); ti1=clean.groupby('outcome_category')['treatment_intensity_score'].quantile(0.25).reindex(out_order); ti3=clean.groupby('outcome_category')['treatment_intensity_score'].quantile(0.75).reindex(out_order); axF.errorbar(ti.values,range(len(ti)),xerr=[ti.values-ti1.values,ti3.values-ti.values],fmt='o',color='#1b4332'); axF.set_yticks(range(len(ti))); axF.set_yticklabels([wrap(v,18) for v in ti.index]); axF.set_xlabel('Median treatment intensity (IQR)'); axF.set_title('Treatment intensity by outcome',loc='left'); despine_and_tidy(axF,'x'); add_panel_letter(axF,'F')
    save_figure(fig,state,'figure_5_treatment_outcomes','Figure 5. Treatment and outcomes','main','07_multiplanel_assembled_figures','Treatment and outcome overview including poisoning-type-specific supportive-treatment heatmaps, intervention rate summaries, denominator-aware outcome distributions, and Wilson 95% confidence intervals for death and severe outcome rates. Outcome rates use observed denominators within each poisoning type and do not rely on positive-event counts alone.',variables=treat_cols+['outcome_category','death_flag','severe_outcome','treatment_intensity_score'],analysis_type='multiplanel treatment/outcome',n=len(clean),is_main=True); state.mark_coverage('fig5_treatment_outcomes','generated','figure_5_treatment_outcomes')
    # Figure 6
    death_uni=univariable_or_table(clean,'death_flag',logger); death_adj_cc,death_meta_cc,death_model_cc,death_X_cc,death_y_cc=fit_adjusted_logistic(clean,'death_flag',impute=False,logger=logger); death_adj_imp,death_meta_imp,death_model_imp,death_X_imp,death_y_imp=fit_adjusted_logistic(clean,'death_flag',impute=True,logger=logger)
    severe_uni=univariable_or_table(clean,'severe_outcome',logger); severe_adj_cc,severe_meta_cc,severe_model_cc,severe_X_cc,severe_y_cc=fit_adjusted_logistic(clean,'severe_outcome',impute=False,logger=logger); severe_adj_imp,severe_meta_imp,severe_model_imp,severe_X_imp,severe_y_imp=fit_adjusted_logistic(clean,'severe_outcome',impute=True,logger=logger)
    results.update({'death_uni':death_uni,'death_adj_cc':death_adj_cc,'death_adj_imp':death_adj_imp,'severe_uni':severe_uni,'severe_adj_cc':severe_adj_cc,'severe_adj_imp':severe_adj_imp,'death_meta_cc':death_meta_cc,'death_meta_imp':death_meta_imp,'severe_meta_cc':severe_meta_cc,'severe_meta_imp':severe_meta_imp})
    fig=plt.figure(figsize=(17,12),constrained_layout=True); gs=GridSpec(3,2,figure=fig); fig.suptitle('Figure 6. Risk factors and modeling',x=0.01,ha='left',fontsize=18,fontweight='bold')
    axA=fig.add_subplot(gs[0,0]); forest_plot(axA,death_uni,'Univariable odds ratios for death'); add_panel_letter(axA,'A')
    axB=fig.add_subplot(gs[0,1]); forest_plot(axB,death_adj_cc,'Adjusted odds ratios for death (complete-case)'); add_panel_letter(axB,'B')
    axC=fig.add_subplot(gs[1,0]); forest_plot(axC,severe_uni,'Univariable odds ratios for severe outcome'); add_panel_letter(axC,'C')
    axD=fig.add_subplot(gs[1,1]); forest_plot(axD,severe_adj_cc,'Adjusted odds ratios for severe outcome (complete-case)'); add_panel_letter(axD,'D')
    axE=fig.add_subplot(gs[2,0]); sens=death_adj_cc[['term','or']].merge(death_adj_imp[['term','or']],on='term',suffixes=('_cc','_imp'),how='outer') if not death_adj_cc.empty and not death_adj_imp.empty else pd.DataFrame()
    if sens.empty: axE.axis('off'); axE.text(0.5,0.5,'Sensitivity analysis unavailable',ha='center',va='center')
    else:
        sens=sens.head(10); axE.scatter(sens['or_cc'],sens['or_imp'],color='#1d3557'); maxv=np.nanmax(np.r_[sens['or_cc'].values,sens['or_imp'].values]) if np.isfinite(np.r_[sens['or_cc'].values,sens['or_imp'].values]).any() else 2; axE.plot([0,maxv],[0,maxv],linestyle='--',color='#bc4749',linewidth=1)
        for _,r in sens.iterrows(): axE.text(r['or_cc'],r['or_imp'],wrap(r['term'].replace('_',' '),14),fontsize=7)
        axE.set_xlabel('OR complete-case'); axE.set_ylabel('OR median-imputed'); axE.set_title('Sensitivity analysis: complete-case vs imputed death model',loc='left'); despine_and_tidy(axE,'both')
    add_panel_letter(axE,'E')
    axF=fig.add_subplot(gs[2,1]); roc_info,_=cross_validated_roc_and_calibration(death_X_imp,death_y_imp,logger) if death_X_imp is not None and death_y_imp is not None else (None,None)
    if roc_info is not None: axF.plot(roc_info['diag']['fpr'],roc_info['diag']['tpr'],color='#1d3557',lw=2,label=f"CV AUC = {roc_info['auc']:.2f}"); axF.plot([0,1],[0,1],linestyle='--',color='#adb5bd'); axF.set_xlabel('False-positive rate'); axF.set_ylabel('True-positive rate'); axF.legend(frameon=False); axF.set_title('Exploratory internally cross-validated ROC (death model)',loc='left'); despine_and_tidy(axF,'both'); qa_note='Exploratory ROC included with internal cross-validation.'
    else: axF.axis('off'); axF.text(0.5,0.6,'ROC/calibration not shown',ha='center',va='center',fontweight='bold'); axF.text(0.5,0.43,'Reasons may include insufficient complete observations, limited event structure, or unavailable sklearn support.',ha='center',va='center',wrap=True); qa_note='ROC/calibration omitted due to statistical appropriateness rule.'
    add_panel_letter(axF,'F')
    save_figure(fig,state,'figure_6_modeling','Figure 6. Risk factors and modeling','main','07_multiplanel_assembled_figures','Association panels display univariable and adjusted odds ratios for death and the composite severe outcome. Complete-case models are used for the main adjusted estimates; median-imputed models are shown as sensitivity analyses. Where appropriate, an internally cross-validated ROC panel is added and explicitly labelled exploratory. Observational associations should not be interpreted causally.',variables=['death_flag','severe_outcome','age_years','sex','poison_type_major','low_gcs','hypoxia','hypotension','shock_binary'],analysis_type='multiplanel modeling',n=len(clean),is_main=True,is_model=True,qa_notes=qa_note); state.mark_coverage('fig6_modeling','generated','figure_6_modeling')
    return results

def generate_supplementary_and_exploratory(clean,dq,metadata,model_results,state,logger):
    # Missingness matrix
    fig,ax=plt.subplots(figsize=(16,8)); vars_show=[c for c in ['study_site','sex','age_years','poison_type','component','admission_date_for_plots','time_to_presentation_hrs','gcs','spo2','sbp','creatinine_mg_dl','ph','outcome_category','severe_outcome'] if c in clean.columns]; miss=clean[vars_show].isna().astype(int).sample(min(len(clean),400),random_state=SEED)
    sns.heatmap(miss.T,cmap=sns.color_palette(['#f8f9fa','#1d3557'],as_cmap=True),cbar=False,ax=ax); ax.set_title('Supplementary Figure S1. Missingness matrix (sampled rows for display)',loc='left'); ax.set_xlabel('Record (sampled)'); ax.set_ylabel('Variable'); save_figure(fig,state,'supp_s1_missingness_matrix','Supplementary Figure S1. Missingness matrix','supplementary','04_supplementary_figures','Matrix of missingness for selected high-value analysis variables. To preserve readability, a random sample of records is shown on the x-axis while all rows remain in the cleaned dataset.',variables=vars_show,analysis_type='missingness heatmap',n=len(clean)) ; state.mark_coverage('supp_missingness_matrix','generated','supp_s1_missingness_matrix')
    # Data quality dashboard
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True); date_flags=clean['admission_date_quality_flag'].fillna('missing').value_counts(); axes[0,0].barh(range(len(date_flags)),date_flags.values,color='#1d3557'); axes[0,0].set_yticks(range(len(date_flags))); axes[0,0].set_yticklabels([wrap(v,22) for v in date_flags.index]); axes[0,0].set_title('Admission-date quality flags',loc='left'); despine_and_tidy(axes[0,0],'x')
    implausible_counts={c.replace('_flag',''):int((clean[c]=='implausible').sum()) for c in clean.columns if c.endswith('_flag') and c!='admission_date_quality_flag' and c!='ingestion_date_quality_flag'}; implausible_counts=pd.Series(implausible_counts).sort_values(ascending=False).head(12); axes[0,1].barh(range(len(implausible_counts)),implausible_counts.values,color='#bc4749'); axes[0,1].set_yticks(range(len(implausible_counts))); axes[0,1].set_yticklabels([wrap(v,18) for v in implausible_counts.index]); axes[0,1].set_title('Implausible numeric values flagged',loc='left'); despine_and_tidy(axes[0,1],'x')
    top_missing=dq.sort_values('missing_pct',ascending=False).head(15); axes[1,0].barh(range(len(top_missing)),top_missing['missing_pct'],color='#6c757d'); axes[1,0].set_yticks(range(len(top_missing))); axes[1,0].set_yticklabels([wrap(v,18) for v in top_missing['variable']]); axes[1,0].set_title('Highest missingness variables',loc='left'); axes[1,0].set_xlabel('% missing'); despine_and_tidy(axes[1,0],'x')
    axes[1,1].axis('off'); txt=f"Rows: {metadata['n_rows']}\nDate plotting window: {metadata['study_window']['start']} to {metadata['study_window']['end']}\nPII columns excluded from public outputs: {len(metadata['pii_excluded_columns'])}\nCanonical poisoning types: {len(metadata['canonical_poison_types'])}\nCanonical components: {len(metadata['canonical_components'])}"; axes[1,1].text(0.02,0.95,txt,va='top',fontsize=11,bbox=dict(boxstyle='round',fc='#f8fafc',ec='#1d3557')); save_figure(fig,state,'supp_s2_data_quality_dashboard','Supplementary Figure S2. Data-quality dashboard','supplementary','04_supplementary_figures','Dashboard summarising date quality flags, implausible numeric values, and missingness patterns. Implausible records are flagged and preserved in the cleaned dataset; temporal plots exclude implausible dates only from plotting.',variables=['admission_date_quality_flag']+[c for c in clean.columns if c.endswith('_flag')],analysis_type='quality dashboard',n=len(clean)); state.mark_coverage('supp_data_quality_dashboard','generated','supp_s2_data_quality_dashboard')
    # All poison categories
    fig,ax=plt.subplots(figsize=(10,8)); vc=clean['poison_type'].value_counts().sort_values(); ax.barh(range(len(vc)),vc.values,color='#4e79a7'); ax.set_yticks(range(len(vc))); ax.set_yticklabels([wrap(v,18) for v in vc.index]); ax.set_title('Supplementary Figure S3. All poisoning categories',loc='left'); ax.set_xlabel('Patients'); despine_and_tidy(ax,'x'); save_figure(fig,state,'supp_s3_all_poison_categories','Supplementary Figure S3. All poisoning categories','supplementary','04_supplementary_figures','Complete distribution of harmonised poisoning categories, including rare groups that were collapsed in the main manuscript figures for legibility.',variables=['poison_type'],analysis_type='full categorical distribution',n=len(clean)); state.mark_coverage('supp_all_poison_types','generated','supp_s3_all_poison_categories')
    # Harmonization results
    fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True); pm=clean[['poison_type_raw','poison_type']].dropna().value_counts().reset_index(name='n').head(15); axes[0].barh(range(len(pm)),pm['n'],color='#1d3557'); axes[0].set_yticks(range(len(pm))); axes[0].set_yticklabels([wrap(f"{a} → {b}",30) for a,b in zip(pm['poison_type_raw'],pm['poison_type'])]); axes[0].set_title('Poisoning-type harmonization examples',loc='left'); despine_and_tidy(axes[0],'x')
    cm=clean[['component_raw','component']].dropna().value_counts().reset_index(name='n').head(15); axes[1].barh(range(len(cm)),cm['n'],color='#6a4c93'); axes[1].set_yticks(range(len(cm))); axes[1].set_yticklabels([wrap(f"{a} → {b}",30) for a,b in zip(cm['component_raw'],cm['component'])]); axes[1].set_title('Component harmonization examples',loc='left'); despine_and_tidy(axes[1],'x'); save_figure(fig,state,'supp_s4_harmonization','Supplementary Figure S4. Harmonization examples','supplementary','04_supplementary_figures','Representative raw-to-clean mapping examples for poisoning type and component name harmonization informed by the Unique Names and DropDowns sheets.',variables=['poison_type_raw','poison_type','component_raw','component'],analysis_type='harmonization audit',n=len(clean)); state.mark_coverage('supp_name_harmonization','generated','supp_s4_harmonization')
    # Stratified context figures
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    for ax,col,title in zip(axes,['occupation','living_area','presentation_area'],['Occupation','Living area','Presentation area']):
        vc=clean[col].fillna('Missing').value_counts().head(10).sort_values(); ax.barh(range(len(vc)),vc.values,color='#1d3557'); ax.set_yticks(range(len(vc))); ax.set_yticklabels([wrap(v,16) for v in vc.index]); ax.set_title(title,loc='left'); despine_and_tidy(ax,'x')
    save_figure(fig,state,'supp_s5_context_strata','Supplementary Figure S5. Contextual stratification','supplementary','04_supplementary_figures','Distributions of occupation, living area, and presentation area. High-cardinality categories are truncated to the most frequent levels for readability.',variables=['occupation','living_area','presentation_area'],analysis_type='context descriptive',n=len(clean)); state.mark_coverage('supp_stratified_context','generated','supp_s5_context_strata')
    # Labs overview
    lab_cols=[c for c in ['wbc_count_mm3','creatinine_mg_dl','ph','hco3_mmol_l','sodium_mmol_l','potassium_mmol_l','hemoglobin_g_dl','bilirubin_mg_dl','sgpt_u_l','sgot_u_l'] if c in clean.columns and clean[c].notna().sum()>=20]
    if lab_cols:
        fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True); completeness=clean[lab_cols].notna().mean().sort_values(); axes[0,0].barh(range(len(completeness)),completeness.values,color='#4e79a7'); axes[0,0].set_yticks(range(len(completeness))); axes[0,0].set_yticklabels([wrap(v,18) for v in completeness.index]); axes[0,0].set_title('Lab completeness',loc='left'); axes[0,0].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); despine_and_tidy(axes[0,0],'x')
        abnormal=[]
        ref={'creatinine_mg_dl':('>1.2',lambda s:s>1.2),'ph':('<7.35',lambda s:s<7.35),'hco3_mmol_l':('<22',lambda s:s<22),'wbc_count_mm3':('>11000',lambda s:s>11000),'potassium_mmol_l':('>5.2',lambda s:s>5.2)}
        for col,(lab,fn) in ref.items():
            if col in clean.columns and clean[col].notna().sum()>=20: abnormal.append({'lab':f'{col} {lab}','pct':fn(clean[col].dropna()).mean()})
        abdf=pd.DataFrame(abnormal).sort_values('pct') if abnormal else pd.DataFrame({'lab':[],'pct':[]}); axes[0,1].barh(range(len(abdf)),abdf['pct'],color='#bc4749'); axes[0,1].set_yticks(range(len(abdf))); axes[0,1].set_yticklabels([wrap(v,18) for v in abdf['lab']]); axes[0,1].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axes[0,1].set_title('Selected abnormality prevalence',loc='left'); despine_and_tidy(axes[0,1],'x')
        shown=lab_cols[:4]
        for ax,col in zip(axes.flat[2:],shown[:2]): sns.boxplot(data=clean,x='outcome_category',y=col,ax=ax,showfliers=False); ax.set_title(f'{col} by outcome',loc='left'); ax.tick_params(axis='x',rotation=30); despine_and_tidy(ax,'y')
        save_figure(fig,state,'supp_s6_labs_overview','Supplementary Figure S6. Laboratory overview','supplementary','04_supplementary_figures','Overview of laboratory-data completeness, selected abnormality prevalence, and example distributions by outcome. Boxplots use observed non-missing values and suppress extreme fliers for readability.',variables=lab_cols+['outcome_category'],analysis_type='lab overview',n=len(clean)); state.mark_coverage('supp_labs_overview','generated','supp_s6_labs_overview')
    else: state.mark_coverage('supp_labs_overview','skipped',reason='Insufficient non-missing laboratory variables for overview figure')
    # Correlations
    corr_cols=[c for c in ['gcs','spo2','sbp','dbp','pulse_bpm','respiratory_rate','creatinine_mg_dl','ph','hco3_mmol_l','glucose_mmol_l'] if c in clean.columns and clean[c].notna().sum()>=30]
    if len(corr_cols)>=4:
        fig,ax=plt.subplots(figsize=(10,8)); corr=clean[corr_cols].corr(method='spearman'); sns.heatmap(corr,cmap='vlag',center=0,ax=ax); ax.set_title('Supplementary Figure S7. Correlation heatmap for vitals and labs',loc='left'); save_figure(fig,state,'supp_s7_vital_lab_correlation','Supplementary Figure S7. Correlation heatmap for vitals and labs','supplementary','04_supplementary_figures','Spearman correlation matrix across selected vital-sign and laboratory variables with sufficient completeness.',variables=corr_cols,analysis_type='correlation heatmap',n=len(clean)); state.mark_coverage('supp_vitals_lab_correlations','generated','supp_s7_vital_lab_correlation')
    else: state.mark_coverage('supp_vitals_lab_correlations','skipped',reason='Fewer than four sufficiently complete numeric variables for correlation heatmap')
    # Alluvial and follow-up
    for analysis_id,fig_id,cols,title in [('supp_alluvial_site_type_outcome','supp_s8_alluvial_site_type_outcome',['study_site','poison_type_plot','outcome_category'],'Site → poisoning type → outcome'),('supp_alluvial_age_type_outcome','supp_s9_alluvial_age_type_outcome',['age_group','poison_type_plot','outcome_category'],'Age group → poisoning type → outcome'),('supp_alluvial_type_treatment_outcome','supp_s10_alluvial_type_treat_outcome',['poison_type_plot','outcome_category','followup_status'],'Poisoning type → outcome → follow-up status')]:
        fig,ax=plt.subplots(figsize=(12,7)); ok=custom_alluvial_three_stage(ax,clean.assign(poison_type_plot=clean['poison_type_major']),cols,title=title)
        if ok: save_figure(fig,state,fig_id,title,'supplementary','04_supplementary_figures',f'Static alluvial summary of {title.lower()}. Categories are truncated to the most frequent levels to preserve readability.',variables=cols,analysis_type='static alluvial',n=len(clean)); state.mark_coverage(analysis_id,'generated',fig_id)
        else: plt.close(fig); state.mark_coverage(analysis_id,'skipped',reason=f'Insufficient complete data for {title}')
    fig,ax=plt.subplots(figsize=(12,7)); ok=custom_alluvial_three_stage(ax,clean.assign(poison_type_plot=clean['poison_type_major']),['poison_type_plot','outcome_category','followup_status'],title='Poisoning type → discharge outcome → follow-up status')
    if ok: save_figure(fig,state,'supp_s11_followup_pathway','Supplementary Figure S11. Follow-up pathway','supplementary','04_supplementary_figures','Static pathway figure linking poisoning type, discharge outcome, and follow-up status. Follow-up categories represent recorded follow-up status and include Missing/unknown when relevant.',variables=['poison_type_major','outcome_category','followup_status'],analysis_type='follow-up alluvial',n=len(clean)); state.mark_coverage('supp_followup_pathway','generated','supp_s11_followup_pathway')
    else: plt.close(fig); state.mark_coverage('supp_followup_pathway','skipped',reason='Insufficient complete follow-up data for pathway figure')
    # Absconded
    fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True); by_site=clean.groupby('study_site')['absconded_flag'].mean().sort_values(); axes[0].barh(range(len(by_site)),by_site.values,color='#bc4749'); axes[0].set_yticks(range(len(by_site))); axes[0].set_yticklabels([wrap(v,14) for v in by_site.index]); axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axes[0].set_title('Absconded/DORB rate by site',loc='left'); despine_and_tidy(axes[0],'x')
    by_type=clean.groupby('poison_type_major')['absconded_flag'].mean().sort_values(); axes[1].barh(range(len(by_type)),by_type.values,color='#ca6702'); axes[1].set_yticks(range(len(by_type))); axes[1].set_yticklabels([wrap(v,14) for v in by_type.index]); axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); axes[1].set_title('Absconded/DORB rate by poisoning type',loc='left'); despine_and_tidy(axes[1],'x'); save_figure(fig,state,'supp_s12_absconded_rates','Supplementary Figure S12. Absconded/DORB rates','supplementary','04_supplementary_figures','Absconded/DORB proportions by site and by major poisoning type using observed denominators.',variables=['study_site','poison_type_major','absconded_flag'],analysis_type='rate comparison',n=len(clean)); state.mark_coverage('supp_absconded_rates','generated','supp_s12_absconded_rates')
    # Delay/amount severity
    fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True); tab=pd.crosstab(clean['presentation_time_category'],clean['severe_outcome'],normalize='index'); tab.plot(kind='bar',stacked=True,ax=axes[0],color=['#d9dde3','#bc4749']); axes[0].set_title('Time-to-presentation category vs severe outcome',loc='left'); axes[0].set_xlabel(''); axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); despine_and_tidy(axes[0],'y')
    if 'amount_ingested_ml' in clean.columns and clean['amount_ingested_ml'].notna().sum()>=30: sns.boxplot(data=clean,x='outcome_category',y='amount_ingested_ml',ax=axes[1],showfliers=False); axes[1].set_title('Amount ingested vs outcome',loc='left'); axes[1].tick_params(axis='x',rotation=30); despine_and_tidy(axes[1],'y'); state.mark_coverage('supp_delay_amount_vs_severity','generated','supp_s13_delay_amount_severity')
    else: axes[1].axis('off'); axes[1].text(0.5,0.5,'Usable amount-ingested data unavailable',ha='center',va='center'); state.mark_coverage('supp_delay_amount_vs_severity','generated','supp_s13_delay_amount_severity')
    save_figure(fig,state,'supp_s13_delay_amount_severity','Supplementary Figure S13. Delay and amount versus severity','supplementary','04_supplementary_figures','Association between presentation delay and severe outcome, with a companion amount-ingested plot where usable numeric data exist. Missing or implausible values are not interpreted as zero.',variables=['presentation_time_category','severe_outcome','amount_ingested_ml','outcome_category'],analysis_type='delay/amount severity',n=len(clean))
    # Site heterogeneity and sensitivities
    fig,ax=plt.subplots(figsize=(10,8)); rows=[]
    for site,sub in clean.groupby('study_site'):
        if len(sub)<20: continue
        k=int((sub['death_flag']==1).sum()); n=int(sub['death_flag'].notna().sum()); ph,lo,hi=wilson_ci(k,n); rows.append({'site':site,'rate':ph,'low':lo,'high':hi})
    site_tbl=pd.DataFrame(rows).sort_values('rate'); plot_rate_with_ci(ax,site_tbl,'rate','low','high','site'); ax.set_title('Supplementary Figure S14. Site heterogeneity in death rate',loc='left'); save_figure(fig,state,'supp_s14_site_heterogeneity','Supplementary Figure S14. Site heterogeneity','supplementary','04_supplementary_figures','Site-level death-rate heterogeneity with Wilson 95% confidence intervals using observed denominators.',variables=['study_site','death_flag'],analysis_type='heterogeneity forest',n=len(clean)); state.mark_coverage('supp_site_heterogeneity','generated','supp_s14_site_heterogeneity')
    fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True); sparse=clean['poison_type'].value_counts(); sparse_keep=sparse[sparse>=30].index; full=clean.groupby('poison_type_major')['severe_outcome'].mean().sort_values(); filt=clean[clean['poison_type'].isin(sparse_keep)].groupby('poison_type')['severe_outcome'].mean().sort_values(); axes[0].barh(range(len(full)),full.values,color='#4e79a7'); axes[0].set_yticks(range(len(full))); axes[0].set_yticklabels([wrap(v,16) for v in full.index]); axes[0].set_title('Main grouping with rare categories collapsed',loc='left'); despine_and_tidy(axes[0],'x'); axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%'))
    axes[1].barh(range(len(filt)),filt.values,color='#1d3557'); axes[1].set_yticks(range(len(filt))); axes[1].set_yticklabels([wrap(v,16) for v in filt.index]); axes[1].set_title('Sensitivity excluding sparse categories',loc='left'); axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); despine_and_tidy(axes[1],'x'); save_figure(fig,state,'supp_s15_sparse_sensitivity','Supplementary Figure S15. Sensitivity excluding sparse categories','supplementary','04_supplementary_figures','Comparison of severe-outcome rates using the main collapsed poisoning-type grouping and a sensitivity analysis restricted to poisoning types with at least 30 records.',variables=['poison_type','poison_type_major','severe_outcome'],analysis_type='sensitivity analysis',n=len(clean)); state.mark_coverage('supp_sparse_category_sensitivity','generated','supp_s15_sparse_sensitivity')
    impl_cols=[c for c in clean.columns if c.endswith('_flag') and c not in {'admission_date_quality_flag','ingestion_date_quality_flag'}]; impl_mask=pd.DataFrame({c:clean[c]=='implausible' for c in impl_cols if clean[c].dtype=='object'}) if impl_cols else pd.DataFrame(index=clean.index)
    if not impl_mask.empty and 'severe_outcome' in clean.columns:
        clean_impl=clean.loc[~impl_mask.any(axis=1)]; fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True); a=clean.groupby('poison_type_major')['severe_outcome'].mean().sort_values(); b=clean_impl.groupby('poison_type_major')['severe_outcome'].mean().reindex(a.index)
        axes[0].barh(range(len(a)),a.values,color='#6c757d'); axes[0].set_yticks(range(len(a))); axes[0].set_yticklabels([wrap(v,16) for v in a.index]); axes[0].set_title('All records',loc='left'); axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); despine_and_tidy(axes[0],'x')
        axes[1].barh(range(len(b)),b.values,color='#1b4332'); axes[1].set_yticks(range(len(b))); axes[1].set_yticklabels([wrap(v,16) for v in b.index]); axes[1].set_title('Excluding implausible numeric values',loc='left'); axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{100*v:.0f}%')); despine_and_tidy(axes[1],'x'); save_figure(fig,state,'supp_s16_implausible_numeric_sensitivity','Supplementary Figure S16. Sensitivity to implausible numeric values','supplementary','04_supplementary_figures','Sensitivity analysis comparing severe-outcome rates before and after excluding records with any implausible numeric flag. Records are not deleted from the cleaned dataset; this is a reporting-only sensitivity analysis.',variables=['severe_outcome']+impl_cols,analysis_type='sensitivity analysis',n=len(clean)); state.mark_coverage('supp_implausible_numeric_sensitivity','generated','supp_s16_implausible_numeric_sensitivity')
    else: state.mark_coverage('supp_implausible_numeric_sensitivity','skipped',reason='No implausible numeric flags available for sensitivity analysis')
    # Model diagnostics
    if model_results.get('death_adj_imp') is not None and not model_results['death_adj_imp'].empty:
        fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True); roc_info,calib=cross_validated_roc_and_calibration(model_results.get('death_X_imp'),model_results.get('death_y_imp'),logger) if model_results.get('death_X_imp') is not None else (None,None)
        if roc_info is not None: axes[0].plot(roc_info['diag']['fpr'],roc_info['diag']['tpr'],color='#1d3557'); axes[0].plot([0,1],[0,1],'--',color='#adb5bd'); axes[0].set_title(f"ROC (AUC {roc_info['auc']:.2f})",loc='left'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); despine_and_tidy(axes[0],'both'); axes[1].plot(calib['mean_pred'],calib['frac_pos'],'o-',color='#bc4749'); axes[1].plot([0,1],[0,1],'--',color='#adb5bd'); axes[1].set_title('Calibration',loc='left'); axes[1].set_xlabel('Mean predicted'); axes[1].set_ylabel('Observed fraction'); despine_and_tidy(axes[1],'both'); save_figure(fig,state,'supp_s17_model_diagnostics','Supplementary Figure S17. Model diagnostics','supplementary','04_supplementary_figures','Exploratory model diagnostics for the death model based on internal cross-validation. These diagnostic panels are shown only when event counts and data structure support internal validation.',variables=['death_flag'],analysis_type='model diagnostics',n=len(clean)); state.mark_coverage('supp_model_diagnostics','generated','supp_s17_model_diagnostics')
        else: plt.close(fig); state.mark_coverage('supp_model_diagnostics','skipped',reason='ROC/calibration diagnostics not statistically appropriate or insufficient data')
    else: state.mark_coverage('supp_model_diagnostics','skipped',reason='Adjusted death model unavailable')
    # PCA if enough labs
    if OPTIONAL_IMPORTS['sklearn'] and len(lab_cols)>=5 and clean[lab_cols].dropna().shape[0]>=50:
        X=clean[lab_cols].dropna(); comps=PCA(n_components=2,random_state=SEED).fit_transform(StandardScaler().fit_transform(X)); pca_df=pd.DataFrame({'PC1':comps[:,0],'PC2':comps[:,1],'Outcome':clean.loc[X.index,'outcome_category'].fillna('Missing')})
        fig,ax=plt.subplots(figsize=(8,7));
        for i,(lab,sub) in enumerate(pca_df.groupby('Outcome')): ax.scatter(sub['PC1'],sub['PC2'],s=22,alpha=0.8,label=lab,color=CATEGORICAL_BASE[i%len(CATEGORICAL_BASE)])
        ax.set_title('Supplementary Figure S18. PCA of laboratory data',loc='left'); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left'); despine_and_tidy(ax,'both'); save_figure(fig,state,'supp_s18_pca_labs','Supplementary Figure S18. PCA of laboratory data','supplementary','04_supplementary_figures','Exploratory PCA using complete laboratory cases only. This plot is descriptive and not intended to imply clustering validity or predictive performance.',variables=lab_cols+['outcome_category'],analysis_type='PCA',n=len(X)); state.mark_coverage('supp_pca_labs','generated','supp_s18_pca_labs')
    else: state.mark_coverage('supp_pca_labs','skipped',reason='Insufficient complete laboratory data for PCA/UMAP')
    # Exploratory component heatmap
    fig,ax=plt.subplots(figsize=(10,8)); exp_tbl=pd.crosstab(clean['component_major'],clean['outcome_category']); exp_tbl=exp_tbl.loc[exp_tbl.sum(axis=1).sort_values(ascending=False).index[:20]]; sns.heatmap(exp_tbl,cmap=sns.light_palette('#2a9d8f',as_cmap=True),ax=ax); ax.set_title('Exploratory Figure E1. Component × outcome heatmap',loc='left'); save_figure(fig,state,'exp_e1_component_outcome_heatmap','Exploratory Figure E1. Component × outcome heatmap','exploratory','05_exploratory_figures','Exploratory cross-tabulation of major component groups against outcome categories.',variables=['component_major','outcome_category'],analysis_type='exploratory heatmap',n=len(clean)); state.mark_coverage('exp_component_heatmaps','generated','exp_e1_component_outcome_heatmap')
    fig,ax=plt.subplots(figsize=(10,6)); obs=clean.groupby('admission_date_quality_flag').size().sort_values(); ax.barh(range(len(obs)),obs.values,color='#4e79a7'); ax.set_yticks(range(len(obs))); ax.set_yticklabels([wrap(v,22) for v in obs.index]); ax.set_title('Exploratory Figure E2. Temporal QA flags',loc='left'); despine_and_tidy(ax,'x'); save_figure(fig,state,'exp_e2_temporal_qc','Exploratory Figure E2. Temporal QA flags','exploratory','05_exploratory_figures','Exploratory audit of admission-date flags demonstrating that implausible year-1900 or out-of-window dates were identified and isolated from temporal plotting.',variables=['admission_date_quality_flag'],analysis_type='temporal QA',n=len(clean)); state.mark_coverage('exp_temporal_qc','generated','exp_e2_temporal_qc')

def save_version_control(input_path:Path, script_path:Path, outdir:Path, run_id:str, metadata:Dict[str,Any]):
    vc=outdir/'09_version_control'; data={'run_id':run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'input_file':str(input_path),'input_sha256':sha256_of_file(input_path),'script_file':str(script_path),'script_sha256':sha256_of_file(script_path),'python_version':sys.version,'platform':platform.platform(),'package_versions':{'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':mpl.__version__,**{k:str(v) for k,v in OPTIONAL_IMPORTS.items()}},'analysis_configuration':{'seed':SEED,'main_sheet':MAIN_SHEET,'study_window':metadata.get('study_window',{}),'pii_patterns':PII_PATTERNS,'symptom_columns':SYMPTOM_COLUMNS},'metadata':metadata}
    (vc/'run_metadata.json').write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
    if yaml is not None: (vc/'analysis_configuration.yaml').write_text(yaml.safe_dump(data['analysis_configuration'],sort_keys=False),encoding='utf-8')
    else: (vc/'analysis_configuration.yaml').write_text('# pyyaml unavailable; JSON configuration written in run_metadata.json\n',encoding='utf-8')

def save_clean_outputs(clean,raw_header_map,dq,outdir:Path):
    safe=clean.copy(); safe.to_csv(outdir/'01_clean_data'/'clean_analysis_dataset.csv',index=False); dq.to_csv(outdir/'01_clean_data'/'data_quality_summary.csv',index=False); pd.DataFrame(raw_header_map).to_csv(outdir/'01_clean_data'/'header_mapping.csv',index=False)
    try:
        if OPTIONAL_IMPORTS['pyarrow']: safe.to_parquet(outdir/'01_clean_data'/'clean_analysis_dataset.parquet',index=False)
    except Exception: pass

def qa_posthoc(state,clean):
    rows=[]
    for rec in state.figure_registry:
        notes=[]
        if 'temporal' in rec.analysis_type and clean['admission_date_for_plots'].isna().all(): notes.append('No valid temporal data')
        if rec.n is None or rec.n<=0: notes.append('Missing n annotation')
        if rec.section=='main' and 'Figure' not in rec.title: notes.append('Main figure title check')
        rows.append({'figure_id':rec.figure_id,'title':rec.title,'qa_status':'pass' if not notes else 'review','notes':' | '.join(notes)})
    pd.DataFrame(rows).to_csv(state.outdir/'08_quality_control'/'visual_qa_results.csv',index=False)

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args(argv)
    input_path=Path(args.input).resolve(); outdir=Path(args.outdir).resolve(); script_path=Path(__file__).resolve(); run_id=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
    initialise_output_tree(outdir,args.overwrite); logger=init_logging(outdir); logger.info('Starting poison publication figure pipeline v2'); state=AnalysisState(outdir,run_id)
    try:
        raw,unique_names,dropdowns,header_map=load_workbook_sheets(input_path,logger); clean,dq,metadata=build_analysis_dataset(raw,unique_names,dropdowns,logger); save_clean_outputs(clean,header_map,dq,outdir); save_summary_tables(clean,dq,state)
        model_results=generate_main_figures(clean,state,logger); death_adj_imp,_,_,death_X_imp,death_y_imp=fit_adjusted_logistic(clean,'death_flag',impute=True,logger=logger); model_results['death_X_imp']=death_X_imp; model_results['death_y_imp']=death_y_imp
        generate_supplementary_and_exploratory(clean,dq,metadata,model_results,state,logger); save_version_control(input_path,script_path,outdir,run_id,metadata); write_registry_and_scorecards(state); make_gallery(state); qa_posthoc(state,clean); logger.info('Pipeline completed successfully. Generated %s figures.',len(state.figure_registry)); return 0
    except Exception as e:
        logger.error('Pipeline failed: %s',e); logger.error(traceback.format_exc()); raise

if __name__=='__main__': raise SystemExit(main())
