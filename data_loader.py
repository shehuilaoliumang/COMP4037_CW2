import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_THIS_DIR, 'nhs_data')

# Output directory tree — auto-created on first run
OUTPUT_ROOT = os.path.join(_THIS_DIR, 'output')
OUTPUT_DATA_DIR    = os.path.join(OUTPUT_ROOT, 'data')
OUTPUT_FIGURES_DIR = os.path.join(OUTPUT_ROOT, 'figures')
OUTPUT_DRAFTS_DIR  = os.path.join(OUTPUT_ROOT, 'drafts')


def ensure_output_dirs():
    """Create output/{data,figures,drafts} if they don't already exist."""
    for d in (OUTPUT_DATA_DIR, OUTPUT_FIGURES_DIR, OUTPUT_DRAFTS_DIR):
        os.makedirs(d, exist_ok=True)

FILES = {
    '2019-20': 'hosp-epis-stat-admi-diag-2019-20-tab supp.xlsx',
    '2020-21': 'hosp-epis-stat-admi-diag-2020-21-tab.xlsx',
    '2021-22': 'hosp-epis-stat-admi-diag-2021-22-tab.xlsx',
    '2022-23': 'hosp-epis-stat-admi-diag-2022-23-tab_V2.xlsx',
    '2023-24': 'hosp-epis-stat-admi-diag-2023-24-tab.xlsx',
}

# 23 age groups available in new-format files (2012-2024)
AGE_COLUMNS = [
    'Age 0', 'Age 1-4', 'Age 5-9', 'Age 10-14',
    'Age 15', 'Age 16', 'Age 17', 'Age 18', 'Age 19',
    'Age 20-24', 'Age 25-29', 'Age 30-34', 'Age 35-39',
    'Age 40-44', 'Age 45-49', 'Age 50-54', 'Age 55-59',
    'Age 60-64', 'Age 65-69', 'Age 70-74', 'Age 75-79',
    'Age 80-84', 'Age 85-89', 'Age 90+'
]

# ICD-10 Chapter mapping (grouped by first letter of code)
# This gives us the hierarchical structure for treemap/heatmap grouping
ICD10_CHAPTERS = {
    'A': ('I',    'Infectious & parasitic diseases'),
    'B': ('I',    'Infectious & parasitic diseases'),
    'C': ('II',   'Neoplasms'),
    'D': ('II',   'Neoplasms / Blood disorders'),
    'E': ('IV',   'Endocrine, nutritional & metabolic'),
    'F': ('V',    'Mental & behavioural disorders'),
    'G': ('VI',   'Nervous system'),
    'H': ('VII',  'Eye / Ear diseases'),
    'I': ('IX',   'Circulatory system'),
    'J': ('X',    'Respiratory system'),
    'K': ('XI',   'Digestive system'),
    'L': ('XII',  'Skin & subcutaneous'),
    'M': ('XIII', 'Musculoskeletal'),
    'N': ('XIV',  'Genitourinary system'),
    'O': ('XV',   'Pregnancy & childbirth'),
    'P': ('XVI',  'Perinatal conditions'),
    'Q': ('XVII', 'Congenital malformations'),
    'R': ('XVIII','Symptoms & abnormal findings'),
    'S': ('XIX',  'Injury & poisoning'),
    'T': ('XIX',  'Injury & poisoning'),
    'U': ('XXII', 'Special codes (incl. COVID-19)'),
    'V': ('XX',   'External causes'),
    'W': ('XX',   'External causes'),
    'X': ('XX',   'External causes'),
    'Y': ('XX',   'External causes'),
    'Z': ('XXI',  'Health services factors'),
}


def _find_header_row(df_raw, max_rows=25):
    """Find the row index containing 'Admissions' and 'Female' headers."""
    for i in range(min(max_rows, len(df_raw))):
        row_vals = [str(v).replace('\n', ' ') for v in df_raw.iloc[i].values if pd.notna(v)]
        row_str = ' '.join(row_vals)
        if ('Admission' in row_str) and ('Female' in row_str):
            return i
    return None


def _normalize_columns(columns):
    """Clean column names: remove newlines, (FAE)/(FCE) suffixes, extra spaces.

    2023-24 file uses columns like 'Age 0 \n(FCE)' and 'Emergency \n(FAE)' while
    2019-20 to 2022-23 use plain 'Age 0' / 'Emergency'. We strip the suffixes to
    unify them.
    """
    cleaned = []
    for c in columns:
        s = str(c).replace('\n', ' ').strip()
        # Remove trailing (FAE) / (FCE) / (Days) / (Years) markers
        for suffix in ['(FAE)', '(FCE)', '(Days)', '(Years)']:
            s = s.replace(suffix, '').strip()
        # Collapse multiple spaces
        s = ' '.join(s.split())
        cleaned.append(s)
    return cleaned


def _find_admissions_column(columns):
    """Identify the 'Admissions' column across different naming conventions."""
    # 2019-20 to 2022-23: called "Admissions"
    # 2023-24: called "Finished Admission Episodes (FAE)"
    for c in columns:
        if c == 'Admissions':
            return c
    for c in columns:
        cl = c.lower().replace(' ', '')
        if 'admissionepisodes' in cl and 'fae' in cl:
            return c
    for c in columns:
        if 'Admission' in c and 'method' not in c.lower() and 'Consultant' not in c:
            return c
    return None


def _build_code_desc(df):
    """Handle both combined (code+desc in one col) and split (code | desc in 2 cols) layouts."""
    # Check a non-total data row to detect format
    sample_val = None
    for i in range(len(df)):
        v = df.iloc[i, 0]
        if pd.notna(v) and str(v).strip() not in ('Total', 'nan', ''):
            sample_val = str(v).strip()
            break

    if sample_val and len(sample_val) <= 10 and '-' in sample_val:
        # Split format (e.g. 2023-24): Col 0 = "A00-A09", Col 1 = "Intestinal infectious..."
        desc_col = df.columns[1]
        df['Code_Desc'] = df.iloc[:, 0].astype(str).str.strip() + ' ' + df[desc_col].astype(str).str.strip()
    else:
        # Combined format (e.g. 2019-20): Col 0 = "A00-A09 Intestinal infectious..."
        df['Code_Desc'] = df.iloc[:, 0].astype(str).str.strip()

    return df


def load_year(year_label, filepath):
    """
    Load one financial year's Primary Diagnosis Summary sheet.
    Returns a clean DataFrame with standardized columns.
    """
    full_path = os.path.join(DATA_ROOT, filepath) if not os.path.isabs(filepath) else filepath

    # Locate the Primary Diagnosis Summary sheet
    xl = pd.ExcelFile(full_path)
    target_sheet = None
    for s in xl.sheet_names:
        sl = s.lower()
        if 'summary' in sl and 'iagnosis' in sl:
            target_sheet = s
            break

    if target_sheet is None:
        raise ValueError(f"No Primary Diagnosis Summary sheet found in {filepath}")

    # Find header row
    df_raw = pd.read_excel(full_path, sheet_name=target_sheet, header=None)
    header_row = _find_header_row(df_raw)
    if header_row is None:
        raise ValueError(f"Could not locate header row in {filepath}")

    # Read with identified header
    df = pd.read_excel(full_path, sheet_name=target_sheet, header=header_row)
    df.columns = _normalize_columns(df.columns)

    # 2023-24 file has duplicate names (e.g. two 'Emergency' columns — one for
    # admission method, one for bed-days). pandas suffixes duplicates as '.1';
    # we drop the '.1' versions so that the FIRST occurrence (= admission method
    # count) is used, matching earlier years' semantics.
    keep_mask = [not str(c).endswith('.1') for c in df.columns]
    df = df.loc[:, keep_mask]

    # Standardize admissions column
    admis_col = _find_admissions_column(df.columns)
    if admis_col is None:
        raise ValueError(f"Could not find Admissions column in {filepath}")
    if admis_col != 'Admissions':
        df = df.rename(columns={admis_col: 'Admissions'})

    # Build Code_Desc column
    df = _build_code_desc(df)

    # Extract ICD code (e.g. "A00-A09" or "U00-U49")
    df['Code'] = df['Code_Desc'].str.extract(r'^([A-Z]\d+-[A-Z]?\d+|[A-Z]\d+)')

    # Extract human-readable description (strip code prefix)
    df['Description'] = df['Code_Desc'].str.replace(
        r'^[A-Z]\d+-[A-Z]?\d+\s*|^[A-Z]\d+\s*', '', regex=True
    ).str.strip()

    # Keep only rows with valid ICD codes (drops Totals, blanks, footnotes)
    df = df[df['Code'].notna()].reset_index(drop=True)

    # Map to ICD chapter
    df['Chapter_Num'] = df['Code'].str[0].map(lambda x: ICD10_CHAPTERS.get(x, ('?', '?'))[0])
    df['Chapter_Name'] = df['Code'].str[0].map(lambda x: ICD10_CHAPTERS.get(x, ('?', '?'))[1])

    # Add year label
    df['Year'] = year_label

    return df


def load_all_years():
    """Load all 5 financial years into a dictionary of DataFrames."""
    data = {}
    for year_label, filename in FILES.items():
        print(f"  Loading {year_label}...", end=' ')
        df = load_year(year_label, filename)
        data[year_label] = df
        print(f"shape = {df.shape}")
    return data


def build_long_format(data_dict):
    """
    Build a long-format DataFrame with one row per (year × disease category).
    Keeps only columns we need for the analysis.
    """
    keep_cols = [
        'Year', 'Code', 'Description', 'Chapter_Num', 'Chapter_Name',
        'Admissions', 'Emergency', 'Waiting list', 'Planned'
    ] + AGE_COLUMNS

    frames = []
    for year_label, df in data_dict.items():
        # Some columns may not exist in all years — fill with NaN
        row = df.copy()
        for c in keep_cols:
            if c not in row.columns:
                row[c] = np.nan
        frames.append(row[keep_cols])

    long_df = pd.concat(frames, ignore_index=True)
    # Ensure numeric
    numeric_cols = ['Admissions', 'Emergency', 'Waiting list', 'Planned'] + AGE_COLUMNS
    for c in numeric_cols:
        long_df[c] = pd.to_numeric(long_df[c], errors='coerce')

    return long_df


# ===================================================================
# Execute data loading
# ===================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Loading NHS Hospital Admissions Data (5 financial years)")
    print("=" * 70)

    data = load_all_years()
    long_df = build_long_format(data)

    # Make sure output/ tree exists
    ensure_output_dirs()

    # Save in three formats — each serves a different purpose:
    #   .pkl  — fastest to reload in Python, preserves exact dtypes
    #   .csv  — human-readable, openable in Excel / Notepad / any text editor
    #   .xlsx — native Excel format, best for quick visual inspection
    pkl_path  = os.path.join(OUTPUT_DATA_DIR, 'long_df.pkl')
    csv_path  = os.path.join(OUTPUT_DATA_DIR, 'long_df.csv')
    xlsx_path = os.path.join(OUTPUT_DATA_DIR, 'long_df.xlsx')

    long_df.to_pickle(pkl_path)
    long_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # BOM helps Excel read non-ASCII
    long_df.to_excel(xlsx_path, index=False, sheet_name='NHS_5year_long')

    print(f"\n✅ Long-format DataFrame built: {long_df.shape}")
    print(f"   Saved to output/data/ (3 formats):")
    print(f"     • long_df.pkl   (Python re-load)")
    print(f"     • long_df.csv   (Excel / text editor)")
    print(f"     • long_df.xlsx  (Excel native)")