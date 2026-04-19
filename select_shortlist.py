import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============ Paths ============
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL = os.path.join(_THIS_DIR, 'output', 'data', 'long_df.pkl')
_OUT_DIR = os.path.join(_THIS_DIR, 'output', 'data')
os.makedirs(_OUT_DIR, exist_ok=True)

# ============ Load data ============
long_df = pd.read_pickle(_PKL)

# ============ Build wide pivot ============
pivot = long_df.pivot_table(
    index='Code', columns='Year', values='Admissions', aggfunc='first'
).reset_index()

meta = (
    long_df.groupby('Code')
    .agg(Description=('Description', lambda s: max(s.dropna().astype(str), key=len, default='')),
         Chapter_Num=('Chapter_Num', 'first'),
         Chapter_Name=('Chapter_Name', 'first'))
    .reset_index()
)
pivot = pivot.merge(meta, on='Code', how='left')

pivot['baseline']      = pivot['2019-20']
pivot['pct_lockdown']  = (pivot['2020-21'] - pivot['2019-20']) / pivot['2019-20'] * 100
pivot['pct_long_term'] = (pivot['2023-24'] - pivot['2019-20']) / pivot['2019-20'] * 100

# Keep candidates with usable baseline
pool = pivot[pivot['baseline'].notna() & (pivot['baseline'] > 0)].copy()

# ============ Story-critical whitelist ============
WHITELIST = {
    'F50-F59': 'Eating disorders — teenage doubling',
    'U00-U49': 'COVID-19 codes — pandemic marker',
    'J20-J22': 'Lower resp infections — biggest drop (-70%)',
    'K00-K14': 'Oral cavity — dentist closure (-57%)',
    'E65-E68': 'Obesity — long-term non-recovery',
    'I20-I25': 'Ischaemic heart disease — chronic non-recovery',
    'M00-M25': 'Arthropathies — elective surgery backlog',
    'Z30-Z39': 'Pregnancy care — cannot be postponed',
    'F00-F09': 'Dementia — aging population anchor',
}

# ============ Per-chapter selection (max 2 per chapter) ============
keep_rows = []
MAX_PER_CHAPTER = 2

for chap_num, chap_df in pool.groupby('Chapter_Num'):
    selected_codes = []
    reasons = {}
    chap_df = chap_df.copy()

    # Priority 1: whitelist hits inside this chapter
    for _, r in chap_df.iterrows():
        if r['Code'] in WHITELIST:
            selected_codes.append(r['Code'])
            reasons[r['Code']] = f"Whitelist: {WHITELIST[r['Code']]}"
        if len(selected_codes) >= MAX_PER_CHAPTER:
            break

    # Priority 2: chapter's largest volume
    if len(selected_codes) < MAX_PER_CHAPTER:
        by_vol = chap_df.sort_values('baseline', ascending=False)
        for _, r in by_vol.iterrows():
            if r['Code'] not in selected_codes:
                selected_codes.append(r['Code'])
                reasons[r['Code']] = 'Largest volume in chapter'
                break

    # Priority 3: biggest absolute Lockdown shock
    if len(selected_codes) < MAX_PER_CHAPTER:
        chap_df['_abs_lock'] = chap_df['pct_lockdown'].abs()
        by_shock = chap_df.sort_values('_abs_lock', ascending=False, na_position='last')
        for _, r in by_shock.iterrows():
            if r['Code'] not in selected_codes and pd.notna(r['pct_lockdown']):
                selected_codes.append(r['Code'])
                reasons[r['Code']] = 'Biggest lockdown shock in chapter'
                break

    for c in selected_codes:
        row = chap_df[chap_df['Code'] == c].iloc[0].to_dict()
        row['Reason'] = reasons[c]
        keep_rows.append(row)

shortlist = pd.DataFrame(keep_rows)

# ============ Sort by Chapter, then by volume ============
_roman_order = {
    'I': 1, 'II': 2, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'IX': 9,
    'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
    'XXI': 21, 'XXII': 22, '?': 99
}
shortlist['_sort'] = shortlist['Chapter_Num'].map(_roman_order)
shortlist = shortlist.sort_values(['_sort', 'baseline'], ascending=[True, False]).reset_index(drop=True)

# ============ Format and save ============
final_cols = [
    'Chapter_Num', 'Chapter_Name', 'Code', 'Description',
    '2019-20', '2020-21', '2021-22', '2022-23', '2023-24',
    'pct_lockdown', 'pct_long_term', 'Reason'
]
final = shortlist[final_cols].copy()

for c in ['pct_lockdown', 'pct_long_term']:
    final[c] = final[c].round(1)

for c in ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']:
    final[c] = final[c].apply(lambda v: int(v) if pd.notna(v) else None)

print(f"\n✅ Tight shortlist built: {len(final)} diseases across "
      f"{final['Chapter_Num'].nunique()} chapters\n")

csv_path  = os.path.join(_OUT_DIR, 'shortlist_final.csv')
xlsx_path = os.path.join(_OUT_DIR, 'shortlist_final.xlsx')
final.to_csv(csv_path, index=False, encoding='utf-8-sig')
final.to_excel(xlsx_path, index=False, sheet_name='Shortlist_40')

print(f"Saved:")
print(f"  {csv_path}")
print(f"  {xlsx_path}")

# ============ Console preview ============
print("\n" + "=" * 112)
print("FINAL SHORTLIST (grouped by ICD Chapter)")
print("=" * 112)
for chap_num in final['Chapter_Num'].unique():
    chap_rows = final[final['Chapter_Num'] == chap_num]
    chap_name = chap_rows['Chapter_Name'].iloc[0]
    print(f"\n  Chapter {chap_num}. {chap_name}")
    print(f"  {'-' * 108}")
    print(f"  {'Code':<10} {'Description':<50} {'2019':>10} {'2020':>10} "
          f"{'Lock%':>7} {'Long%':>7}")
    for _, r in chap_rows.iterrows():
        desc = str(r['Description'])[:49]
        v19 = f"{r['2019-20']:,}" if r['2019-20'] is not None else '   -   '
        v20 = f"{r['2020-21']:,}" if r['2020-21'] is not None else '   -   '
        lock = f"{r['pct_lockdown']:.1f}" if pd.notna(r['pct_lockdown']) else '  -  '
        long = f"{r['pct_long_term']:.1f}" if pd.notna(r['pct_long_term']) else '  -  '
        print(f"  {r['Code']:<10} {desc:<50} {v19:>10} {v20:>10} "
              f"{lock:>7} {long:>7}")

print(f"\n📝 Missing values (shown as '-') will render as GREY cells in the heatmap.")