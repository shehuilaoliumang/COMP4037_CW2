import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the pre-built long DataFrame from output/data/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH = os.path.join(_THIS_DIR, 'output', 'data', 'long_df.pkl')

if not os.path.exists(_PKL_PATH):
    raise FileNotFoundError(
        f"long_df.pkl not found at {_PKL_PATH}. "
        "Please run data_loader.py first to generate it."
    )

long_df = pd.read_pickle(_PKL_PATH)

print("=" * 80)
print("📊 DATA VALIDATION SUMMARY")
print("=" * 80)

# =========================================================
# 1. Row counts per year
# =========================================================
print("\n【1】 每年疾病类别数 (应在 200-220 之间)")
print("-" * 80)
year_counts = long_df.groupby('Year').size().reset_index(name='Categories')
print(year_counts.to_string(index=False))

# =========================================================
# 2. Total admissions per year (should match earlier analysis)
# =========================================================
print("\n【2】 每年总入院数 (应显示 Lockdown 暴跌)")
print("-" * 80)
totals = long_df.groupby('Year')['Admissions'].sum().reset_index()
totals['Admissions'] = totals['Admissions'].astype(int)
totals['vs 2019-20'] = ((totals['Admissions'] - totals['Admissions'].iloc[0]) /
                        totals['Admissions'].iloc[0] * 100).round(1).astype(str) + '%'
print(totals.to_string(index=False))

# =========================================================
# 3. ICD Chapter distribution
# =========================================================
print("\n【3】 ICD-10 章节分布 (2019-20 基准年)")
print("-" * 80)
chap = long_df[long_df['Year'] == '2019-20'].groupby(['Chapter_Num', 'Chapter_Name']).agg(
    Categories=('Code', 'count'),
    Total_Admissions=('Admissions', 'sum')
).reset_index()
chap['Total_Admissions'] = chap['Total_Admissions'].astype(int)
chap = chap.sort_values('Total_Admissions', ascending=False)
print(chap.to_string(index=False))

# =========================================================
# 4. Spot-check: key disease codes across 5 years
# =========================================================
print("\n【4】 关键疾病 5 年轨迹检查 (Admissions)")
print("-" * 80)
checks = [
    ('F50-F59', 'Eating disorders'),
    ('U00-U49', 'COVID-19 (U codes)'),
    ('J20-J22', 'Lower resp infections'),
    ('M00-M25', 'Arthropathies'),
    ('I20-I25', 'Ischaemic heart disease'),
    ('K00-K14', 'Oral cavity'),
    ('Z30-Z39', 'Pregnancy/birth-related'),
]

pivot = long_df.pivot_table(index='Code', columns='Year', values='Admissions', aggfunc='first')
for code, label in checks:
    if code in pivot.index:
        vals = pivot.loc[code]
        vals_str = '  '.join(f"{int(v):>9,}" if pd.notna(v) else f"{'-':>9}" for v in vals)
        print(f"  {code:<10} {label:<25} {vals_str}")

# =========================================================
# 5. Age column completeness
# =========================================================
print("\n【5】 年龄列完整性检查 (non-null 计数，应全部 = 1058)")
print("-" * 80)
age_cols = [c for c in long_df.columns if c.startswith('Age ')]
age_nulls = long_df[age_cols].notna().sum()
for col in age_cols:
    n = age_nulls[col]
    flag = "✓" if n == len(long_df) else "⚠"
    print(f"  {flag} {col:<12}: {n} non-null")

# =========================================================
# 6. Eating disorders age distribution (the golden finding)
# =========================================================
print("\n【6】 进食障碍 F50-F59 年龄分布验证 (10-14 岁应翻倍)")
print("-" * 80)
ed = long_df[long_df['Code'] == 'F50-F59'].set_index('Year')
sel_ages = ['Age 10-14', 'Age 15', 'Age 16', 'Age 17', 'Age 20-24', 'Age 30-34', 'Age 40-44']
print(f"  {'Year':<10} " + '  '.join(f"{a:>10}" for a in sel_ages))
for y in ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']:
    if y in ed.index:
        vals = [f"{int(ed.loc[y, a]):>10,}" if pd.notna(ed.loc[y, a]) else f"{'-':>10}" for a in sel_ages]
        print(f"  {y:<10} " + '  '.join(vals))

# =========================================================
# 7. Admission mode breakdown
# =========================================================
print("\n【7】 入院方式列检查 (Emergency/Waiting/Planned 应都有数据)")
print("-" * 80)
mode_check = long_df.groupby('Year')[['Emergency', 'Waiting list', 'Planned']].sum().astype(int)
print(mode_check.to_string())

print("\n" + "=" * 80)
print("✅ 如果以上数据都合理，说明 5 年数据已正确加载")
print("=" * 80)