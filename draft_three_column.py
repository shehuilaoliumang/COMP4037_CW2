import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ============ Paths ============
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL = os.path.join(_THIS_DIR, 'output', 'data', 'long_df.pkl')
_SHORTLIST = os.path.join(_THIS_DIR, 'output', 'data', 'shortlist_final.csv')
_OUT_DIR = os.path.join(_THIS_DIR, 'output', 'drafts')
os.makedirs(_OUT_DIR, exist_ok=True)

# ============ Load data ============
long_df = pd.read_pickle(_PKL)
shortlist = pd.read_csv(_SHORTLIST)

# Keep the shortlist's chapter order (37 rows)
row_codes = shortlist['Code'].tolist()

# Pivot to a Code × Year table for subtraction
pivot = long_df.pivot_table(index='Code', columns='Year', values='Admissions', aggfunc='first')

# Keep the 37 rows in the shortlist order
pivot = pivot.loc[row_codes]

# ============ Compute the three panels ============
# Helper to avoid division-by-zero issues
def pct_change(new, base):
    """(new - base) / base * 100, returning NaN where base is NaN/0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((base > 0) & np.isfinite(base),
                        (new - base) / base * 100, np.nan)

panel_shock = pd.Series(
    pct_change(pivot['2020-21'].values, pivot['2019-20'].values),
    index=pivot.index, name='SHOCK'
)
panel_recovery = pd.Series(
    pct_change(pivot['2022-23'].values, pivot['2020-21'].values),
    index=pivot.index, name='RECOVERY'
)
panel_legacy = pd.Series(
    pct_change(pivot['2023-24'].values, pivot['2019-20'].values),
    index=pivot.index, name='LEGACY'
)

# Combine into a single DataFrame with one column per panel
matrix_df = pd.concat([panel_shock, panel_recovery, panel_legacy], axis=1)

print("Matrix preview (first rows):")
print(matrix_df.head(10).round(1))
print(f"\nAny missing values? {matrix_df.isna().any().any()}")
print("Range per panel (min / max):")
for col in matrix_df.columns:
    print(f"  {col:<10}  min={matrix_df[col].min():.1f}   max={matrix_df[col].max():.1f}")

# ============ Build disease row labels: "Code — Short description" ============
desc_map = shortlist.set_index('Code')['Description'].to_dict()

def shorten(desc, n=42):
    d = str(desc).strip()
    return d if len(d) <= n else d[:n-1] + '…'

row_labels = [f"{c}  —  {shorten(desc_map.get(c, ''))}" for c in row_codes]

# ============ Plot three side-by-side heatmaps ============
# Clip extreme values so the colour scale stays readable
CLIP = 100  # show anything beyond ±100% as the same extreme colour

matrix = matrix_df.values.astype(float)
matrix_clipped = np.clip(matrix, -CLIP, CLIP)

# We need 3 separate Axes with a shared y-axis
fig, axes = plt.subplots(
    1, 3,
    figsize=(10, 12),
    gridspec_kw={'wspace': 0.15, 'width_ratios': [1, 1, 1]},
    sharey=True,
)

panel_titles = [
    'SHOCK\n(2020-21 vs 2019-20)',
    'RECOVERY\n(2022-23 vs 2020-21)',
    'LEGACY\n(2023-24 vs 2019-20)',
]

# Use a masked array so NaN renders as the "bad" colour (grey).
# Semantics we want:
#   NEGATIVE (drop in admissions) -> RED  (damage/suppression)
#   POSITIVE (rise in admissions) -> BLUE (rebound/surge)
# matplotlib's 'RdBu' (without _r) already gives us red-at-low, blue-at-high.
cmap = plt.get_cmap('RdBu').copy()
cmap.set_bad(color='#cccccc')  # grey for missing

for i, ax in enumerate(axes):
    col_data = matrix_clipped[:, i].reshape(-1, 1)      # n_rows × 1
    col_masked = np.ma.masked_invalid(col_data)
    im = ax.imshow(
        col_masked,
        aspect='auto',
        cmap=cmap,
        vmin=-CLIP, vmax=CLIP,
        interpolation='nearest',
    )
    ax.set_title(panel_titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks(range(len(row_codes)))

# Y-labels only on the first axis
axes[0].set_yticklabels(row_labels, fontsize=7)

# ============ Draw horizontal separators between ICD chapters ============
chapters_in_order = shortlist['Chapter_Num'].tolist()
for ax in axes:
    prev = None
    for i, ch in enumerate(chapters_in_order):
        if prev is not None and ch != prev:
            ax.axhline(i - 0.5, color='black', linewidth=0.6)
        prev = ch

# ============ Shared colourbar on the right ============
cbar = fig.colorbar(
    im, ax=axes.ravel().tolist(),
    orientation='vertical', fraction=0.025, pad=0.02,
    extend='both'
)
cbar.set_label(f'% change  (clipped at ±{CLIP}%)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Grey legend patch for missing data
grey_patch = mpatches.Patch(color='#cccccc', label='Missing data in source')
fig.legend(
    handles=[grey_patch],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.01),
    frameon=False,
    fontsize=8
)

# Overall title
fig.suptitle(
    "DRAFT v2 — Lockdown's Selective Toll on NHS Admissions\n"
    "37 diseases × 3 lenses (Shock / Recovery / Legacy)",
    fontsize=12, y=0.995
)

plt.tight_layout(rect=(0, 0.01, 1, 0.97))
out_path = os.path.join(_OUT_DIR, 'v2_three_column_draft.png')
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\n✅ Saved: {out_path}")
plt.close()