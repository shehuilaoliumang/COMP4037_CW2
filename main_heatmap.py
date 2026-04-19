import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ============ Paths ============
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL = os.path.join(_THIS_DIR, 'output', 'data', 'long_df.pkl')
_SHORTLIST = os.path.join(_THIS_DIR, 'output', 'data', 'shortlist_final.csv')
_LABEL_MAP = os.path.join(_THIS_DIR, 'output', 'data', 'icd_label_map.csv')
_OUT_DIR = os.path.join(_THIS_DIR, 'output', 'figures')
os.makedirs(_OUT_DIR, exist_ok=True)

# ============ Load ============
long_df = pd.read_pickle(_PKL)
shortlist = pd.read_csv(_SHORTLIST)
label_map = pd.read_csv(_LABEL_MAP)   # ICD short-label lookup (see Data Preparation)
row_codes = shortlist['Code'].tolist()

# Build the Code → short-label dictionary used for figure y-axis
short_label_lookup = dict(zip(label_map['ICD_Code'], label_map['Short_Label']))

pivot = long_df.pivot_table(index='Code', columns='Year', values='Admissions', aggfunc='first')
pivot = pivot.loc[row_codes]

# ============ Compute three panels ============
def pct_change(new, base):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((base > 0) & np.isfinite(base),
                        (new - base) / base * 100, np.nan)

shock    = pct_change(pivot['2020-21'].values, pivot['2019-20'].values)
recovery = pct_change(pivot['2022-23'].values, pivot['2020-21'].values)
legacy   = pct_change(pivot['2023-24'].values, pivot['2019-20'].values)
matrix   = np.vstack([shock, recovery, legacy]).T   # shape (37, 3)

# ============ Paletton-derived palette ============
# Diverging scale anchored at PURE WHITE so it stands out from the warm beige page.
# (Earlier versions reused the page beige as the heatmap centre — too camouflaged.)
PAL_DEEP_RED   = '#7A2E2E'    # strong drop    (Lockdown damage)
PAL_MID_RED    = '#C25F52'
PAL_LIGHT_RED  = '#EBC5BE'
PAL_CHART_BG   = '#FFFFFF'    # heatmap centre = crisp white (foreground)
PAL_LIGHT_BLUE = '#B9CAD9'
PAL_MID_BLUE   = '#567299'
PAL_DEEP_BLUE  = '#2E4A7A'    # strong rise

# Page background is warm and slightly darker, so chart floats on it
PAGE_BG        = '#EDE4D6'    # slightly darker beige → clearer figure/ground contrast

diverging = LinearSegmentedColormap.from_list(
    'paletton_div',
    [PAL_DEEP_RED, PAL_MID_RED, PAL_LIGHT_RED,
     PAL_CHART_BG,
     PAL_LIGHT_BLUE, PAL_MID_BLUE, PAL_DEEP_BLUE],
    N=256
)
diverging.set_bad(color='#c2c2c2')   # slightly darker grey for missing — visible on white

# Chapter band colours (muted so they frame, don't compete)
# Each hue picks up a secondary tone from the same Paletton harmony
CHAPTER_COLOURS = {
    'I':    '#B99B7D',   # warm sand       Infectious
    'II':   '#A68A6C',   # camel           Neoplasms
    'IV':   '#C1A875',   # ochre           Endocrine
    'V':    '#8C7B9B',   # dusty lavender  Mental
    'VI':   '#7C8FA3',   # slate           Nervous
    'VII':  '#A5B0A0',   # sage            Eye/Ear
    'IX':   '#C08275',   # terracotta      Circulatory
    'X':    '#D1A889',   # muted peach     Respiratory
    'XI':   '#A88F6E',   # bronze          Digestive
    'XII':  '#B8A59A',   # taupe           Skin
    'XIII': '#859187',   # moss            Musculoskeletal
    'XIV':  '#9A838F',   # mauve           Genitourinary
    'XV':   '#CC9B8A',   # rose-tan        Pregnancy
    'XVI':  '#AEA27A',   # olive           Perinatal
    'XVII': '#8B9E92',   # grey-green      Congenital
    'XVIII':'#A8988F',   # greige          Symptoms
    'XIX':  '#B38476',   # rust            Injury
    'XXI':  '#9DA7A0',   # stone           Health services
    'XXII': '#6E4A52',   # oxblood         COVID
}

# ============ Typography ============
# Use high-quality serif display + clean sans body — avoids generic AI look.
# Fall back gracefully if fonts missing on the user's system.
plt.rcParams.update({
    'font.family':    ['DejaVu Sans'],   # safe cross-platform fallback
    'font.size':      9,
    'axes.edgecolor': '#3a3a3a',
    'axes.linewidth': 0.6,
    'xtick.color':    '#3a3a3a',
    'ytick.color':    '#3a3a3a',
})

# ============ Clip extremes so they don't wash out the scale ============
CLIP = 100
matrix_clipped = np.clip(matrix, -CLIP, CLIP)
mask = ~np.isfinite(matrix)                  # True where missing
matrix_masked = np.ma.masked_array(matrix_clipped, mask=mask)

# ============ Soft desaturation for tiny-basis rows ============
# When baseline < 1,000 the % values become unstable; we flag them so the
# reader knows to take those cells with a pinch of salt.
tiny_basis = pivot['2019-20'].values < 1000

# ============ Figure layout ============
# Grid: [chapter_name_col | chapter_band | disease_labels | panel1 | panel2 | panel3 | colorbar]
# We keep the per-disease labels adjacent to the heatmap for readability,
# and put chapter names further to the left with brackets indicating span.
fig = plt.figure(figsize=(14.5, 15.5))   # taller to leave annotation bands top & bottom
fig.patch.set_facecolor(PAGE_BG)

gs = fig.add_gridspec(
    nrows=1, ncols=6,
    # Short labels fit comfortably in a narrower column, so we shrink the
    # labels column (1.50 → 1.20) and grow each heatmap panel (0.85 → 0.95).
    # This gives the actual data more visual weight.
    width_ratios=[0.48, 0.20, 1.20, 0.95, 0.95, 0.95],
    wspace=0.04
)
# Columns: chapter names (text-only ax) | chapter colour band | disease labels (text-only ax) | 3 panels
ax_chname = fig.add_subplot(gs[0, 0])
ax_band   = fig.add_subplot(gs[0, 1], sharey=ax_chname)
ax_labels = fig.add_subplot(gs[0, 2], sharey=ax_chname)
ax_shock  = fig.add_subplot(gs[0, 3], sharey=ax_chname)
ax_recov  = fig.add_subplot(gs[0, 4], sharey=ax_chname)
ax_legacy = fig.add_subplot(gs[0, 5], sharey=ax_chname)

# ============ Chapter colour band (column 2) ============
chapters = shortlist['Chapter_Num'].tolist()
band = np.zeros((len(chapters), 1, 3))
for i, ch in enumerate(chapters):
    hex_col = CHAPTER_COLOURS.get(ch, '#999999')
    rgb = tuple(int(hex_col[j:j+2], 16) / 255 for j in (1, 3, 5))
    band[i, 0] = rgb
ax_band.imshow(band, aspect='auto', interpolation='nearest')
ax_band.set_xticks([])
ax_band.set_yticks([])
ax_band.set_ylim(len(chapters) - 0.5, -0.5)
for s in ax_band.spines.values():
    s.set_visible(False)

# ============ Chapter names (column 1) ============
# Draw one label per chapter, vertically centred on its group
ax_chname.set_xlim(0, 1)
ax_chname.set_ylim(len(chapters) - 0.5, -0.5)
ax_chname.set_xticks([])
ax_chname.set_yticks([])
for s in ax_chname.spines.values():
    s.set_visible(False)
ax_chname.patch.set_alpha(0)

chapter_name_map = shortlist.set_index('Chapter_Num')['Chapter_Name'].to_dict()
prev = None
start_idx = 0
for i, ch in enumerate(chapters + [None]):
    if prev is not None and ch != prev:
        centre = (start_idx + i - 1) / 2
        label = f"{prev}.  {chapter_name_map.get(prev, '')}"
        # No hard truncation — the column is sized to fit full names now
        ax_chname.text(
            0.97, centre, label,
            ha='right', va='center',
            fontsize=8, color='#2a2a2a',
            fontweight='semibold'
        )
        start_idx = i
    prev = ch

# ============ Disease labels (column 3) ============
# Render each disease row's "CODE — description" using a text-only axis so
# we have precise control over alignment (monospace code + variable-width desc).
ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(len(chapters) - 0.5, -0.5)
ax_labels.set_xticks([])
ax_labels.set_yticks([])
for s in ax_labels.spines.values():
    s.set_visible(False)
ax_labels.patch.set_alpha(0)

desc_map = short_label_lookup   # Use lay-friendly labels built in build_label_map.py
for i, code in enumerate(row_codes):
    desc = str(desc_map.get(code, '')).strip()
    # Short labels are all under 45 chars by design; no truncation needed.
    # Code in bold monospace
    ax_labels.text(
        0.02, i, code,
        ha='left', va='center',
        fontsize=7.5, family='monospace',
        color='#1f1f1f', fontweight='bold'
    )
    # Description — further right so there's a comfortable gap after the code
    ax_labels.text(
        0.25, i, desc,
        ha='left', va='center',
        fontsize=7.5,
        color='#2a2a2a'
    )

# ============ Three heatmap panels ============
panel_axes = [ax_shock, ax_recov, ax_legacy]
panel_titles = ['SHOCK', 'RECOVERY', 'LEGACY']
panel_sub = [
    '2020-21  vs  2019-20',
    '2022-23  vs  2020-21',
    '2023-24  vs  2019-20',
]

im = None
for idx, ax in enumerate(panel_axes):
    col = matrix_masked[:, idx].reshape(-1, 1)
    im = ax.imshow(
        col,
        aspect='auto',
        cmap=diverging,
        vmin=-CLIP, vmax=CLIP,
        interpolation='nearest',
    )
    ax.set_xticks([])
    ax.set_yticks([])   # labels drawn in ax_labels now
    # Panel title (two-line)
    ax.set_title(
        f"{panel_titles[idx]}\n{panel_sub[idx]}",
        fontsize=10.5, fontweight='bold', color='#2a2a2a', pad=12
    )
    # Darker, slightly thicker spines so the panel reads as a framed chart on paper
    for s in ax.spines.values():
        s.set_color('#2a2a2a')
        s.set_linewidth(0.9)

# ============ Horizontal chapter separators across all panels ============
prev = None
for i, ch in enumerate(chapters):
    if prev is not None and ch != prev:
        for ax in panel_axes + [ax_band]:
            ax.axhline(i - 0.5, color='#3a3a3a', linewidth=0.4)
    prev = ch

# ============ Small-basis row markers ============
# Add a tiny diamond glyph to the far-right of each tiny-basis row as a gentle warning
for i, flag in enumerate(tiny_basis):
    if flag:
        for ax in panel_axes:
            ax.plot(0, i, marker='x', markersize=5, color='#666666',
                    markeredgewidth=0.8, alpha=0.55, clip_on=False)

# ============ Key-finding annotations (Step 4b) ============
# We selectively call out a small number of cells with annotation bubbles
# pointing to the insights most worth reading. Keeping the count low (2 now,
# may grow to 4-5 if visual density allows) to avoid clutter.
#
# Placement strategy:
#   - Annotations for rows NEAR THE TOP of the figure → put bubble to the
#     upper-right, using the left or right of the heatmap cluster as space.
#   - For bottom rows → lower-right space.
#   - Use `figure.transFigure` for the bubble position so we aren't at the
#     mercy of the data coordinate system (each panel has x-range [-0.5, 0.5]
#     which is too narrow for meaningful offsets).

annotations = [
    {
        # --- TOP BAND, above SHOCK column title ---
        'code':      'J20-J22',
        'panel_idx': 0,
        'text':      "Lower respiratory infections:\n-70% as social distancing\ncut community spread",
        'box_xy_fig': (0.36, 0.93),
    },
    {
        # --- TOP BAND, above RECOVERY column title ---
        'code':      'H25-H28',
        'panel_idx': 1,
        'text':      "Cataract surgery backlog\nunleashed: +153% rebound\nfrom the 2020-21 low",
        'box_xy_fig': (0.55, 0.93),
    },
    {
        # --- TOP BAND, above LEGACY column title ---
        'code':      'F50-F59',
        'panel_idx': 2,
        'text':      "Eating, sleep & behavioural\ndisorders remain +18%\nabove the 2019 baseline",
        'box_xy_fig': (0.75, 0.93),
    },
    {
        # --- BOTTOM BAND, below SHOCK column ---
        'code':      'U00-U49',
        'panel_idx': 0,
        'text':      "COVID-19 codes were first\nactivated in 2020: +1 748%",
        'box_xy_fig': (0.37, 0.075),
    },
]

code_to_row = {c: i for i, c in enumerate(row_codes)}
panel_ax_by_idx = [ax_shock, ax_recov, ax_legacy]

for ann in annotations:
    row_i = code_to_row[ann['code']]
    target_ax = panel_ax_by_idx[ann['panel_idx']]

    # Cell centre in the panel's data coords is (0, row_i); imshow x-axis is [-0.5, 0.5].
    # Convert to figure coords so we can draw the arrow with an absolute text box position.
    target_ax.annotate(
        ann['text'],
        xy=(0, row_i), xycoords=target_ax.transData,
        xytext=ann['box_xy_fig'], textcoords='figure fraction',
        fontsize=8.0,
        color='#1a1a1a',
        ha='left', va='center',
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor='#FDFBF6',
            edgecolor='#2a2a2a',
            linewidth=0.8,
            alpha=0.95,
        ),
        arrowprops=dict(
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.18',
            color='#2a2a2a',
            linewidth=0.9,
            shrinkA=4, shrinkB=4,
        ),
        annotation_clip=False,
        zorder=10,
    )
# Use fig.add_axes with manual positioning since our gridspec is packed tight
ax_cbar = fig.add_axes([0.965, 0.15, 0.012, 0.65])
cbar = fig.colorbar(
    im, cax=ax_cbar,
    orientation='vertical',
    extend='both',
    extendfrac=0.04,
)
cbar.set_label('% change (clipped ±100%)', fontsize=9, color='#2a2a2a')
cbar.ax.tick_params(labelsize=8, color='#5a5a5a')
cbar.outline.set_linewidth(0.5)
cbar.outline.set_edgecolor('#6a6a6a')

# ============ Title block ============
fig.suptitle(
    "Lockdown's Selective Toll on NHS Admissions",
    fontsize=16, fontweight='bold', color='#1a1a1a', y=0.985
)
fig.text(
    0.5, 0.963,
    "37 disease categories, 3 analytical lenses — England, 2019-2024",
    ha='center', fontsize=10, style='italic', color='#4a4a4a'
)

# Footer: legend + simplification note
grey_patch = mpatches.Patch(color='#c2c2c2', label='Missing data in source')
x_marker = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='#666666',
                      markeredgecolor='#666666', markersize=7, label='Small-basis series (baseline < 1 000)')
fig.legend(
    handles=[grey_patch, x_marker],
    loc='lower center', bbox_to_anchor=(0.5, 0.030),
    ncol=2, frameon=False, fontsize=8
)

fig.text(
    0.5, 0.010,
    "Disease labels are lay-friendly short forms of ICD-10 descriptions "
    "(never narrower than the original); full mapping in report.",
    ha='center', fontsize=7, style='italic', color='#5a5a5a'
)

plt.tight_layout(rect=(0.0, 0.11, 1.0, 0.86))
out_path = os.path.join(_OUT_DIR, 'v3_main_heatmap.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PAGE_BG)
print(f"✅ Saved: {out_path}")
plt.close()