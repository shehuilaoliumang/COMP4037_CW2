# COMP4037 Coursework 2

This repository contains code for analysing NHS Hospital Episode Statistics (HES) primary diagnosis admissions (England, financial years 2019-20 to 2023-24) and producing shortlist tables plus heatmap figures.

## Repository Layout

- `data_loader.py`: load and standardise raw NHS Excel files into one long-format dataset.
- `validate_data.py`: run sanity checks on the generated dataset.
- `select_shortlist.py`: select the final ICD shortlist used in visual analysis.
- `build_label_map.py`: build lay-friendly ICD labels for plotting.
- `draft_three_column.py`: generate the draft three-panel figure.
- `main_heatmap.py`: generate the final main heatmap figure.
- `nhs_data/`: source NHS spreadsheets.
- `output/`: generated outputs (ignored by Git in this repository).

## Suggested Environment

- Python 3.11+
- Packages used by scripts: `pandas`, `numpy`, `matplotlib`, `openpyxl`

Install dependencies (if needed):

```bash
pip install pandas numpy matplotlib openpyxl
```

## Reproducible Run Order

Run from project root:

```bash
python data_loader.py
python validate_data.py
python select_shortlist.py
python build_label_map.py
python draft_three_column.py
python main_heatmap.py
```

## Main Outputs

Generated files are written under `output/`:

- `output/data/long_df.pkl`, `long_df.csv`, `long_df.xlsx`
- `output/data/shortlist_final.csv`, `shortlist_final.xlsx`
- `output/data/icd_label_map.csv`, `icd_label_map.xlsx`
- `output/drafts/v2_three_column_draft.png`
- `output/figures/v3_main_heatmap.png`

