import os
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, 'output', 'data')
os.makedirs(_OUT_DIR, exist_ok=True)

LABEL_MAP = [
    ('A00-A09',   'Intestinal infections',                  'Shortened equivalent of "intestinal infectious diseases" — same scope.'),
    ('A92-A99',   'Vector-borne viral fevers',              '"Vector-borne" is the standard public-health synonym for arthropod-borne (mosquitoes, ticks, lice, fleas). Same scope.'),
    ('C81-C96',   'Lymphoid & blood-forming cancers',       'Covers the same scope as the ICD category (lymphomas, leukaemias, myeloma).'),
    ('D10-D36',   'Benign neoplasms',                       'Identical to ICD text.'),
    ('E70-E90',   'Metabolic disorders',                    'Identical to ICD text.'),
    ('E65-E68',   'Obesity & overeating',                   '"Overeating" is plain English for "hyperalimentation"; same scope.'),
    ('F00-F09',   'Organic mental disorders',               'Shortened equivalent of "organic, including symptomatic, mental disorders"; same scope.'),
    ('F50-F59',   'Eating, sleep & behavioural disorders',  'Directly names the three main subgroups (F50 eating, F51 sleep, F52-F59 behavioural/sexual) of the ICD block.'),
    ('G40-G47',   'Episodic neurological disorders',        'Generic equivalent of "episodic and paroxysmal disorders" — keeps full scope (epilepsy, migraine, TIA, sleep attacks, etc.).'),
    ('G50-G59',   'Nerve & plexus disorders',               'Covers the same scope as "nerve, nerve root and plexus disorders".'),
    ('H25-H28',   'Cataract & lens disorders',              'H25-H28 covers cataract plus other lens disorders — label explicitly names both. Same scope.'),
    ('H65-H75',   'Middle-ear disease',                     'Shortened equivalent (mastoid disorders are anatomically part of the middle ear).'),
    ('I30-I52',   'Other heart disease',                    'Identical to ICD text.'),
    ('I20-I25',   'Ischaemic heart disease',                'Identical to ICD text.'),
    ('J09-J18',   'Influenza & pneumonia',                  'Identical to ICD text.'),
    ('J20-J22',   'Acute lower respiratory infections',     'Shortened (dropped the redundant "other"); same scope.'),
    ('K55-K64',   'Intestinal disorders',                   'Shortened equivalent of "other diseases of intestines".'),
    ('K00-K14',   'Oral, salivary & jaw disorders',         'Directly names the three anatomical regions in the ICD description; same scope.'),
    ('L00-L08',   'Skin infections',                        'Shortened equivalent of "infections of the skin and subcutaneous tissue".'),
    ('L60-L75',   'Hair, nail & sweat-gland disorders',     'ICD "skin appendages" comprises exactly these structures.'),
    ('M00-M25',   'Joint disorders (arthropathies)',        'Lay English with clinical term retained in brackets; same scope.'),
    ('M40-M54',   'Back & spine disorders',                 'ICD "dorsopathies" = back/neck/spine disorders; same scope.'),
    ('N30-N39',   'Urinary system disorders',               'Shortened equivalent of "other diseases of the urinary system".'),
    ('N60-N64',   'Breast disorders',                       'Identical to ICD text.'),
    ('O94-O99',   'Other obstetric conditions',             'Shortened from "other obstetric conditions, not elsewhere classified".'),
    ('O80-O84',   'Delivery (childbirth)',                  'Added "(childbirth)" as lay clarifier; same scope.'),
    ('P20-P29',   'Newborn respiratory & circulatory',     'Uses "circulatory" (not "heart") to match ICD "cardiovascular"; same scope.'),
    ('P35-P39',   'Newborn infections',                     'Shortened equivalent of "infections specific to the perinatal period".'),
    ('Q20-Q28',   'Congenital circulatory defects',         'Uses "circulatory" (not "heart") to preserve ICD scope including great vessels.'),
    ('Q10-Q18',   'Congenital eye, ear & face defects',     'Shortened from "congenital malformations of eye, ear, face and neck"; covers same scope.'),
    ('R10-R19',   'Digestive symptoms (undiagnosed)',       'Plain-English paraphrase of "symptoms & signs involving the digestive system & abdomen"; same scope (undiagnosed).'),
    ('R83-R89',   'Abnormal lab/tissue findings',           'Plain-English paraphrase of "abnormal findings on examination of other body fluids, substances and tissues, without diagnosis"; same scope.'),
    ('T80-T88',   'Complications of surgical/medical care', 'Identical scope.'),
    ('T90-T98',   'Long-term consequences of injury',       'Plain-English for "sequelae"; same scope.'),
    ('Z30-Z39',   'Reproductive health services',           'Covers antenatal care, contraception, fertility services — same scope as "health services in circumstances related to reproduction".'),
    ('Z00-Z13',   'Examination & investigation',            'Identical to ICD text.'),
    ('U00-U49',   'Provisional codes (COVID-19 in 2020-24)', 'Kept ICD-10 header name ("provisional codes"), with bracketed note clarifying that during our observation window these codes represent COVID-19 (U07.1 / U07.2). Does not claim U00-U49 = COVID-19 in general.'),
]

df = pd.DataFrame(LABEL_MAP, columns=['Code', 'Short_Label', 'Rationale'])

sl = pd.read_csv(os.path.join(_OUT_DIR, 'shortlist_final.csv'))
merged = df.merge(sl[['Code', 'Chapter_Num', 'Chapter_Name', 'Description']],
                  on='Code', how='left')
merged = merged[['Chapter_Num', 'Chapter_Name', 'Code', 'Description', 'Short_Label', 'Rationale']]
merged.columns = ['Chapter', 'Chapter_Name', 'ICD_Code', 'Original_ICD_Description',
                  'Short_Label', 'Rationale']

out_csv = os.path.join(_OUT_DIR, 'icd_label_map.csv')
out_xlsx = os.path.join(_OUT_DIR, 'icd_label_map.xlsx')
merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
merged.to_excel(out_xlsx, index=False, sheet_name='ICD_label_map')

print(f"✅ Saved label map: {len(merged)} rows")
print(f"  {out_csv}")
print(f"  {out_xlsx}")

print("\n" + "=" * 118)
print(f"{'Code':<10} | {'Original ICD-10':<52} | {'Short label (figure)':<40}")
print("-" * 118)
for _, r in merged.iterrows():
    orig = r['Original_ICD_Description']
    if len(orig) > 50:
        orig = orig[:48] + '…'
    print(f"{r['ICD_Code']:<10} | {orig:<52} | {r['Short_Label']:<40}")