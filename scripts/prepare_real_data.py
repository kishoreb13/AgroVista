"""Prepare an India district/crop/season/year APY CSV for AgroVista.

This script does not generate observations. It standardises a downloaded official
APY file and derives yield (tonnes/ha) only when production and area are present.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

ALIASES = {
    'state': ['state','state_name','state/ut'],
    'district': ['district','district_name'],
    'crop': ['crop','crop_name','commodity'],
    'season': ['season','crop_season'],
    'year': ['year','crop_year','year_code'],
    'area': ['area','area_ha','area_(ha)','area_1000_ha','area_in_ha'],
    'production': ['production','production_tonnes','production_(tonnes)','production_1000_tonnes'],
    'yield': ['yield','yield_tonnes_ha','yield_(kg/ha)','yield_kg_ha'],
}
def norm(x): return re.sub(r'[^a-z0-9]+','_',str(x).strip().lower()).strip('_')
def find_col(df, key):
    lookup={norm(c):c for c in df.columns}
    for a in ALIASES[key]:
        if norm(a) in lookup: return lookup[norm(a)]
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to official downloaded APY CSV')
    ap.add_argument('--output', default='data/processed/agrovista_real_apy.csv')
    ap.add_argument('--area-unit', choices=['ha','thousand_ha'], default='ha')
    ap.add_argument('--production-unit', choices=['tonnes','thousand_tonnes'], default='tonnes')
    args=ap.parse_args()
    src=Path(args.input)
    if not src.exists(): raise SystemExit(f'Input not found: {src}')
    raw=pd.read_csv(src, low_memory=False)
    mapping={k:find_col(raw,k) for k in ALIASES}
    required=['state','district','crop','season','year','area','production']
    missing=[k for k in required if mapping[k] is None]
    if missing:
        print('Detected columns:', list(raw.columns))
        raise SystemExit('Could not identify required fields: '+', '.join(missing)+'. Edit ALIASES in this script to match the downloaded file.')
    out=pd.DataFrame()
    for k,c in mapping.items():
        if c: out[k]=raw[c]
    for k in ['area','production']:
        out[k]=pd.to_numeric(out[k].astype(str).str.replace(',','',regex=False),errors='coerce')
    if args.area_unit=='thousand_ha': out['area']*=1000
    if args.production_unit=='thousand_tonnes': out['production']*=1000
    if mapping['yield']:
        y=pd.to_numeric(raw[mapping['yield']].astype(str).str.replace(',','',regex=False),errors='coerce')
        # Explicitly convert kg/ha to tonnes/ha when source field says kg/ha.
        if 'kg' in norm(mapping['yield']): y=y/1000
        out['yield_tonnes_ha']=y
    else:
        out['yield_tonnes_ha']=out['production']/out['area'].where(out['area']>0)
    out['year']=out['year'].astype(str).str.extract(r'(\d{4})',expand=False)
    out['state']=out['state'].astype(str).str.strip()
    out['district']=out['district'].astype(str).str.strip()
    out['crop']=out['crop'].astype(str).str.strip()
    out['season']=out['season'].astype(str).str.strip()
    out=out.dropna(subset=['state','district','crop','season','year','yield_tonnes_ha'])
    out=out[(out['area']>0)&(out['production']>=0)&(out['yield_tonnes_ha']>=0)]
    out=out.drop_duplicates(subset=['state','district','crop','season','year'])
    out['data_source']='Government of India APY statistics; see docs/REAL_DATA_GUIDE.md'
    out['is_synthetic']=False
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(f'Saved {len(out):,} real APY rows to {args.output}')
    print('Columns:', ', '.join(out.columns))
    print('Yield = production (tonnes) / area (hectares), unless source yield was supplied.')
    print('Verify the source units and data dictionary before interpreting results.')
if __name__=='__main__': main()
