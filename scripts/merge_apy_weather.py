from pathlib import Path
import argparse
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--apy',default='data/processed/agrovista_real_apy.csv'); p.add_argument('--weather',default='data/processed/nasa_power_seasonal_weather.csv'); p.add_argument('--output',default='data/processed/agrovista_real_enriched.csv'); a=p.parse_args()
left=pd.read_csv(a.apy); right=pd.read_csv(a.weather)
keys=['state','district','season','year']
for frame in (left,right):
    for k in keys:
        if k not in frame: raise SystemExit(f'Missing merge key: {k}')
        frame[k]=frame[k].astype(str).str.strip()
out=left.merge(right,on=keys,how='left',validate='many_to_one')
Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
print(f'Wrote {len(out):,} rows; weather coverage: {out.rainfall_mm.notna().mean():.1%}')
