"""Download daily NASA POWER weather for locations listed in a CSV.

locations.csv columns: state,district,latitude,longitude
Outputs seasonal/year aggregates that can be merged with prepared APY data.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import requests

URL='https://power.larc.nasa.gov/api/temporal/daily/point'
SEASONS={
 'Kharif':('06-01','10-31'),
 'Rabi':('11-01','03-31'),
 'Zaid':('03-01','05-31'),
}
def get_year_weather(row, year, session):
    start=f'{year}0101'; end=f'{year}1231'
    params={'parameters':'PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M','community':'AG','longitude':row.longitude,'latitude':row.latitude,'start':start,'end':end,'format':'JSON'}
    r=session.get(URL,params=params,timeout=90); r.raise_for_status()
    payload=r.json()['properties']['parameter']
    records=[]
    for date in payload['PRECTOTCORR']:
        records.append({'date':pd.to_datetime(date),'rainfall_mm':payload['PRECTOTCORR'][date],
                        'temperature_c':payload['T2M'][date],'temperature_max_c':payload['T2M_MAX'][date],
                        'temperature_min_c':payload['T2M_MIN'][date],'humidity_percent':payload['RH2M'][date]})
    d=pd.DataFrame(records); d=d.replace(-999,np.nan)
    d['month']=d.date.dt.month
    for season,months in {'Kharif':[6,7,8,9,10],'Rabi':[11,12,1,2,3],'Zaid':[3,4,5]}.items():
        x=d[d.month.isin(months)]
        if not x.empty:
            records_out={'season':season,'year':year,'rainfall_mm':x.rainfall_mm.sum(min_count=1),
                         'temperature_c':x.temperature_c.mean(),'temperature_max_c':x.temperature_max_c.mean(),
                         'temperature_min_c':x.temperature_min_c.mean(),'humidity_percent':x.humidity_percent.mean(),
                         'weather_source':'NASA POWER Daily API'}
            yield records_out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--locations',required=True); p.add_argument('--start-year',type=int,default=2001); p.add_argument('--end-year',type=int,default=2025); p.add_argument('--output',default='data/processed/nasa_power_seasonal_weather.csv'); a=p.parse_args()
    loc=pd.read_csv(a.locations)
    needed={'state','district','latitude','longitude'}
    if not needed.issubset(loc.columns): raise SystemExit(f'locations file must contain {sorted(needed)}')
    session=requests.Session(); rows=[]
    for _,place in loc.iterrows():
        for year in range(a.start_year,a.end_year+1):
            try:
                for item in get_year_weather(place,year,session): rows.append({'state':place.state,'district':place.district,**item})
            except Exception as e: print(f'Skipping {place.state}/{place.district}/{year}: {e}')
            time.sleep(.15)
    out=pd.DataFrame(rows)
    if out.empty: raise SystemExit('No weather records downloaded.')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
    print(f'Saved {len(out):,} seasonal weather rows to {a.output}')
if __name__=='__main__': main()
