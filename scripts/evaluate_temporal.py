"""Chronological yield-model evaluation for annual/seasonal APY records."""
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

p=argparse.ArgumentParser(); p.add_argument('--input',default='data/processed/agrovista_real_enriched.csv'); p.add_argument('--target',default='yield_tonnes_ha'); p.add_argument('--year',default='year'); a=p.parse_args()
df=pd.read_csv(a.input)
if a.target not in df or a.year not in df: raise SystemExit(f'Required columns not found. Available: {list(df.columns)}')
df[a.year]=pd.to_numeric(df[a.year],errors='coerce'); df[a.target]=pd.to_numeric(df[a.target],errors='coerce'); df=df.dropna(subset=[a.year,a.target]).copy()
years=sorted(df[a.year].unique())
if len(years)<3: raise SystemExit('At least 3 distinct years are required for a meaningful chronological holdout.')
test_years=years[-max(1,round(len(years)*.2)):]; train=df[~df[a.year].isin(test_years)]; test=df[df[a.year].isin(test_years)]
exclude={a.target,'production','production_tonnes','area','area_ha','farm_area_ha','profit','profit_inr','revenue','revenue_inr','water_efficiency','water_productivity','is_synthetic'}
Xtr=train.drop(columns=[c for c in exclude if c in train]); Xte=test.drop(columns=[c for c in exclude if c in test]); ytr=train[a.target]; yte=test[a.target]
# Drop identifiers and constant features; keep year as a feature only if it is known at prediction time.
Xtr=Xtr.drop(columns=[c for c in Xtr if c.lower() in ['id','record_id','farm_id']]); Xte=Xte.reindex(columns=Xtr.columns)
Xtr=Xtr.loc[:,Xtr.nunique(dropna=True)>1]; Xte=Xte[Xtr.columns]
cat=Xtr.select_dtypes(include=['object','category','bool']).columns.tolist(); num=[c for c in Xtr.columns if c not in cat]
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Random Forest':RandomForestRegressor(n_estimators=250,min_samples_leaf=2,max_features=0.8,random_state=42,n_jobs=-1),
        'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=180,learning_rate=.06,l2_regularization=1.0,random_state=42)}
print(f'Train years: {min(train[a.year]):.0f}-{max(train[a.year]):.0f} ({len(train)} rows)')
print(f'Test years: {min(test[a.year]):.0f}-{max(test[a.year]):.0f} ({len(test)} rows)')
print(f'Target: {a.target}; unit: verify source metadata (expected tonnes/ha for derived APY yield)')
for name,model in models.items():
    pipe=Pipeline([('prep',pre),('model',model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    print(f'{name}: MAE={mean_absolute_error(yte,pred):.4f}, RMSE={np.sqrt(mean_squared_error(yte,pred)):.4f}, R2={r2_score(yte,pred):.4f}')
print('Caution: results are dataset-specific. Check district overlap, missingness, duplicate rows, and feature leakage before reporting.')
