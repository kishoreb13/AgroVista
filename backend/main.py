from pathlib import Path
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import f_oneway

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "demo_agricultural_dataset.csv"
app = FastAPI(title="AgroVista API", version="1.0.0", description="Seasonal agriculture intelligence and ML analytics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
df = pd.read_csv(DATA_PATH)
IS_DEMO = True

ALIASES = {
    "season": ["season","crop_season"], "crop": ["crop","crop_name"], "state": ["state"],
    "yield": ["yield_tonnes_ha","yield","crop_yield","yield_ton_ha"],
    "production": ["production_tonnes","production"], "profit": ["profit_inr","profit"],
    "revenue": ["revenue_inr","revenue"], "cost": ["cost_inr","cost"],
    "water_eff": ["water_efficiency","water_productivity"], "water": ["water_usage_l","water_usage"],
    "disease": ["disease_risk_score","disease_risk"], "pest": ["pest_risk_score","pest_risk"],
    "area": ["farm_area_ha","area_ha","farm_area"],
    "temperature": ["temperature_c","temperature","temp_c"],
    "rainfall": ["rainfall_mm","rainfall","rainfall_millimeters"],
    "irrigation": ["irrigation_method","irrigation","irrigation_type"]
}
def norm(s): return str(s).strip().lower().replace(" ","_").replace("-","_")
def col(key):
    lookup={norm(c):c for c in df.columns}
    for alias in ALIASES[key]:
        if alias in lookup: return lookup[alias]
    return None
def numeric(key, default=0):
    c=col(key)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series([default]*len(df), index=df.index)
def category(key, default="Unknown"):
    c=col(key)
    return df[c].fillna(default).astype(str) if c else pd.Series([default]*len(df), index=df.index)
def safe(v):
    if pd.isna(v) or not np.isfinite(float(v)): return None
    return round(float(v), 4)
def season_summary():
    d=pd.DataFrame({"Season":category("season"),"Yield":numeric("yield"),"Profit":numeric("profit"),"Production":numeric("production"),"WaterEff":numeric("water_eff")})
    g=d.groupby("Season").agg(avg_yield=("Yield","mean"),total_production=("Production","sum"),avg_profit=("Profit","mean"),avg_water_eff=("WaterEff","mean"),farms=("Season","size")).reset_index()
    return [{k:(v if isinstance(v,str) else safe(v)) for k,v in row.items()} for row in g.to_dict("records")]
def api_scores():
    y=numeric("yield"); p=numeric("profit"); w=numeric("water_eff"); r=(numeric("disease")+numeric("pest"))/2
    def mm(s, inverse=False):
        s=s.replace([np.inf,-np.inf],np.nan).fillna(s.median())
        lo,hi=s.quantile(.05),s.quantile(.95)
        z=((s-lo)/(hi-lo)) if hi>lo else s*0+0.5
        z=z.clip(0,1)
        return 1-z if inverse else z
    # Equal, transparent weights; risk is inverted. This is a proposed composite index, not an official standard.
    score=100*(.30*mm(y)+.25*mm(p)+.20*mm(w)+.25*mm(r,True))
    return score
class PredictionInput(BaseModel):
    rainfall_mm: float = Field(500, ge=0, le=3000)
    temperature_c: float = Field(27, ge=-10, le=60)
    humidity_percent: float = Field(65, ge=0, le=100)
    soil_ph: float = Field(6.5, ge=0, le=14)
    nitrogen_kg_ha: float = Field(100, ge=0)
    phosphorus_kg_ha: float = Field(45, ge=0)
    potassium_kg_ha: float = Field(50, ge=0)
    farm_area_ha: float = Field(2, gt=0)
    irrigation_method: str = "Drip"
    crop: str = "Rice"
    season: str = "Kharif"
@app.get("/api/health")
def health(): return {"status":"ok","rows":len(df),"demo_data":IS_DEMO}
@app.get("/api/metadata")
def metadata():
    return {"rows":len(df),"columns":list(df.columns),"demo_data":IS_DEMO,
            "seasons":sorted(category("season").unique().tolist()),"crops":sorted(category("crop").unique().tolist()),
            "states":sorted(category("state").unique().tolist()),"missing_cells":int(df.isna().sum().sum()),
            "duplicates":int(df.duplicated().sum())}
@app.post("/api/upload")
async def upload(file: UploadFile=File(...)):
    global df, IS_DEMO
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Please upload a CSV file.")
    try:
        candidate=pd.read_csv(BytesIO(await file.read()))
        if len(candidate)<10 or len(candidate.columns)<3: raise ValueError("Dataset must contain at least 10 rows and 3 columns.")
        if not any(norm(c) in ALIASES["yield"] for c in candidate.columns): raise ValueError("A yield target column is required (for example Yield_Tonnes_Ha).")
        df=candidate; IS_DEMO=False
        return {"message":"Dataset loaded successfully","rows":len(df),"columns":len(df.columns),"demo_data":False}
    except Exception as e: raise HTTPException(400,str(e))
@app.get("/api/dashboard")
def dashboard():
    y=numeric("yield"); p=numeric("profit"); prod=numeric("production")
    s=season_summary()
    return {"rows":len(df),"avg_yield":safe(y.mean()),"total_production":safe(prod.sum()),"avg_profit":safe(p.mean()),
            "avg_api":safe(api_scores().mean()),"seasons":s,"demo_data":IS_DEMO,
            "crop_yield":pd.DataFrame({"Crop":category("crop"),"Yield":y}).groupby("Crop").Yield.mean().sort_values(ascending=False).head(8).round(3).to_dict(),
            "monthly_note":"Seasonal dataset; no date/month field is assumed."}
@app.get("/api/analytics")
def analytics():
    return {"environment":{"rainfall":safe(numeric("yield").corr(numeric("yield"))*0+numeric("rainfall").mean()),
                           "temperature":safe(numeric("yield").corr(numeric("yield"))*0+numeric("temperature").mean()) if "temperature" in " ".join(norm(c) for c in df.columns) else None},
            "irrigation":pd.DataFrame({"Method":category("irrigation"),"Water":numeric("water"),"Yield":numeric("yield")}).groupby("Method").mean(numeric_only=True).round(3).reset_index().to_dict("records"),
            "api_by_season":pd.DataFrame({"Season":category("season"),"API":api_scores()}).groupby("Season").API.mean().round(2).to_dict(),
            "risk_by_season":pd.DataFrame({"Season":category("season"),"Risk":(numeric("disease")+numeric("pest"))/2}).groupby("Season").Risk.mean().round(2).to_dict(),
            "profit_by_season":pd.DataFrame({"Season":category("season"),"Profit":numeric("profit")}).groupby("Season").Profit.mean().round(2).to_dict()}
@app.get("/api/statistics")
def statistics():
    groups=[pd.to_numeric(numeric("yield")[category("season")==s],errors="coerce").dropna() for s in sorted(category("season").unique())]
    groups=[g for g in groups if len(g)>1]
    result={}
    if len(groups)>=2:
        f,p=f_oneway(*groups); result={"yield_anova_f":safe(f),"yield_anova_p":safe(p),"significant_at_05":bool(p<.05)}
    else: result={"message":"At least two seasons with two observations each are required."}
    return {"anova":result,"interpretation":"A p-value below 0.05 indicates evidence of a difference in group means; it does not establish causation.","season_counts":category("season").value_counts().to_dict()}
@app.get("/api/models")
def models():
    target=col("yield")
    if not target: raise HTTPException(400,"Yield target column not found.")
    data=df.copy()
    y=pd.to_numeric(data[target],errors="coerce")
    usable=y.notna()
    data=data.loc[usable].copy(); y=y.loc[usable]
    # Exclude leakage and identifiers/economic outcomes derived from yield.
    excluded={target}
    for k in ["production","profit","revenue","water_eff"]:
        c=col(k)
        if c: excluded.add(c)
    excluded.update([c for c in data.columns if norm(c) in ["farm_id","id","record_id"]])
    X=data.drop(columns=list(excluded),errors="ignore")
    X=X.loc[:,X.nunique(dropna=True)>1]
    if len(X)<30 or X.shape[1]==0: raise HTTPException(400,"Not enough usable rows/features to train a model.")
    cat=X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num=X.columns.difference(cat).tolist()
    pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                           ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    estimators={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1),
                "Gradient Boosting":GradientBoostingRegressor(n_estimators=150,learning_rate=.05,max_depth=3,random_state=42)}
    scores=[]; best=None; best_r=-1e9; importances=[]
    for name,est in estimators.items():
        pipe=Pipeline([("preprocess",pre),("model",est)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        r2=r2_score(yte,pred)
        scores.append({"model":name,"mae":safe(mean_absolute_error(yte,pred)),"rmse":safe(np.sqrt(mean_squared_error(yte,pred))),"r2":safe(r2)})
        if r2>best_r: best_r=r2; best=(name,pipe,pred)
    name,pipe,pred=best
    perm=permutation_importance(pipe,Xte,yte,n_repeats=5,random_state=42,n_jobs=-1)
    importances=sorted([{"feature":str(f),"importance":safe(i)} for f,i in zip(X.columns,perm.importances_mean)],key=lambda x:x["importance"],reverse=True)[:12]
    return {"scores":scores,"best_model":name,"feature_importance":importances,
            "actual_vs_predicted":[{"actual":safe(a),"predicted":safe(b)} for a,b in zip(yte.iloc[:100],pred[:100])],
            "train_rows":len(Xtr),"test_rows":len(Xte),"target":target,
            "validation":"Single 80/20 holdout with random_state=42; metrics are dataset-dependent."}
@app.post("/api/predict")
def predict(inp:PredictionInput):
    # Train a model on current data and build a one-row prediction using matching feature names.
    target=col("yield")
    if not target: raise HTTPException(400,"Yield target missing.")
    data=df.copy(); y=pd.to_numeric(data[target],errors="coerce"); ok=y.notna(); data=data.loc[ok]; y=y.loc[ok]
    excluded={target}
    for k in ["production","profit","revenue","water_eff"]:
        c=col(k)
        if c: excluded.add(c)
    excluded.update([c for c in data.columns if norm(c) in ["farm_id","id","record_id"]])
    X=data.drop(columns=list(excluded),errors="ignore"); X=X.loc[:,X.nunique(dropna=True)>1]
    cat=X.select_dtypes(include=["object","category","bool"]).columns.tolist(); num=X.columns.difference(cat).tolist()
    pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                           ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    pipe=Pipeline([("preprocess",pre),("model",RandomForestRegressor(n_estimators=120,random_state=42,n_jobs=-1))])
    pipe.fit(X,y)
    values={"rainfall_mm":inp.rainfall_mm,"temperature_c":inp.temperature_c,"humidity_percent":inp.humidity_percent,"soil_ph":inp.soil_ph,
            "nitrogen_kg_ha":inp.nitrogen_kg_ha,"phosphorus_kg_ha":inp.phosphorus_kg_ha,"potassium_kg_ha":inp.potassium_kg_ha,
            "farm_area_ha":inp.farm_area_ha,"irrigation_method":inp.irrigation_method,"crop":inp.crop,"season":inp.season}
    row={}
    for c in X.columns:
        row[c]=values.get(norm(c), data[c].mode().iloc[0] if data[c].notna().any() and c in cat else (float(pd.to_numeric(data[c],errors="coerce").median()) if c in num else 0))
    result=float(pipe.predict(pd.DataFrame([row],columns=X.columns))[0])
    return {"predicted_yield_tonnes_ha":safe(result),"model":"Random Forest Regressor","note":"Scenario estimate, not a guarantee. Unspecified input features use dataset medians/modes."}

@app.post("/api/deep-learning")
def deep_learning():
    try:
        from backend.deep_learning import run_deep_learning
        target=col("yield")
        if not target: raise HTTPException(400,"Yield target column not found.")
        return run_deep_learning(df,target,epochs=45)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500,str(e))

app.mount("/",StaticFiles(directory=BASE/"frontend",html=True),name="frontend")
