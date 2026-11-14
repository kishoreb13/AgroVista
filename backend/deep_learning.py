"""Deep-learning experiments for AgroVista.

Models:
- Tabular MLP regressor
- Residual MLP regressor
- Feature-vector 1D CNN regressor
- Dense autoencoder for unsupervised anomaly scoring

The 1D CNN treats the ordered preprocessed feature vector as a 1D signal. It is an
experimental tabular architecture, not a temporal CNN. LSTM/GRU are intentionally
not applied unless a genuine time/sequence column exists.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def _safe(v):
    try:
        v=float(v)
        return round(v,5) if np.isfinite(v) else None
    except Exception:
        return None

def run_deep_learning(frame: pd.DataFrame, target_col: str, epochs: int=45):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed. Install it with: pip install -r requirements-dl.txt") from exc

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    data=frame.copy()
    y=pd.to_numeric(data[target_col],errors="coerce")
    valid=y.notna()
    data=data.loc[valid].copy()
    y=y.loc[valid].astype(float)
    excluded={target_col}
    # Avoid obvious target leakage from outcomes derived from yield.
    aliases={"production_tonnes","production","profit_inr","profit","revenue_inr","revenue","water_efficiency","water_productivity","farm_id","id","record_id"}
    excluded.update(c for c in data.columns if str(c).strip().lower().replace(" ","_") in aliases)
    X=data.drop(columns=list(excluded),errors="ignore")
    X=X.loc[:,X.nunique(dropna=True)>1]
    if len(X)<50 or X.shape[1]<2:
        raise ValueError("Deep learning needs at least 50 usable rows and two input features.")
    cats=X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    nums=X.columns.difference(cats).tolist()
    prep=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)
    ])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    A=prep.fit_transform(Xtr).astype("float32")
    B=prep.transform(Xte).astype("float32")
    if A.shape[1]<2: raise ValueError("Preprocessing produced fewer than two model inputs.")
    ym=float(ytr.mean()); ys=float(ytr.std()) or 1.0
    yt=((ytr.to_numpy()-ym)/ys).astype("float32")
    yv=((yte.to_numpy()-ym)/ys).astype("float32")
    device="cpu"
    tx=torch.tensor(A,dtype=torch.float32); ty=torch.tensor(yt,dtype=torch.float32).view(-1,1)
    vx=torch.tensor(B,dtype=torch.float32)
    loader=DataLoader(TensorDataset(tx,ty),batch_size=min(64,len(tx)),shuffle=True)
    n_features=A.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(n_features,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.20),
                                   nn.Linear(128,64),nn.ReLU(),nn.Dropout(.12),nn.Linear(64,1))
        def forward(self,x): return self.net(x)
    class ResidualBlock(nn.Module):
        def __init__(self,w):
            super().__init__()
            self.block=nn.Sequential(nn.Linear(w,w),nn.BatchNorm1d(w),nn.ReLU(),nn.Dropout(.15),nn.Linear(w,w))
            self.act=nn.ReLU()
        def forward(self,x): return self.act(x+self.block(x))
    class ResMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.start=nn.Sequential(nn.Linear(n_features,96),nn.ReLU())
            self.body=nn.Sequential(ResidualBlock(96),ResidualBlock(96))
            self.head=nn.Sequential(nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x): return self.head(self.body(self.start(x)))
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.BatchNorm1d(32),nn.ReLU(),
                                    nn.Conv1d(32,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(8),nn.Flatten(),
                                    nn.Linear(64*8,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
        def forward(self,x): return self.conv(x.unsqueeze(1))

    models={"Deep MLP":MLP(),"Residual MLP":ResMLP(),"1D CNN (feature vector)":CNN1D()}
    results=[]; predictions={}; curves={}
    for name,model in models.items():
        model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        loss_fn=nn.HuberLoss()
        history=[]
        for epoch in range(epochs):
            model.train(); losses=[]
            for xb,yb in loader:
                optimizer.zero_grad()
                loss=loss_fn(model(xb),yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            pred=(model(vx).cpu().numpy().ravel()*ys)+ym
        actual=yte.to_numpy()
        mae=mean_absolute_error(actual,pred); rmse=np.sqrt(mean_squared_error(actual,pred)); r2=r2_score(actual,pred)
        results.append({"model":name,"mae":_safe(mae),"rmse":_safe(rmse),"r2":_safe(r2)})
        predictions[name]=[{"actual":_safe(a),"predicted":_safe(b)} for a,b in zip(actual[:100],pred[:100])]
        curves[name]=[_safe(x) for x in history]
    # Dense autoencoder: anomaly score = reconstruction error on the feature space.
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.Linear(64,16),nn.ReLU(),nn.Linear(16,8))
            self.decoder=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,64),nn.ReLU(),nn.Linear(64,n_features))
        def forward(self,x): return self.decoder(self.encoder(x))
    ae=Autoencoder()
    aeopt=torch.optim.Adam(ae.parameters(),lr=.001)
    for _ in range(max(20,epochs//2)):
        ae.train()
        for (xb,) in DataLoader(TensorDataset(tx),batch_size=min(64,len(tx)),shuffle=True):
            aeopt.zero_grad(); loss=((ae(xb)-xb)**2).mean(); loss.backward(); aeopt.step()
    ae.eval()
    with torch.no_grad():
        train_err=((ae(tx)-tx)**2).mean(dim=1).numpy()
        test_err=((ae(vx)-vx)**2).mean(dim=1).numpy()
    threshold=float(np.quantile(train_err,.95))
    anomaly_count=int((test_err>threshold).sum())
    best=max(results,key=lambda x: x["r2"] if x["r2"] is not None else -1e9)
    return {
        "models":results,"best_model":best["model"],"predictions":predictions,
        "loss_curves":curves,"autoencoder":{"threshold_95":_safe(threshold),"test_anomalies":anomaly_count,
          "test_rows":len(test_err),"anomaly_rate_percent":_safe(100*anomaly_count/max(1,len(test_err)))},
        "input_features":int(n_features),"train_rows":len(Xtr),"test_rows":len(Xte),
        "epochs":epochs,"device":device,
        "notes":["Tabular MLP and residual MLP learn nonlinear feature interactions.",
                 "1D CNN is applied to the encoded feature vector; it is not a time-series model.",
                 "Autoencoder flags unusual feature patterns; an anomaly is not automatically a bad or diseased farm.",
                 "A genuine LSTM/GRU experiment requires chronological observations grouped by farm/crop, not randomly ordered rows."]
    }
