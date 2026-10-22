"""Motor educativo de trading con IA. Solo datos sintéticos por defecto; nunca envía órdenes reales."""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    max_position: float = 1.0
    risk_fraction: float = 0.01
    annualization: int = 252

def make_synthetic_ohlcv(n=1000, seed=7, start="2020-01-01", freq="D"):
    """Genera OHLCV sintético determinista; no representa cotizaciones reales."""
    if n < 5: raise ValueError("n debe ser >= 5")
    rng=np.random.default_rng(seed)
    idx=pd.date_range(start, periods=n, freq=freq, tz="UTC")
    regimes=np.where(np.arange(n)//max(40,n//5)%2==0, 0.00035, -0.0001)
    rets=regimes+rng.normal(0,0.012,size=n)
    close=100*np.exp(np.cumsum(rets))
    open_=np.r_[close[0],close[:-1]]*(1+rng.normal(0,0.002,size=n))
    high=np.maximum(open_,close)*(1+rng.uniform(0,0.008,size=n))
    low=np.minimum(open_,close)*(1-rng.uniform(0,0.008,size=n))
    volume=rng.lognormal(9,0.45,size=n)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":volume},index=idx).rename_axis("timestamp")

def validate_ohlcv(data):
    required=["open","high","low","close","volume"]
    missing=set(required)-set(data.columns)
    if missing: raise ValueError(f"Faltan columnas OHLCV: {sorted(missing)}")
    df=data.copy()
    if df.index.has_duplicates: raise ValueError("Índice temporal duplicado")
    if not df.index.is_monotonic_increasing: raise ValueError("Índice temporal desordenado")
    if df[required].isna().any().any(): raise ValueError("Hay valores ausentes en OHLCV")
    if (df[["open","high","low","close"]] <= 0).any().any(): raise ValueError("Los precios deben ser positivos")
    if (df["volume"] < 0).any(): raise ValueError("El volumen no puede ser negativo")
    bad=(df.high < df[["open","close","low"]].max(axis=1)) | (df.low > df[["open","close","high"]].min(axis=1))
    if bad.any(): raise ValueError(f"Hay velas OHLC incoherentes: {int(bad.sum())}")
    return df

def build_features(data):
    df=validate_ohlcv(data)
    c=df.close; r=c.pct_change()
    out=pd.DataFrame(index=df.index)
    out["ret_1"]=r
    for lag in (2,3,5): out[f"ret_lag_{lag}"]=r.shift(lag)
    for w in (5,10,20):
        out[f"volatility_{w}"]=r.rolling(w).std()
        out[f"momentum_{w}"]=c.pct_change(w)
        out[f"ma_distance_{w}"]=c/c.rolling(w).mean()-1
    out["range_pct"]=(df.high-df.low)/c
    out["volume_change"]=df.volume.pct_change().replace([np.inf,-np.inf],np.nan)
    out["volume_relative"]=df.volume/df.volume.rolling(20).mean()
    out=out.replace([np.inf,-np.inf],np.nan)
    return out

def build_dataset(data, horizon=1, threshold=0.0):
    X=build_features(data)
    future=data.close.shift(-horizon)/data.close-1
    y=(future>threshold).astype(int).rename("target")
    valid=X.notna().all(axis=1) & future.notna()
    return X.loc[valid], y.loc[valid], future.loc[valid]

def chronological_split(X,y,train_fraction=0.7, validation_fraction=0.15):
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction+validation_fraction>=1:
        raise ValueError("Fracciones inválidas")
    n=len(X); a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
    return (X.iloc[:a],y.iloc[:a]),(X.iloc[a:b],y.iloc[a:b]),(X.iloc[b:],y.iloc[b:])

def make_model(kind="logistic", seed=7):
    if kind=="logistic":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))])
    if kind in ("forest","random_forest"):
        return RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=10,class_weight="balanced",random_state=seed,n_jobs=-1)
    raise ValueError("kind debe ser 'logistic' o 'forest'")

def _positive_probability(model,X):
    p=model.predict_proba(X)
    classes=list(model.classes_)
    return p[:,classes.index(1)] if 1 in classes else np.zeros(len(X))

def walk_forward_probabilities(X,y,initial_fraction=0.6,step=100,kind="logistic"):
    if step<1: raise ValueError("step debe ser positivo")
    n=len(X); start=max(30,int(n*initial_fraction))
    if start>=n: raise ValueError("No hay observaciones posteriores al entrenamiento inicial")
    result=pd.Series(np.nan,index=X.index,name="prob_up",dtype=float)
    for left in range(start,n,step):
        right=min(n,left+step)
        model=make_model(kind)
        model.fit(X.iloc[:left],y.iloc[:left])
        result.iloc[left:right]=_positive_probability(model,X.iloc[left:right])
    return result

def classify_regime(data, window=20, quantile=0.7):
    r=data.close.pct_change()
    vol=r.rolling(window).std()
    threshold=vol.expanding(min_periods=window).quantile(quantile)
    return pd.Series(np.where(vol>threshold,"high_volatility","normal"),index=data.index,name="regime").where(vol.notna())

def tune_logistic_temporally(X,y, candidates=None, train_fraction=0.65):
    """Selección temporal sencilla: el último bloque es validación, no test final."""
    candidates=candidates or [0.01,0.1,1.0,10.0]
    cut=int(len(X)*train_fraction)
    if cut<20 or cut>=len(X): raise ValueError("Muestra insuficiente para separar entrenamiento y validación")
    rows=[]
    for c in candidates:
        m=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=float(c),max_iter=1000,class_weight="balanced"))])
        m.fit(X.iloc[:cut],y.iloc[:cut])
        p=_positive_probability(m,X.iloc[cut:])
        rows.append({"C":float(c),"brier":float(brier_score_loss(y.iloc[cut:],p)),"accuracy":float(accuracy_score(y.iloc[cut:],p>=0.5))})
    return pd.DataFrame(rows).sort_values(["brier","C"]).reset_index(drop=True)

def size_position(equity, price, stop_distance, risk_fraction=0.01, max_position=1.0):
    if equity<=0 or price<=0 or stop_distance<=0 or risk_fraction<0 or max_position<0: raise ValueError("Parámetros de riesgo inválidos")
    units=(equity*risk_fraction)/(price*stop_distance)
    return float(min(units, equity*max_position/price))

def backtest_positions(data, positions, config=None):
    config=config or BacktestConfig()
    df=validate_ohlcv(data)
    pos=pd.Series(positions,index=df.index,dtype=float).fillna(0).clip(-config.max_position,config.max_position)
    # La posición de la barra anterior captura el siguiente rendimiento: no se opera retrospectivamente.
    returns=df.close.pct_change().fillna(0)
    held=pos.shift(1).fillna(0)
    turnover=pos.diff().abs().fillna(pos.abs())
    net=held*returns-turnover*(config.fee_rate+config.slippage)
    equity=config.initial_capital*(1+net).cumprod()
    return pd.DataFrame({"close":df.close,"position":pos,"strategy_return":net,"equity":equity,"drawdown":equity/equity.cummax()-1},index=df.index)

def performance_metrics(result, annualization=252):
    r=result["strategy_return"].dropna()
    equity=result["equity"]
    vol=r.std(ddof=1)
    sharpe=float(np.sqrt(annualization)*r.mean()/vol) if len(r)>1 and vol>0 else 0.0
    peak=equity.cummax()
    return {"total_return":float(equity.iloc[-1]/equity.iloc[0]-1),"annualized_volatility":float(vol*np.sqrt(annualization)) if len(r)>1 else 0.0,"sharpe":sharpe,"max_drawdown":float((equity/peak-1).min()),"bars":int(len(r))}

def monte_carlo_final_equity(returns, simulations=1000, seed=7, initial_capital=10000):
    r=pd.Series(returns).dropna().to_numpy()
    if len(r)<2 or simulations<1: raise ValueError("Se requieren al menos dos retornos y una simulación")
    rng=np.random.default_rng(seed)
    samples=rng.choice(r,size=(simulations,len(r)),replace=True)
    final=initial_capital*np.prod(1+samples,axis=1)
    return {"p05":float(np.quantile(final,.05)),"median":float(np.median(final)),"p95":float(np.quantile(final,.95)),"simulations":int(simulations)}

def portfolio_returns(returns, weights=None):
    frame=pd.DataFrame(returns).astype(float)
    if weights is None: weights=np.repeat(1/frame.shape[1],frame.shape[1])
    weights=np.asarray(weights,dtype=float)
    if len(weights)!=frame.shape[1] or not np.isclose(weights.sum(),1): raise ValueError("Los pesos deben corresponder a las columnas y sumar 1")
    return frame.fillna(0).dot(weights).rename("portfolio_return")

def classification_metrics(y_true, probabilities, threshold=0.5):
    y=pd.Series(y_true).astype(int); p=pd.Series(probabilities,index=y.index).astype(float).clip(0,1)
    pred=(p>=threshold).astype(int)
    out={"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"brier":float(brier_score_loss(y,p))}
    out["roc_auc"]=float(roc_auc_score(y,p)) if y.nunique()==2 else float("nan")
    return out

def compare_strategies(data, probabilities, threshold=0.55, config=None):
    p=pd.Series(probabilities,index=data.index).reindex(data.index)
    rule=(data.close>data.close.rolling(20).mean()).astype(float).fillna(0)
    ai=(p>=threshold).astype(float).fillna(0)
    hybrid=(rule.eq(1)&ai.eq(1)).astype(float)
    rows=[]
    for name,pos in (("rules",rule),("ai",ai),("hybrid",hybrid)):
        bt=backtest_positions(data,pos,config)
        rows.append({"strategy":name,**performance_metrics(bt)})
    return pd.DataFrame(rows)

def run_experiment(n=700, seed=7, output_dir="salida"):
    data=make_synthetic_ohlcv(n=n,seed=seed)
    X,y,_=build_dataset(data)
    if len(X)<100: raise ValueError("Se necesitan más barras para el experimento")
    prob=walk_forward_probabilities(X,y,initial_fraction=0.6,step=100)
    # Reindex probabilities al calendario de datos, manteniendo NaN como ausencia de señal.
    positions=pd.Series(0.0,index=data.index)
    positions.loc[prob.index]=np.where(prob.fillna(0)>=0.55,1.0,0.0)
    bt=backtest_positions(data,positions)
    metrics=performance_metrics(bt)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    bt.to_csv(out/"backtest_curve.csv",index_label="timestamp")
    comparison=compare_strategies(data,positions.astype(float).where(positions.index.isin(prob.index)),config=BacktestConfig())
    comparison.to_csv(out/"strategy_comparison.csv",index=False)
    payload={"note":"Experimento educativo con datos sintéticos; no demuestra rentabilidad futura.","bars":int(n),"strategy":"walk-forward logistic + risk-gated simulated backtest","seed":int(seed),"metrics":metrics,"config":asdict(BacktestConfig())}
    (out/"experiment_results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    return payload

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars",type=int,default=2500)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--output",default="salida")
    args=parser.parse_args()
    print(json.dumps(run_experiment(args.bars,args.seed,args.output),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
