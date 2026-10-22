import numpy as np
import pandas as pd
import pytest
from motor_trading_ia import (
    BacktestConfig, make_synthetic_ohlcv, validate_ohlcv, build_features,
    build_dataset, chronological_split, make_model, backtest_positions,
    performance_metrics, size_position, portfolio_returns, run_experiment,
    walk_forward_probabilities, classification_metrics, compare_strategies
)

def test_synthetic_reproducible_and_valid():
    a=make_synthetic_ohlcv(150,seed=9)
    b=make_synthetic_ohlcv(150,seed=9)
    pd.testing.assert_frame_equal(a,b)
    assert len(validate_ohlcv(a))==150

def test_validation_rejects_bad_price():
    d=make_synthetic_ohlcv(30)
    d.iloc[5,d.columns.get_loc("close")]=-1
    with pytest.raises(ValueError): validate_ohlcv(d)

def test_features_and_dataset_temporal_shapes():
    d=make_synthetic_ohlcv(250)
    X,y,future=build_dataset(d)
    assert len(X)==len(y)==len(future)
    assert X.notna().all().all()
    assert set(y.unique()).issubset({0,1})
    assert X.index.max()<d.index[-1]

def test_chronological_split_keeps_order():
    d=make_synthetic_ohlcv(300); X,y,_=build_dataset(d)
    tr,va,te=chronological_split(X,y)
    assert tr[0].index.max()<va[0].index.min()
    assert va[0].index.max()<te[0].index.min()

def test_model_fit_predict():
    d=make_synthetic_ohlcv(300); X,y,_=build_dataset(d)
    m=make_model(); m.fit(X.iloc[:150],y.iloc[:150])
    p=m.predict_proba(X.iloc[150:160])
    assert p.shape==(10,2)

def test_backtest_metrics_and_position_lag():
    d=make_synthetic_ohlcv(100)
    pos=pd.Series(1.0,index=d.index)
    bt=backtest_positions(d,pos,BacktestConfig(fee_rate=0,slippage=0))
    assert bt.position.iloc[0]==1
    assert bt.strategy_return.iloc[0]==0
    assert "max_drawdown" in performance_metrics(bt)

def test_size_position_respects_cap():
    assert size_position(10000,100,0.05,max_position=0.2) <= 20.000001

def test_portfolio_weights():
    r=pd.DataFrame({"a":[.1,0.0],"b":[0.0,.1]})
    p=portfolio_returns(r,[.5,.5])
    assert np.allclose(p,[.05,.05])

def test_classification_metrics():
    m=classification_metrics([0,1,0,1],[.1,.8,.3,.7])
    assert m["accuracy"]==1.0

def test_walk_forward_and_comparison():
    d=make_synthetic_ohlcv(400); X,y,_=build_dataset(d)
    p=walk_forward_probabilities(X,y,initial_fraction=.6,step=50)
    assert p.iloc[:int(len(X)*.6)].isna().all()
    assert p.notna().any()
    table=compare_strategies(d,p.reindex(d.index))
    assert set(table.strategy)=={"rules","ai","hybrid"}

def test_run_experiment_outputs(tmp_path):
    result=run_experiment(n=250,seed=7,output_dir=tmp_path)
    assert result["bars"]==250
    assert (tmp_path/"experiment_results.json").exists()
    assert (tmp_path/"backtest_curve.csv").exists()
    assert (tmp_path/"strategy_comparison.csv").exists()
