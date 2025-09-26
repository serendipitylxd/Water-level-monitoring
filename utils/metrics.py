# utils/metrics.py
# -*- coding: utf-8 -*-
import numpy as np

def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = {}
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]; yp = y_pred[mask]
    m["count"] = int(mask.sum())
    if m["count"] == 0:
        m.update(dict(MAE=None, RMSE=None, Bias=None, Corr=None)); return m
    m["MAE"]  = float(np.mean(np.abs(yp - yt)))
    m["RMSE"] = float(np.sqrt(np.mean((yp - yt)**2)))
    m["Bias"] = float(np.mean(yp - yt))
    if len(yt) >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
        m["Corr"] = float(np.corrcoef(yt, yp)[0,1])
    else:
        m["Corr"] = None
    return m

