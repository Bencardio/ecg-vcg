"""Exporte le modèle PTB-XL entraîné en ONNX et fixe un seuil par énoncé
(F1 maximal sur le pli 9, entrées disposées en grille 3x4 comme dans l'outil).

    python export_ptbxl.py <racine PTB-XL> <dossier out> <dossier modeles>
"""
import ast
import json
import sys

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import precision_recall_curve

from xresnet1d import xresnet1d101

ROOT, OUT, DEST = sys.argv[1:4]
meta = json.load(open(OUT + "/xresnet1d101_ptbxl.json"))
codes, supers = meta["codes"], meta["supers"]
m = xresnet1d101(num_classes=len(codes) + 5, input_channels=12, kernel_size=5,
                 ps_head=0.5, lin_ftrs_head=[128])
m.load_state_dict(torch.load(OUT + "/xresnet1d101_ptbxl.pt"))
m.eval()
torch.onnx.export(m, torch.randn(1, 12, 250), DEST + "/xresnet1d101_ptbxl.onnx",
                  input_names=["signal"], output_names=["logits"],
                  dynamic_axes={"signal": {0: "n"}, "logits": {0: "n"}},
                  opset_version=17, dynamo=False)

db = pd.read_csv(ROOT + "ptbxl_database.csv", index_col="ecg_id")
db = db[db.strat_fold == 9]
db.scp_codes = db.scp_codes.apply(ast.literal_eval)
stm = pd.read_csv(ROOT + "scp_statements.csv", index_col=0)
diag = stm[stm.diagnostic == 1.0]
Y = np.zeros((len(db), len(codes) + 5))
for i, d in enumerate(db.scp_codes):
    for c in d:
        if c in codes:
            Y[i, codes.index(c)] = 1
        if c in diag.index and isinstance(diag.loc[c].diagnostic_class, str):
            Y[i, len(codes) + supers.index(diag.loc[c].diagnostic_class)] = 1
X = np.stack([np.nan_to_num(wfdb.rdsamp(ROOT + f)[0].T) for f in db.filename_lr]).astype(np.float32)
X = (X - meta["mean_mV"]) / meta["std_mV"]
x = np.empty((len(X), 12, 250), np.float32)
for j in range(4):
    x[:, 3 * j:3 * j + 3] = X[:, 3 * j:3 * j + 3, 250 * j:250 * (j + 1)]
with torch.no_grad():
    P = torch.sigmoid(m(torch.from_numpy(x))).numpy()
names = codes + ["super:" + s for s in supers]
thr = {}
for k, n in enumerate(names):
    if Y[:, k].sum() < 5:
        continue
    pr, rc, th = precision_recall_curve(Y[:, k], P[:, k])
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    thr[n] = round(float(th[int(np.argmax(f1[:-1]))]), 4)
meta["thresholds"] = thr
meta["seuils"] = "F1 maximal, pli 9 de PTB-XL, grille 3x4 (export_ptbxl.py)"
json.dump(meta, open(DEST + "/xresnet1d101_ptbxl.json", "w"), ensure_ascii=False, indent=1)
print(len(thr), "seuils", meta["test_auc"])
