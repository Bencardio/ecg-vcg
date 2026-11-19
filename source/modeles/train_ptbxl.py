"""Entraîne xresnet1d101 (Strodthoff et al. 2021, benchmark PTB-XL) sur PTB-XL 100 Hz.

Protocole du benchmark : plis 1-8 entraînement, 9 validation, 10 test ;
fenêtres aléatoires de 2,5 s, AdamW + one-cycle (lr 1e-2, wd 1e-2, 50 époques,
lots de 128), test par fenêtres glissantes de 2,5 s (pas 1,25 s) agrégées par
le maximum. Sorties : 71 libellés SCP (« all ») + 5 superclasses diagnostiques.

Ajout : la moitié des fenêtres d'entraînement imite la grille 3x4 (chaque
groupe de 3 dérivations pris à un instant différent), et une variation
d'amplitude de ±15 %, pour les ECG imprimés et scannés.
"""
import ast
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from sklearn.metrics import roc_auc_score

from xresnet1d import xresnet1d101

ROOT = sys.argv[1]
OUT = sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.default_rng(0)
W = 250  # 2,5 s à 100 Hz

db = pd.read_csv(ROOT + "ptbxl_database.csv", index_col="ecg_id")
db.scp_codes = db.scp_codes.apply(ast.literal_eval)
stm = pd.read_csv(ROOT + "scp_statements.csv", index_col=0)
diag = stm[stm.diagnostic == 1.0]
all_codes = sorted(set(c for d in db.scp_codes for c in d))
supers = ["CD", "HYP", "MI", "NORM", "STTC"]
Y = np.zeros((len(db), len(all_codes) + 5), np.float32)
for i, d in enumerate(db.scp_codes):
    for c in d:
        Y[i, all_codes.index(c)] = 1
        if c in diag.index and isinstance(diag.loc[c].diagnostic_class, str):
            Y[i, len(all_codes) + supers.index(diag.loc[c].diagnostic_class)] = 1
print(len(all_codes), "codes", flush=True)

t0 = time.time()
X = np.stack([wfdb.rdsamp(ROOT + f)[0] for f in db.filename_lr]).astype(np.float32)
X = np.nan_to_num(X).transpose(0, 2, 1)  # N, 12, 1000 (mV)
print("chargé", X.shape, "%.0f s" % (time.time() - t0), flush=True)
fold = db.strat_fold.values
tr, va, te = fold <= 8, fold == 9, fold == 10
mean, std = float(X[tr].mean()), float(X[tr].std())
X = (X - mean) / std

model = xresnet1d101(num_classes=Y.shape[1], input_channels=12, kernel_size=5,
                     ps_head=0.5, lin_ftrs_head=[128])
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
Xtr, Ytr = X[tr], Y[tr]
steps = EPOCHS * int(np.ceil(len(Xtr) / 128))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=steps)
GROUPS = [(0, 3), (3, 6), (6, 9), (9, 12)]


def crops(xb):
    n = len(xb)
    out = np.empty((n, 12, W), np.float32)
    for k in range(n):
        if rng.random() < 0.5:
            s = rng.integers(0, 1000 - W + 1)
            out[k] = xb[k, :, s:s + W]
        else:
            for a, b in GROUPS:
                s = rng.integers(0, 1000 - W + 1)
                out[k, a:b] = xb[k, a:b, s:s + W]
    out *= rng.uniform(0.85, 1.15, (n, 1, 1)).astype(np.float32)
    return out


def predict(Xs, layout):
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(Xs), 256):
            xb = Xs[i:i + 256]
            if layout == "glissant":
                p = [torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(xb[:, :, s:s + W])))).numpy()
                     for s in range(0, 1000 - W + 1, W // 2)]
                res.append(np.max(p, axis=0))
            else:  # grille 3x4 : colonnes 0-2,5 / 2,5-5 / 5-7,5 / 7,5-10 s
                x = np.empty((len(xb), 12, W), np.float32)
                for j, (a, b) in enumerate(GROUPS):
                    x[:, a:b] = xb[:, a:b, j * W:(j + 1) * W]
                res.append(torch.sigmoid(model(torch.from_numpy(x))).numpy())
    return np.concatenate(res)


def aucs(y, p):
    n = len(all_codes)
    out = {}
    for name, sl in (("all", slice(0, n)), ("super", slice(n, n + 5))):
        yy, pp = y[:, sl], p[:, sl]
        keep = (yy.sum(0) > 0) & (yy.sum(0) < len(yy))
        out[name] = float(roc_auc_score(yy[:, keep], pp[:, keep], average="macro"))
    return out


for ep in range(EPOCHS):
    model.train()
    perm = rng.permutation(len(Xtr))
    tl = 0.0
    t0 = time.time()
    for i in range(0, len(perm), 128):
        idx = perm[i:i + 128]
        xb = torch.from_numpy(crops(Xtr[idx]))
        yb = torch.from_numpy(Ytr[idx])
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tl += loss.item() * len(idx)
    msg = "époque %d perte %.4f (%.0f s)" % (ep + 1, tl / len(perm), time.time() - t0)
    if (ep + 1) % 10 == 0 or ep + 1 == EPOCHS:
        msg += " val %s" % aucs(Y[va], predict(X[va], "glissant"))
    print(msg, flush=True)

result = {"test_glissant": aucs(Y[te], predict(X[te], "glissant")),
          "test_grille_3x4": aucs(Y[te], predict(X[te], "3x4"))}
print(result, flush=True)
torch.save(model.state_dict(), OUT + "/xresnet1d101_ptbxl.pt")
desc = {s: stm.loc[s].description if s in stm.index else s for s in all_codes}
json.dump({"mean_mV": mean, "std_mV": std, "fs": 100, "window": W,
           "codes": all_codes, "supers": supers, "descriptions": desc,
           "test_auc": result}, open(OUT + "/xresnet1d101_ptbxl.json", "w"),
          ensure_ascii=False, indent=1)
