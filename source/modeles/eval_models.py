"""Mesure, sur des ECG numériques annotés, la perte liée aux entrées que
l'outil peut fournir à partir d'un PDF ou d'un scan :

- PTB-XL plis 9 (seuils) et 10 (test), 500 Hz : ECGFounder 12D (10 s brutes
  vs battement médian placé sur les R de DII), ECGFounder 1D (DI vs DII).
- CODE-test (Ribeiro 2020, 827 ECG, 400 Hz, annotations de cardiologues) :
  Ribeiro (10 s brutes vs battement médian), ECGFounder 12D, ECGFounder 1D.

La simulation passe par le vrai code de l'outil : ECGRecording en grille
3x4 (chaque colonne = tranche de 2,5 s successive), bande rythme DII 10 s,
beat_detection.analyze puis ai_models.prepare.

    python eval_models.py <dossier_onnx> <sortie.json> [n_max]
"""
import ast
import glob
import json
import sys

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
import wfdb
from scipy.signal import resample
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

sys.path.insert(0, "/home/user/work/src")
from ecg_hcm import ai_models as A  # noqa: E402
from ecg_hcm import beat_detection as BD  # noqa: E402
from ecg_hcm.signal_builder import ECGRecording, SignalTrace  # noqa: E402

ONNX, OUT = sys.argv[1], sys.argv[2]
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
LEADS = A.LEADS
TASKS = [x[0] for x in A.L.ECGFOUNDER_LABELS]
sess = {k: ort.InferenceSession(f"{ONNX}/{f}", providers=["CPUExecutionProvider"])
        for k, f in (("ribeiro", "ribeiro2020_fp16.onnx"), ("ecgfounder12", "ecgfounder_12d_fp16.onnx"),
                     ("ecgfounder1", "ecgfounder_1d_fp16.onnx"))}


def run(k, x):
    s = sess[k]
    return s.run(None, {s.get_inputs()[0].name: x.astype(np.float32)})[0]


class Res:
    pass


def simulate(sig12, fs):
    """sig12 : 12 x N (mV, 10 s). Retourne un objet comme PatientECGResult."""
    x = resample(sig12, int(round(sig12.shape[1] * 500 / fs)), axis=1)
    t = np.arange(x.shape[1]) / 500.0
    rec = ECGRecording()
    w = t[-1] / 4 + 1e-9  # 2,5 s pour 10 s ; tracés plus courts : quart de la durée
    for j, ld in enumerate(LEADS):
        col = j // 3
        sel = (t >= w * col) & (t < w * (col + 1))
        rec.leads[ld] = SignalTrace(ld, t[sel], x[j, sel], 500.0, 10.0, "header")
    rec.rhythm = SignalTrace("II", t, x[1], 500.0, 10.0, "header")
    rec.rhythm_lead_name = "II"
    r = Res()
    r.recording = rec
    try:
        r.beat = BD.analyze(rec)
    except Exception:
        r.beat = None
    return r, x


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def ecgf_preds(signals, fs):
    """Pour chaque ECG : probabilités ECGFounder 12D brut / reconstruit,
    1D DI / DII, Ribeiro brut / reconstruit."""
    P = {k: [] for k in ("f12_brut", "f12_median", "f1_DI", "f1_DII", "rib_brut", "rib_median")}
    ok = []
    for sig in signals:
        res, x500 = simulate(sig, fs)
        inp = A.prepare(res)
        if inp["ecgfounder12"][0] is None:
            ok.append(False)
            for k in P:
                P[k].append(np.full(150 if k.startswith("f") else 6, np.nan))
            continue
        ok.append(True)
        sg = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
        x5000 = A._fit_length(x500, 5000)
        P["f12_brut"].append(sg(run("ecgfounder12", zs(x5000)[None]))[0])
        P["f12_median"].append(sg(run("ecgfounder12", inp["ecgfounder12"][0]))[0])
        P["f1_DI"].append(sg(run("ecgfounder1", zs(x5000[:1])[None]))[0])
        P["f1_DII"].append(sg(run("ecgfounder1", inp["ecgfounder1"][0]))[0])
        x400 = A._fit_length(resample(x500, 4000, axis=1), 4096).T[None]
        P["rib_brut"].append(run("ribeiro", x400)[0])
        P["rib_median"].append(run("ribeiro", inp["ribeiro"][0])[0])
    return {k: np.array(v) for k, v in P.items()}, np.array(ok)


def auc(y, p):
    m = np.isfinite(p)
    if y[m].sum() < 5 or y[m].sum() == m.sum():
        return None
    return round(float(roc_auc_score(y[m], p[m])), 3)


def best_threshold(y, p):
    if y.sum() < 10:
        return None
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    i = int(np.nanargmax(f1[:-1]))
    return float(th[i])


report = {}

# ---------------- PTB-XL ----------------
ONLY_CODE = "--code" in sys.argv
if ONLY_CODE:
    report = json.load(open(OUT))["report"]
root = glob.glob("/home/user/work/dl/ptbxl/*/")[0]
lab = pd.read_csv("/home/user/work/ext/ECGFounder/csv/ptbxl_label.csv")
db = pd.read_csv(root + "ptbxl_database.csv")
lab["strat_fold"] = db.strat_fold.values
lab = lab.dropna(subset=["label"])
preds = {}
for fold in (() if ONLY_CODE else (9, 10)):
    d = lab[lab.strat_fold == fold].iloc[:NMAX]
    sigs = [np.nan_to_num(wfdb.rdsamp(root + f)[0].T) for f in d.filename_hr]
    P, ok = ecgf_preds(sigs, 500)
    Y = np.array([json.loads(v) for v in d.label])
    preds[fold] = (P, ok, Y)
    print("PTB-XL pli", fold, "traités", int(ok.sum()), "/", len(ok), flush=True)

if not ONLY_CODE:
    # Seuils ECGFounder (F1 maximal, pli 9) sur les entrées de l'outil.
    P9, ok9, Y9 = preds[9] if not ONLY_CODE else (None, None, None)
    thr = {"ecgfounder12": {}, "ecgfounder1": {}}
    for i, t in enumerate(TASKS):
        a = best_threshold(Y9[ok9, i], P9["f12_median"][ok9, i])
        b = best_threshold(Y9[ok9, i], P9["f1_DII"][ok9, i])
        if a is not None:
            thr["ecgfounder12"][t] = round(a, 4)
        if b is not None:
            thr["ecgfounder1"][t] = round(b, 4)

    P, ok, Y = preds[10]
    rows = []
    for i, t in enumerate(TASKS):
        n = int(Y[ok, i].sum())
        if n < 10:
            continue
        row = {"énoncé": t, "n_pos": n}
        for k in ("f12_brut", "f12_median", "f1_DI", "f1_DII"):
            row[k] = auc(Y[ok, i], P[k][ok, i])
        for k, m in (("f12_median", "ecgfounder12"), ("f1_DII", "ecgfounder1")):
            th = thr[m].get(t)
            if th is not None:
                yp = P[k][ok, i] >= th
                yt = Y[ok, i] == 1
                row["sens_" + k] = round(float((yp & yt).sum() / max(yt.sum(), 1)), 3)
                row["spec_" + k] = round(float((~yp & ~yt).sum() / max((~yt).sum(), 1)), 3)
        rows.append(row)
    report["ptbxl_pli10_ecgfounder"] = rows
    report["ptbxl_n"] = int(ok.sum())
else:
    thr = json.load(open(OUT))["thresholds"]

# ---------------- CODE-test ----------------
base = "/home/user/work/dl/codetest/data/"
X = h5py.File(base + "ecg_tracings.hdf5")["tracings"]
ann = {f: pd.read_csv(base + "annotations/%s.csv" % f) for f in ("gold_standard",)}
gs = ann["gold_standard"].values.astype(int)
n = min(len(gs), NMAX)
sigs = []
for k in range(n):
    s = np.asarray(X[k], dtype=float).T  # 12 x 4096, 400 Hz, zéros aux bords
    nz = np.where(np.abs(s).sum(0) > 0)[0]
    s = s[:, nz[0]:nz[-1] + 1] if len(nz) else s
    sigs.append(s)  # durée réelle (7 à 10 s), sans zéros ajoutés
P, ok = ecgf_preds(sigs, 400)
gs = gs[:n]
cmap = {"1dAVb": "WITH 1ST DEGREE AV BLOCK", "RBBB": "RIGHT BUNDLE BRANCH BLOCK",
        "LBBB": "LEFT BUNDLE BRANCH BLOCK", "SB": "SINUS BRADYCARDIA",
        "AF": "ATRIAL FIBRILLATION", "ST": "SINUS TACHYCARDIA"}
rows = []
for j, (code, fr, th) in enumerate(A.L.RIBEIRO_LABELS):
    y = gs[ok, j]
    row = {"classe": code, "n_pos": int(y.sum()),
           "rib_brut": auc(y, P["rib_brut"][ok, j]), "rib_median": auc(y, P["rib_median"][ok, j])}
    for k in ("rib_brut", "rib_median"):
        yp = P[k][ok, j] >= th
        row["sens_" + k] = round(float((yp & (y == 1)).sum() / max(y.sum(), 1)), 3)
        row["spec_" + k] = round(float((~yp & (y == 0)).sum() / max((y == 0).sum(), 1)), 3)
    i = TASKS.index(cmap[code])
    for k in ("f12_brut", "f12_median", "f1_DI", "f1_DII"):
        row[k] = auc(y, P[k][ok, i])
    rows.append(row)
report["code_test"] = rows
report["code_n"] = int(ok.sum())

json.dump({"report": report, "thresholds": thr}, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
