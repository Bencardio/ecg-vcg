"""
numeriser_ecg.py
================

Numérise des ECG papier scannés (PDF scanné, PNG, JPG, TIFF) avec
Open-ECG-Digitizer (Stenhede et al., npj Digit Med 2026 ;
github.com/Ahus-AIM/Open-ECG-Digitizer, licence CC BY-SA 4.0) et écrit, pour
chaque ECG, les fichiers lus par l'application ECG-VCG
(https://bencardio.github.io/ecg-vcg/) :

- <nom>.csv  : t_s, I ... V6 (NaN hors de la colonne où la dérivation est
  imprimée), rhythm_<dérivation> (bande rythme de 10 s), en µV ;
- <nom>.json : mise en page, vitesse, gain, fréquence, contrôle qualité ;
- <nom>_controle.png : image redressée et tracés extraits, à regarder.

Pourquoi pas la sortie « canonique » d'Open-ECG-Digitizer : avec une bande
rythme, elle remplace la dérivation qui lui ressemble le plus (avec une
préférence pour II) par la bande entière, ce qui peut écraser II par la
bande V1. Ici la mise en page est imposée (par défaut 3x4 + bande rythme V1,
format GE de Saint-Luc) et chaque ligne est rangée à sa place.

Usage (dans le dossier Open-ECG-Digitizer, environnement activé) :
    python numeriser_ecg.py DOSSIER_SCANS DOSSIER_SORTIE
    options : --rythme V1 | II | V5 | aucun   (défaut V1)
              --vitesse 25  --gain 10          (mm/s, mm/mV imprimés)
              --dpi 300                        (rendu des PDF scannés)

Rien n'est envoyé sur Internet : tout tourne sur ce PC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# src/utils.py importe ray.tune (entraînement uniquement) : module factice
# pour ne pas installer ray, lourd et inutile ici.
if "ray" not in sys.modules:
    try:
        import ray.tune  # noqa: F401
    except ImportError:
        ray = types.ModuleType("ray")
        tune = types.ModuleType("ray.tune")
        tune.Stopper = type("Stopper", (), {})
        ray.tune = tune
        sys.modules["ray"] = ray
        sys.modules["ray.tune"] = tune

import torch  # noqa: E402
import yaml  # noqa: E402

from src.config.default import get_cfg  # noqa: E402
from src.model.lead_identifier import LeadIdentifier  # noqa: E402
from src.utils import import_class_from_path  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GRID_3x4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# --- largeur (en pixels) de la zone de tracé gardée par Open-ECG-Digitizer,
# pour en déduire la durée : LeadIdentifier.normalize la recadre sans la
# renvoyer. On refait le même calcul juste avant.
_LAST = {}
_orig_normalize = LeadIdentifier.normalize


def _normalize_spy(self, lines, avg_pixel_per_mm, mv_per_mm):
    ok = torch.sum(~torch.isnan(lines), dim=0).numpy()
    first = int(np.argmax(ok >= self.required_valid_samples))
    last = lines.shape[1] - int(np.argmax(ok[::-1] >= self.required_valid_samples)) - 1
    _LAST["width_px"] = (last - first + 1) if first <= last else lines.shape[1]
    _LAST["px_per_mm"] = float(avg_pixel_per_mm)
    return _orig_normalize(self, lines, avg_pixel_per_mm, mv_per_mm)


LeadIdentifier.normalize = _normalize_spy


def build_model(rhythm: str | None, device: str):
    layouts = {"3x4": {"description": "3x4", "layout": {"rows": 3, "cols": 4},
                       "leads": GRID_3x4, "rhythm_leads": [], "total_rows": 3}}
    if rhythm:
        layouts = {"3x4_r1": {"description": "3x4 + 1 bande rythme", "layout": {"rows": 3, "cols": 4},
                              "leads": GRID_3x4, "rhythm_leads": ["Any"], "total_rows": 4}}
    layout_path = os.path.join(HERE, "src", "config", "lead_layouts_ecgvcg.yml")
    with open(layout_path, "w") as f:
        yaml.safe_dump(layouts, f)
    cfg = get_cfg(os.path.join(HERE, "src", "config", "inference_wrapper_ahus_testset.yml"))
    k = cfg.MODEL.KWARGS
    k.device = device
    k.config.LAYOUT_IDENTIFIER.KWARGS.device = device
    k.config.LAYOUT_IDENTIFIER.config_path = layout_path
    k.config.SEGMENTATION_MODEL.weight_path = os.path.join(HERE, "weights", "unet_weights_07072025.pt")
    k.config.LAYOUT_IDENTIFIER.unet_weight_path = os.path.join(HERE, "weights", "lead_name_unet_weights_07072025.pt")
    k.config.LAYOUT_IDENTIFIER.unet_config_path = os.path.join(HERE, "src", "config", "lead_name_unet.yml")
    cls = import_class_from_path(cfg.MODEL.class_path)
    return cls(**k)


def load_images(path: str, dpi: int):
    """[(suffixe, tenseur 1x3xHxW uint8)] ; une image par page pour un PDF."""
    from PIL import Image
    if path.lower().endswith(".pdf"):
        import pymupdf
        out = []
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                out.append(("" if len(doc) == 1 else "_p%d" % (i + 1), arr.copy()))
    else:
        out = [("", np.asarray(Image.open(path).convert("RGB")))]
    return [(s, torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)) for s, a in out]


def to_table(res: dict, rhythm: str | None, speed: float, gain: float):
    lines = res["signal"]["lines"]
    if lines is None:
        raise ValueError("aucun tracé extrait")
    lines = lines.squeeze().cpu().numpy().astype(float) * (10.0 / gain)  # extraits à 10 mm/mV
    if lines.ndim == 1:
        lines = lines[None, :]
    n = lines.shape[1]
    duration = _LAST["width_px"] / _LAST["px_per_mm"] / speed
    t = np.arange(n) * duration / n
    need = 4 if rhythm else 3
    if lines.shape[0] < need:
        raise ValueError("%d ligne(s) de tracé détectée(s), %d attendues (3x4%s)"
                         % (lines.shape[0], need, " + bande rythme" if rhythm else ""))
    if lines.shape[0] > need:
        # lignes en trop (texte, bord de page) : on garde les plus remplies, dans l'ordre
        fill = (~np.isnan(lines)).sum(axis=1)
        keep = np.sort(np.argsort(fill)[::-1][:need])
        lines = lines[keep]
    cols = {"t_s": t}
    chunk = n // 4
    for r, row in enumerate(GRID_3x4):
        for c, lead in enumerate(row):
            v = np.full(n, np.nan)
            end = (c + 1) * chunk if c < 3 else n
            v[c * chunk:end] = lines[r, c * chunk:end]
            cols[lead] = v
    if rhythm:
        cols["rhythm_" + rhythm] = lines[3]
    return cols, duration


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entree")
    ap.add_argument("sortie")
    ap.add_argument("--rythme", default="V1")
    ap.add_argument("--vitesse", type=float, default=25.0)
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rhythm = None if a.rythme.lower() in ("aucun", "none", "") else a.rythme
    os.makedirs(a.sortie, exist_ok=True)

    files = []
    out_abs = os.path.abspath(a.sortie)
    for root, dirs, names in os.walk(a.entree):
        # ne pas relire les sorties d'un passage précédent (dossier « numerises »)
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != out_abs]
        files += [os.path.join(root, f) for f in sorted(names)
                  if f.lower().endswith(IMAGE_EXT + (".pdf",)) and not f.endswith("_controle.png")]
    if not files:
        sys.exit("Aucun scan (PDF, PNG, JPG, TIFF) dans %s" % a.entree)
    print("Chargement du modèle (%s)..." % a.device)
    model = build_model(rhythm, a.device)

    import pandas as pd
    bilan = []
    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        for suffix, img in load_images(path, a.dpi):
            name = stem + suffix
            print("[%d/%d] %s" % (i, len(files), name), flush=True)
            try:
                with torch.no_grad():
                    res = model(img, layout_should_include_substring=None)
                cols, duration = to_table(res, rhythm, a.vitesse, a.gain)
            except Exception as exc:  # noqa: BLE001
                print("   ÉCHEC : %s" % exc)
                bilan.append({"fichier": name, "statut": "échec", "detail": str(exc)})
                continue
            df = pd.DataFrame(cols)
            df.to_csv(os.path.join(a.sortie, name + ".csv"), index=False, float_format="%.4f")
            nan_pct = {ld: round(100 * float(np.isnan(v).mean()), 1)
                       for ld, v in cols.items() if ld != "t_s"}
            # part de NaN DANS la fenêtre où la dérivation est imprimée
            win = {}
            for ld in LEADS:
                v = cols[ld]
                c = [r.index(ld) for r in GRID_3x4 if ld in r][0]
                q = len(v) // 4
                seg = v[c * q:(c + 1) * q if c < 3 else len(v)]
                win[ld] = round(100 * float(np.isnan(seg).mean()), 1)
            meta = {
                "source": "scan", "image": os.path.basename(path) + suffix,
                "layout": "3x4+1 (%s)" % rhythm if rhythm else "3x4",
                "paper_speed_mm_s": a.vitesse, "gain_limb_mm_mV": a.gain, "gain_chest_mm_mV": a.gain,
                "unit": "uV", "duration_s": round(duration, 3),
                "fs_hz": round(len(df) / duration, 1) if duration > 0 else None,
                "pixels_par_mm": round(_LAST["px_per_mm"], 2),
                "cout_appariement_mise_en_page": float(res["signal"].get("layout_matching_cost", float("nan"))),
                "nan_pct_dans_la_fenetre": win,
                "nan_pct_bande_rythme": nan_pct.get("rhythm_" + rhythm) if rhythm else None,
                "outil": "Open-ECG-Digitizer (Stenhede et al., npj Digit Med 2026)",
            }
            with open(os.path.join(a.sortie, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            _control_png(res, cols, os.path.join(a.sortie, name + "_controle.png"))
            alerte = []
            if not 9.0 <= duration <= 11.0:
                alerte.append("durée %.1f s (10 s attendues : vitesse ou recadrage ?)" % duration)
            mauvais = [ld for ld, p in win.items() if p > 20]
            if mauvais:
                alerte.append("plus de 20 %% illisible : %s" % ", ".join(mauvais))
            if rhythm and (meta["nan_pct_bande_rythme"] or 0) > 20:
                alerte.append("bande rythme illisible à plus de 20 %")
            print("   ok, %.1f s%s" % (duration, ("  ATTENTION : " + " ; ".join(alerte)) if alerte else ""))
            bilan.append({"fichier": name, "statut": "à vérifier" if alerte else "ok",
                          "detail": " ; ".join(alerte), "duree_s": round(duration, 2)})
    pd.DataFrame(bilan).to_csv(os.path.join(a.sortie, "bilan_numerisation.csv"), index=False,
                               encoding="utf-8-sig", sep=";")
    print("\nTerminé. Déposez les .csv ET les .json de %s dans l'application ECG-VCG." % a.sortie)


def _control_png(res, cols, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [1, 1.2]})
    axs[0].imshow(res["aligned"]["image"].squeeze().permute(1, 2, 0).numpy() * 0.999)
    axs[0].set_title("Image redressée")
    axs[0].axis("off")
    t = cols["t_s"]
    names = [k for k in cols if k != "t_s"]
    for i, k in enumerate(names):
        axs[1].plot(t, cols[k] / 1000 - 2.5 * i, lw=0.6, color="k")
        axs[1].text(-0.4, -2.5 * i, k, ha="right", va="center", fontsize=9)
    axs[1].set_xlim(-0.6, t[-1] + 0.1)
    axs[1].set_yticks([])
    axs[1].set_xlabel("s")
    axs[1].set_title("Tracés extraits (mV, décalés)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
