# Point d'entrée de la version navigateur (stlite / Pyodide, GitHub Pages).
# Installe NeuroKit2 (optionnel) au premier passage, puis exécute app.py tel
# quel. L'await de premier niveau n'est permis que dans ce fichier, exécuté
# par stlite ; app.py reste identique à la version installée.
import builtins
import logging

# Dans le navigateur, le niveau de journalisation par défaut est DEBUG :
# pdfminer y écrit alors une ligne par opération de dessin du PDF (plus de
# 100 000 pour un ECG), ce qui sature et fait planter l'onglet pendant
# l'analyse. On ne garde que les avertissements.
for _name in ("pdfminer", "pdfplumber", "matplotlib", "PIL", "fontTools", "numba"):
    logging.getLogger(_name).setLevel(logging.WARNING)

if not getattr(builtins, "_ecg_neurokit_tried", False):
    builtins._ecg_neurokit_tried = True
    try:
        import micropip
        await micropip.install("neurokit2")  # noqa: F704 - top-level await (stlite)
    except Exception as exc:  # l'outil fonctionne sans (méthode tangente)
        print("NeuroKit2 non installé :", exc)

# Dans le navigateur, Python et l'affichage partagent le même fil : sans
# pause, la barre de progression ne bouge qu'à la fin de l'analyse. On
# remplace donc le repère de app.py par une courte pause asynchrone avant
# chaque fichier, ce qui laisse la page se mettre à jour.
import ast
import asyncio  # noqa: F401 - utilisé par le code injecté

with open("app.py", encoding="utf-8") as _f:
    _src = _f.read()
_src = _src.replace("pass  # <laisser-la-main-au-navigateur>",
                    "await __import__('asyncio').sleep(0.05)  #")
_code = compile(_src, "app.py", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
_ret = eval(_code, {"__name__": "__main__"})
if asyncio.iscoroutine(_ret):
    await _ret  # noqa: F704
