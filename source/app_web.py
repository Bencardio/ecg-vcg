# Point d'entrée de la version navigateur (stlite / Pyodide, GitHub Pages).
# Installe NeuroKit2 (optionnel) au premier passage, puis exécute app.py tel
# quel. L'await de premier niveau n'est permis que dans ce fichier, exécuté
# par stlite ; app.py reste identique à la version installée.
import builtins

if not getattr(builtins, "_ecg_neurokit_tried", False):
    builtins._ecg_neurokit_tried = True
    try:
        import micropip
        await micropip.install("neurokit2")  # noqa: F704 - top-level await (stlite)
    except Exception as exc:  # l'outil fonctionne sans (méthode tangente)
        print("NeuroKit2 non installé :", exc)

with open("app.py", encoding="utf-8") as _f:
    exec(compile(_f.read(), "app.py", "exec"), {"__name__": "__main__"})
