"""Construit web/index.html : l'appli Streamlit complète, exécutée dans le
navigateur par stlite (Pyodide). Les PDF ne quittent jamais l'ordinateur.

    python web/build_web.py  ->  web/site/index.html  (à publier sur GitHub Pages)

pdfplumber est embarqué en source (sa dépendance pypdfium2, binaire, n'existe
pas pour le navigateur ; elle ne sert qu'au rendu d'images, jamais utilisé ici).
"""
import glob
import json
import os
import sys

STLITE = "1.9.2"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REQUIREMENTS = ["numpy", "scipy", "pandas", "matplotlib", "openpyxl",
                "pdfminer.six", "pillow"]


def collect() -> dict:
    files = {"app.py": os.path.join(ROOT, "app.py"),
             "app_web.py": os.path.join(HERE, "app_web.py")}
    for p in glob.glob(os.path.join(ROOT, "ecg_hcm", "*.py")):
        files["ecg_hcm/" + os.path.basename(p)] = p
    import pdfplumber
    base = os.path.dirname(pdfplumber.__file__)
    for p in glob.glob(os.path.join(base, "**", "*.py"), recursive=True):
        files["pdfplumber/" + os.path.relpath(p, base).replace(os.sep, "/")] = p
    out = {}
    for name, path in sorted(files.items()):
        with open(path, encoding="utf-8") as f:
            out[name] = f.read()
    return out


def build(out_dir: str) -> str:
    files = collect()
    sys.path.insert(0, ROOT)
    from ecg_hcm import __version__
    payload = json.dumps(files).replace("</", "<\\/")
    html = TEMPLATE.replace("%VERSION%", __version__).replace("%STLITE%", STLITE) \
        .replace("%REQS%", json.dumps(REQUIREMENTS)).replace("%FILES%", payload)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analyse ECG-VCG</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@%STLITE%/build/stlite.css">
<style>
  #chargement { font: 15px/1.5 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 16px; color: #23303a; }
</style>
</head>
<body>
<div id="chargement">
  <p><strong>Analyse ECG-VCG v%VERSION%</strong> se charge dans votre navigateur
  (30 s à 1 min la première fois, plus rapide ensuite).</p>
  <p>Les PDF que vous déposez sont analysés sur cet ordinateur ; ils ne sont envoyés à aucun serveur.</p>
</div>
<div id="root"></div>
<script id="fichiers" type="application/json">%FILES%</script>
<script>
// Firefox : stlite clique sur un lien de téléchargement détaché du document
// puis révoque aussitôt l'URL blob ; Firefox annule alors le téléchargement
// (bouton sans effet). On attache le lien au document et on retarde la
// révocation.
(function () {
  const revoke = URL.revokeObjectURL.bind(URL);
  URL.revokeObjectURL = function (u) { setTimeout(function () { revoke(u); }, 60000); };
  const click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (this.hasAttribute("download") && !this.isConnected) {
      this.style.display = "none";
      document.body.appendChild(this);
      click.call(this);
      const a = this; setTimeout(function () { a.remove(); }, 1000);
      return;
    }
    return click.call(this);
  };
})();
</script>
<script type="module">
import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@%STLITE%/build/stlite.js";
const files = JSON.parse(document.getElementById("fichiers").textContent);
document.getElementById("chargement").remove();
mount({ entrypoint: "app_web.py", requirements: %REQS%, files,
        streamlitConfig: { "browser.gatherUsageStats": false } },
      document.getElementById("root"));
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(build(os.path.join(HERE, "site")))
