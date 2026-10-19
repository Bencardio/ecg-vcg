# Analyse ECG-VCG

Outil de recherche (non certifié) : analyse automatisée de PDF ECG 12
dérivations vectoriels (GE MUSE, Cardionics, Dräger) et VCG dérivée, pour les
études CMH et RAC des Cliniques universitaires Saint-Luc.

**Ouvrir l'outil : https://bencardio.github.io/ecg-vcg/**

- Le programme s'exécute entièrement dans le navigateur (Python via Pyodide /
  stlite). Les PDF déposés sont analysés sur l'ordinateur de l'utilisateur et
  ne sont envoyés à aucun serveur.
- Premier chargement : 30 s à 1 min (téléchargement de Python et des
  bibliothèques, ensuite en cache).
- Ce dépôt ne contient que le programme, jamais de données patient.

`index.html` est généré par `source/build_web.py` à partir du code de l'outil
(`ecg_vcg_hcm`) ; ne pas le modifier à la main.
