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

## Modèles d'interprétation publiés (depuis la v0.17.0)

Option « Interprétation par les modèles publiés » de la barre latérale. Trois
réseaux publiés tournent sur le poste (onnxruntime-web, aucun envoi des ECG),
chacun dans son onglet du détail par patient et dans sa feuille de l'Excel :

| Onglet | Modèle | Poids, licence | Entrée fournie par l'outil |
|---|---|---|---|
| IA : Ribeiro | Ribeiro et al., Nat Commun 2020 | Zenodo 3765717, CC-BY 4.0 | battement médian de chaque dérivation placé sur les pics R réels de la bande rythme, 10 s |
| IA : ECGFounder | Li et al., NEJM AI 2025 (12 et 1 dérivation) | Hugging Face PKUDigitalHealth/ECGFounder, MIT | 12D : même reconstruction ; 1D : bande rythme de 10 s seule (rythme) |
| IA : PTB-XL | xresnet1d101, benchmark de Strodthoff et al., IEEE JBHI 2021 | ré-entraîné pour l'outil sur PTB-XL 1.0.3 (CC-BY 4.0) | les 2,5 s de chaque dérivation telles qu'imprimées |

Pourquoi ces entrées : un PDF ou un scan ne donne que 2,5 s par dérivation
(sauf la bande rythme). Les modèles 12 dérivations sur 10 s reçoivent donc
une reconstruction ; leurs énoncés de rythme ne sont pas interprétables (sauf
la fréquence), le rythme vient d'ECGFounder 1 dérivation sur la vraie bande
rythme. Le modèle PTB-XL travaille nativement sur 2,5 s et a été entraîné pour
moitié sur des fenêtres disposées comme une grille 3x4.

Les poids convertis sont dans `modeles/` (ECGFounder en float16, coupé en
morceaux de moins de 50 Mo ; écart de probabilité avec l'original < 0,001).
Premier usage : environ 140 Mo téléchargés, puis gardés en cache par le
navigateur. Scripts de conversion, d'entraînement et d'évaluation :
`source/modeles/`, résultats de l'évaluation : `modeles/EVALUATION.md`.

La section « Accord des modèles publiés avec la conclusion de l'appareil »
lit un tableau (colonne `fichier`, colonne `conclusion` recopiée de
l'appareil) et calcule sensibilité, spécificité et kappa par catégorie, sur
le poste ; le texte des conclusions n'entre dans aucun export.

