# Évaluation des modèles publiés avec les entrées de l'outil

Question : que perd-on en donnant aux modèles ce qu'un PDF ou un scan permet
(2,5 s par dérivation et une bande rythme de 10 s) plutôt que 10 s
simultanées ? Les ECG numériques annotés sont passés par le code de l'outil
(grille 3x4 simulée, bande rythme DII, battement médian, reconstruction), voir
`source/modeles/eval_models.py`. AUC = aire sous la courbe ROC. Résultats du
19/11/2026, v0.17.0.

## PTB-XL (xresnet1d101 ré-entraîné, pli 10, 2 198 ECG)

| Entrée | AUC macro, 71 énoncés | AUC macro, 5 superclasses |
|---|---|---|
| 10 s simultanées, fenêtres glissantes (protocole du benchmark) | 0,915 | 0,928 |
| Grille 3x4 (une fenêtre de 2,5 s par dérivation, comme l'outil) | 0,900 | 0,919 |
| Valeurs publiées par Strodthoff et al. 2021 | 0,925 | 0,937 |

## Ribeiro 2020 sur CODE-test (716 ECG sur 827 traités, annotations de cardiologues)

| Classe | n | AUC 10 s réelles | AUC battement médian | Sens. / spéc. 10 s | Sens. / spéc. médian |
|---|---|---|---|---|---|
| BAV 1 | 21 | 0,994 | 0,988 | 0,90 / 1,00 | 0,71 / 0,99 |
| BBD | 31 | 0,999 | 0,998 | 1,00 / 0,99 | 1,00 / 0,99 |
| BBG | 27 | 1,000 | 0,999 | 1,00 / 1,00 | 0,96 / 0,99 |
| Bradycardie sinusale | 12 | 0,999 | 0,999 | 0,92 / 1,00 | 0,92 / 0,99 |
| FA | 12 | 0,997 | 0,983 | 0,67 / 1,00 | 0,67 / 0,99 |
| Tachycardie sinusale | 37 | 0,998 | 0,997 | 0,97 / 1,00 | 0,86 / 1,00 |

111 ECG sur 827 n'ont pas donné de battement médian (détection des R sur DII
du code de l'outil) et sont exclus.

## ECGFounder sur PTB-XL (pli 10, 2 188 ECG, correspondance d'étiquettes des auteurs)

AUC, extraits :

| Énoncé | n | 12D, 10 s réelles | 12D, battement médian | 1D sur DI | 1D sur DII |
|---|---|---|---|---|---|
| Fibrillation atriale | 152 | 0,976 | 0,978 | 0,979 | 0,983 |
| Tachycardie sinusale | 81 | 0,984 | 0,985 | 0,992 | 0,992 |
| Bradycardie sinusale | 64 | 0,940 | 0,932 | 0,947 | 0,952 |
| Extrasystoles ventriculaires | 105 | 0,978 | 0,747 | 0,871 | 0,902 |
| BBD | 164 | 0,974 | 0,977 | | |
| BBG | 61 | 0,983 | 0,979 | | |
| Hémibloc antérieur gauche | 161 | 0,969 | 0,966 | | |
| BAV 1 | 77 | 0,909 | 0,892 | 0,914 | 0,834 |
| HVG | 232 | 0,900 | 0,834 | | |
| Infarctus antéroseptal | 231 | 0,954 | 0,937 | | |
| Infarctus inférieur | 265 | 0,858 | 0,842 | | |
| Hypertrophie atriale gauche | 41 | 0,818 | 0,715 | | |

Sur CODE-test, ECGFounder 1D sur DII : FA 0,999, bradycardie 0,982,
tachycardie 0,998.

## Ce qu'on en retient

- Le battement médian replacé sur les R réels garde presque toute
  l'information de conduction (BBD, BBG, hémibloc) et de fréquence. Il perd
  ce qui varie d'un battement à l'autre (extrasystoles : AUC 0,98 vers 0,75)
  et une partie de l'HVG et des oreillettes (0,90 vers 0,83 ; 0,82 vers 0,72).
- Pour le rythme, ECGFounder 1D sur la bande DII fait aussi bien que sur DI,
  la dérivation d'entraînement. Le rythme affiché vient de cette version.
- Le modèle PTB-XL perd 0,015 d'AUC entre 10 s simultanées et la grille 3x4.
- Ces chiffres portent sur des signaux numériques propres ; les ECG imprimés
  puis lus (PDF vectoriel) et surtout scannés ajoutent leur propre erreur, à
  mesurer avec la section « Accord des modèles publiés avec la conclusion de
  l'appareil ».
