# Poids des modèles publiés

Fichiers convertis en ONNX pour l'outil (scripts dans `source/modeles/`).
Aucun ECG ni donnée patient.

- `ribeiro2020_fp16.onnx` : Ribeiro AH, Ribeiro MH, Paixão GMM et al.
  Automatic diagnosis of the 12-lead ECG using a deep neural network.
  Nat Commun 2020;11:1760. Poids d'origine : Zenodo, doi 10.5281/zenodo.3765717,
  licence CC-BY 4.0. Modification : conversion Keras vers ONNX, poids en
  float16. Code d'origine : github.com/antonior92/automatic-ecg-diagnosis (MIT).
- `ecgfounder_12d_fp16.onnx.part0/1`, `ecgfounder_1d_fp16.onnx.part0/1` :
  Li J, Aguirre A, Moura J et al. An electrocardiogram foundation model built
  on over 10 million recordings with external evaluation across multiple
  domains (arXiv 2410.04133). Poids : huggingface.co/PKUDigitalHealth/ECGFounder.
  Licence MIT, Copyright (c) 2025 PKUDigitalHealth : « Permission is hereby
  granted, free of charge, to any person obtaining a copy of this software
  and associated documentation files (the "Software"), to deal in the
  Software without restriction... The above copyright notice and this
  permission notice shall be included in all copies or substantial portions
  of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
  KIND. » Modification : conversion PyTorch vers ONNX, poids en float16,
  fichier coupé en deux.
- `xresnet1d101_ptbxl.onnx` et `.json` : architecture xresnet1d101 du
  benchmark de Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep learning
  for ECG analysis: benchmarks and insights from PTB-XL. IEEE J Biomed Health
  Inform 2021;25:1519-28 (code github.com/helme/ecg_ptbxl_benchmarking,
  GPL-3.0). Entraîné pour l'outil sur PTB-XL 1.0.3 (Wagner P et al., Sci Data
  2020;7:154 ; PhysioNet, CC-BY 4.0) avec `source/modeles/train_ptbxl.py`.
- `ecgfounder_seuils.json` : seuils par énoncé choisis pour l'outil (F1
  maximal sur le pli 9 de PTB-XL, entrées reconstruites comme dans l'outil).
