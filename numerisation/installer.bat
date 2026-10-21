@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo === Installation de l'outil de numerisation des ECG scannes ===
echo (Open-ECG-Digitizer + script pour l'application ECG-VCG)
echo.

set "PY="
for %%V in (3.12 3.13) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V" )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.12 ou 3.13 introuvable.
  echo Installez-le depuis https://www.python.org/downloads/ en cochant "Add python.exe to PATH",
  echo puis relancez installer.bat.
  pause
  exit /b 1
)
echo Python utilise : %PY%

if not exist "Open-ECG-Digitizer-main\src" (
  echo Telechargement d'Open-ECG-Digitizer...
  curl -L -o oed.zip https://github.com/Ahus-AIM/Open-ECG-Digitizer/archive/refs/heads/main.zip || goto :erreur
  tar -xf oed.zip || goto :erreur
  del oed.zip
)
cd "Open-ECG-Digitizer-main"

echo Telechargement des poids du modele (environ 110 Mo)...
for %%W in (unet_weights_07072025.pt lead_name_unet_weights_07072025.pt) do (
  curl -L -o "weights\%%W" "https://media.githubusercontent.com/media/Ahus-AIM/Open-ECG-Digitizer/main/weights/%%W" || goto :erreur
)
copy /y "..\numeriser_ecg.py" . >nul

if not exist "venv\Scripts\python.exe" (
  echo Creation de l'environnement Python...
  %PY% -m venv venv || goto :erreur
)
echo Installation de PyTorch (version processeur, environ 200 Mo)...
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :erreur
echo Installation des autres bibliotheques...
venv\Scripts\python -m pip install numpy pandas matplotlib scikit-image scipy yacs pyyaml tqdm scikit-learn networkx torch-tps pymupdf pillow || goto :erreur

venv\Scripts\python -c "import numeriser_ecg" || goto :erreur
echo.
echo Installation terminee. Pour numeriser : glissez un dossier de scans sur numeriser.bat
pause
exit /b 0

:erreur
echo.
echo ERREUR pendant l'installation : copiez le message ci-dessus a Claude.
pause
exit /b 1
