@echo off
chcp 65001 >nul
setlocal
set "KIT=%~dp0"
set "SCANS=%~1"
if "%SCANS%"=="" set /p SCANS=Dossier contenant les ECG scannes (PDF, PNG, JPG, TIFF) : 
set "SCANS=%SCANS:"=%"
if not exist "%SCANS%" (
  echo Dossier introuvable : %SCANS%
  pause
  exit /b 1
)
set "SORTIE=%SCANS%\numerises"
cd /d "%KIT%Open-ECG-Digitizer-main" || (echo Lancez d'abord installer.bat & pause & exit /b 1)
venv\Scripts\python numeriser_ecg.py "%SCANS%" "%SORTIE%" %2 %3 %4 %5 %6 %7
echo.
echo Resultats dans : %SORTIE%
echo Deposez les .csv et les .json dans l'application ECG-VCG (https://bencardio.github.io/ecg-vcg/).
explorer "%SORTIE%"
pause
