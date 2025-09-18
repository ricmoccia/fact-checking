@echo off
REM ==============================================
REM Setup ambiente virtuale per SciFact (Python 3.8)
REM ==============================================

REM 1) Vai nella cartella del progetto
cd /d %~dp0

REM 2) Se esiste un venv precedente, lo elimina
if exist .venv (
    echo Rimuovo vecchio ambiente...
    rmdir /s /q .venv
)

REM 3) Crea nuovo venv con Python 3.8
echo Creo nuovo venv con Python 3.8...
"C:\Python38\python.exe" -m venv --copies .venv

REM 4) Attiva il venv
call .venv\Scripts\activate.bat

REM 5) Aggiorna pip, setuptools, wheel
echo Aggiorno pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

REM 6) Installa i pacchetti dal file lock
echo Installo pacchetti da requirements.lock.txt...
python -m pip install -r requirements.lock.txt

echo.
echo ==============================================
echo Ambiente pronto! Attivato: .venv
echo Usa "call .venv\Scripts\activate.bat" per riattivarlo in futuro.
echo ==============================================
pause
