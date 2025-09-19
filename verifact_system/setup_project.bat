@echo off
echo =================================================
echo VeriFact System - Setup Progetto Windows
echo =================================================

REM Crea directory principali
mkdir core 2>nul
mkdir ingestion 2>nul
mkdir processing 2>nul
mkdir retrieval 2>nul
mkdir verification 2>nul
mkdir models 2>nul
mkdir api 2>nul
mkdir dashboard 2>nul
mkdir dashboard\components 2>nul
mkdir dashboard\utils 2>nul
mkdir tests 2>nul
mkdir tests\unit 2>nul
mkdir tests\integration 2>nul
mkdir tests\fixtures 2>nul
mkdir docker 2>nul
mkdir scripts 2>nul
mkdir config 2>nul
mkdir logs 2>nul

echo Directory create con successo!

REM Crea file __init__.py
echo. > core\__init__.py
echo. > ingestion\__init__.py
echo. > processing\__init__.py
echo. > retrieval\__init__.py
echo. > verification\__init__.py
echo. > models\__init__.py
echo. > api\__init__.py
echo. > dashboard\__init__.py
echo. > dashboard\components\__init__.py
echo. > dashboard\utils\__init__.py
echo. > tests\__init__.py

REM Crea file di configurazione
echo. > config\development.yaml
echo. > config\production.yaml
echo. > config\model_configs.yaml
echo. > docker\Dockerfile
echo. > docker\docker-compose.yml
echo. > requirements.txt
echo. > README.md
echo. > .env.example
echo. > .gitignore

REM Crea file core/logger.py
echo # VeriFact System - Logger > core\logger.py
echo import logging >> core\logger.py
echo. >> core\logger.py
echo def get_logger(name): >> core\logger.py
echo     return logging.getLogger(name) >> core\logger.py

echo File di base creati!

echo.
echo =================================================
echo Setup completato con successo!
echo =================================================
echo.
echo Prossimi passi:
echo 1. Copia il contenuto dei file dagli artifacts
echo 2. Installa Python dependencies: pip install -r requirements.txt
echo 3. Configura .env file
echo 4. Esegui: python scripts/run_pipeline.py --help
echo.
pause