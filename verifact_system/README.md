 # VeriFact System

![VeriFact Logo](logo.png)

Un sistema avanzato di fact-checking basato su Retrieval-Augmented Generation (RAG) per la verifica automatica di informazioni online.

## 🎯 Panoramica

VeriFact è un sistema intelligente progettato per affrontare la crescente necessità di verificare rapidamente e in modo affidabile le informazioni online. Utilizzando tecniche avanzate di Natural Language Processing e intelligenza artificiale, il sistema è in grado di analizzare claims e fornire valutazioni motivate e trasparenti.

### Caratteristiche Principali

- **Pipeline di Ingestion Intelligente**: Acquisizione automatica da fonti news multiple con filtri qualità
- **Processamento Multilingue**: Supporto per italiano, inglese, spagnolo, francese e tedesco
- **Ricerca Semantica Avanzata**: Sistema di retrieval con query expansion e re-ranking
- **Fact-Checking Esplicabile**: Verifica con reasoning step-by-step e analisi bias
- **Dashboard Interattiva**: Interface web per esplorazione dati e fact-checking
- **API RESTful**: Integrazione facile con sistemi esterni

## 🏗️ Architettura

```
verifact_system/
├── core/              # Configurazioni e utility centrali
├── ingestion/         # Acquisizione dati da fonti multiple
├── processing/        # Elaborazione e normalizzazione testi
├── retrieval/         # Sistema di ricerca semantica
├── verification/      # Engine di fact-checking
├── models/           # Interfacce modelli ML
├── api/              # API REST
├── dashboard/        # Dashboard web Streamlit
└── scripts/          # Script di utilità
```

## 🚀 Quick Start

### Prerequisiti

- Python 3.9+
- PostgreSQL 13+ con estensione pgvector
- Redis (opzionale, per caching)
- Docker e Docker Compose (opzionale)

### Installazione Manuale

1. **Clona la repository**
```bash
git clone 
cd verifact_system
```

2. **Crea virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate     # Windows
```

3. **Installa dipendenze**
```bash
pip install -r requirements.txt
```

4. **Configura environment**
```bash
cp .env.example .env
# Modifica .env con le tue credenziali
```

5. **Setup database**
```bash
# Installa PostgreSQL con pgvector
# Poi esegui:
python scripts/setup_database.py
```

6. **Avvia il sistema**
```bash
# Pipeline di processing
python scripts/run_pipeline.py --mode daily

# Dashboard web
streamlit run dashboard/main_app.py

# API (in altro terminale)
python -m uvicorn api.endpoints:app --reload
```

### Installazione con Docker

```bash
# Avvia tutti i servizi
cd docker
docker-compose up -d

# Verifica stato
docker-compose ps

# Logs
docker-compose logs -f
```

## 📊 Configurazione

Il sistema utilizza file YAML per la configurazione. Modifica `config/development.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  database: verifact_dev
  username: verifact_user
  password: your_password

api:
  news_api_key: "your_newsapi_key"
  gemini_api_key: "your_gemini_key"

models:
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  llm_model: "gemma:3b"
```

## 🔧 Utilizzo

### Pipeline di Data Processing

```bash
# Ingestion giornaliera
python scripts/run_pipeline.py --mode daily --max-articles 100

# Ingestion continua
python scripts/run_pipeline.py --mode continuous

# Pipeline completa
python scripts/run_pipeline.py --mode full
```

### API Usage

```python
import requests

# Verifica claim
response = requests.post("http://localhost:8000/api/v1/verify", json={
    "claim": "Il cambiamento climatico è causato dall'attività umana",
    "language": "it"
})

result = response.json()
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}")
```

### Dashboard Web

Accedi a `http://localhost:8501` per:

- **Data Analytics**: Statistiche del dataset
- **Fact Checker**: Interface di verifica interattiva
- **System Monitor**: Monitoraggio performance

## 🧪 Testing

```bash
# Test unitari
pytest tests/unit/

# Test integrazione
pytest tests/integration/

# Coverage
pytest --cov=. --cov-report=html
```

## 📈 Performance

### Benchmark Tipici

- **Ingestion**: ~50 articoli/minuto
- **Processing**: ~100 documenti/minuto
- **Fact-checking**: ~5-10 secondi per claim
- **Database**: Supporta milioni di documenti

### Ottimizzazioni

- Usa Redis per caching frequente
- Configura pool connessioni database
- Usa GPU per modelli Transformer
- Implementa load balancing per API

## 🤝 Contributi

1. Fork della repository
2. Crea feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Apri Pull Request

### Guidelines

- Segui PEP 8 per Python
- Aggiungi test per nuove features
- Documenta funzioni pubbliche
- Usa type hints

## 📝 API Documentation

### Endpoints Principali

- `POST /api/v1/verify` - Verifica claim
- `GET /api/v1/articles` - Lista articoli
- `POST /api/v1/search` - Ricerca semantica
- `GET /api/v1/stats` - Statistiche sistema

Documentazione completa: `http://localhost:8000/docs`

## 🔒 Sicurezza

- Tutte le password in environment variables
- Rate limiting su API pubbliche
- Sanitizzazione input utente
- HTTPS in produzione
- Backup database regolari

## 📊 Monitoraggio

### Metriche Disponibili

- Accuratezza fact-checking
- Tempo risposta API
- Throughput processing
- Utilizzo risorse
- Qualità dataset

### Logs

```bash
# Logs applicazione
tail -f logs/verifact.log

# Logs Docker
docker-compose logs -f api
```

## 🛠️ Troubleshooting

### Problemi Comuni

**Database connection error**
```bash
# Verifica PostgreSQL attivo
systemctl status postgresql
# Controlla credenziali in .env
```

**Out of memory durante processing**
```bash
# Riduci batch_size in config
batch_size: 16  # invece di 32
```

**API lenta**
```bash
# Abilita caching Redis
REDIS_URL=redis://localhost:6379
```

## 🔮 Roadmap

- [ ] Supporto immagini e video
- [ ] Integrazione social media APIs
- [ ] Mobile app
- [ ] Plugin browser
- [ ] Multi-tenant support
- [ ] Advanced analytics
- [ ] Real-time notifications

## 📄 Licenza

MIT License - vedi `LICENSE` file per dettagli.

## 👥 Autori

- **Tuo Nome** - Sviluppo principale
- **Contributori** - Lista in `CONTRIBUTORS.md`

## 🙏 Acknowledgments

- Webhose per dataset news
- Sentence Transformers community  
- PostgreSQL e pgvector teams
- Streamlit team

---

**Note**: Questo è un progetto accademico per corso Big Data. Per uso in produzione, implementare misure di sicurezza aggiuntive.
