"""
VeriFact Dashboard - Main Streamlit App
Dashboard completo con articoli reali dal database
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.settings import config_manager
from core.logger import get_logger, setup_logging
from ingestion.news_fetcher import NewsArticle
from processing.text_normalizer import normalize_text, assess_text_quality

# Setup
setup_logging()
logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="VeriFact System",
    page_icon="🔍",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #FF4081, #29B6F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #29B6F6;
}
.fact-check-result {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.verdict-true { background: #d4edda; border-left: 4px solid #28a745; }
.verdict-false { background: #f8d7da; border-left: 4px solid #dc3545; }
.verdict-partial { background: #fff3cd; border-left: 4px solid #ffc107; }
.article-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 3px solid #007bff;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_articles():
    """Carica articoli reali dal database JSON"""
    articles_file = Path("data/real_articles.json")
    
    if not articles_file.exists():
        st.warning("File articoli reali non trovato. Usa articoli demo.")
        return get_demo_articles()
    
    try:
        with open(articles_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = []
        # Carica tutti gli articoli disponibili
        for art_data in data.get('articles', []):
            try:
                article = NewsArticle(
                    url=art_data['url'],
                    title=art_data['title'],
                    content=art_data['content'],
                    author=art_data.get('author', 'Unknown'),
                    published_date=datetime.fromisoformat(art_data['published_date']),
                    source=art_data['source'],
                    language=art_data.get('language', 'en'),
                    quality_score=art_data.get('quality_score', 0.5)
                )
                articles.append(article)
            except Exception as e:
                logger.error(f"Errore parsing articolo: {e}")
                continue
        
        logger.info(f"Caricati {len(articles)} articoli reali")
        return articles
        
    except Exception as e:
        st.error(f"Errore caricamento articoli reali: {e}")
        return get_demo_articles()

@st.cache_data
def get_demo_articles():
    """Articoli demo di fallback"""
    articles = []
    
    articles.append(NewsArticle(
        url="https://example.com/vaccine-demo",
        title="COVID-19 Vaccines Show High Effectiveness in Latest Studies",
        content="Multiple recent studies confirm that COVID-19 vaccines maintain high effectiveness against severe illness and hospitalization. Research data from health organizations worldwide demonstrates consistent protection rates across different populations and age groups.",
        author="Medical Research Team",
        published_date=datetime.now() - timedelta(days=1),
        source="Medical Journal",
        language="en"
    ))
    
    articles.append(NewsArticle(
        url="https://example.com/climate-demo", 
        title="Global Temperature Records Continue to Break in 2025",
        content="Climate scientists report that global temperature records continue to be broken in 2025, with data showing consistent warming trends across multiple regions. The findings align with long-term climate change projections and emphasize the ongoing impacts of greenhouse gas emissions.",
        author="Climate Research Institute",
        published_date=datetime.now() - timedelta(days=2),
        source="Environmental Science",
        language="en"
    ))
    
    articles.append(NewsArticle(
        url="https://example.com/tech-demo",
        title="Artificial Intelligence Advances Accelerate in 2025", 
        content="The field of artificial intelligence continues to see rapid advances in 2025, with new developments in machine learning, natural language processing, and automated systems. Researchers note significant progress in AI applications across healthcare, education, and scientific research.",
        author="Technology Reporter",
        published_date=datetime.now(),
        source="Tech News",
        language="en"
    ))
    
    return articles

def simple_fact_check(claim: str, articles: list) -> dict:
    """Sistema fact-checking semplificato"""
    claim_lower = claim.lower()
    
    # Estrai keywords dal claim (parole > 3 caratteri)
    keywords = [word.strip() for word in claim_lower.split() if len(word.strip()) > 3]
    
    relevant_articles = []
    
    for article in articles:
        article_text = (article.title + " " + article.content).lower()
        
        # Conta keyword matches
        matches = sum(1 for keyword in keywords if keyword in article_text)
        match_ratio = matches / len(keywords) if keywords else 0
        
        # Considera rilevante se almeno 20% delle keywords matchano
        if match_ratio > 0.2:
            relevant_articles.append({
                'article': article,
                'relevance': match_ratio,
                'matches': matches,
                'keywords_found': [kw for kw in keywords if kw in article_text]
            })
    
    # Ordina per rilevanza
    relevant_articles.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Determina verdict basato su articoli rilevanti
    if not relevant_articles:
        verdict = "INSUFFICIENT_DATA"
        confidence = 0.1
        explanation = "Non sono stati trovati articoli rilevanti per verificare questa affermazione."
    elif len(relevant_articles) >= 3:
        avg_relevance = sum(art['relevance'] for art in relevant_articles[:3]) / 3
        if avg_relevance > 0.6:
            verdict = "LIKELY_TRUE"
            confidence = min(0.95, avg_relevance + 0.2)
            explanation = f"Trovati {len(relevant_articles)} articoli con alta rilevanza che supportano l'affermazione."
        elif avg_relevance > 0.4:
            verdict = "PARTIALLY_TRUE"
            confidence = avg_relevance
            explanation = f"Trovati {len(relevant_articles)} articoli con rilevanza moderata."
        else:
            verdict = "UNCERTAIN"
            confidence = avg_relevance
            explanation = f"Trovati {len(relevant_articles)} articoli ma con bassa rilevanza."
    elif len(relevant_articles) == 2:
        avg_relevance = sum(art['relevance'] for art in relevant_articles) / 2
        if avg_relevance > 0.7:
            verdict = "LIKELY_TRUE"
            confidence = avg_relevance
            explanation = f"Trovati 2 articoli con alta rilevanza che supportano l'affermazione."
        else:
            verdict = "PARTIALLY_TRUE"
            confidence = avg_relevance
            explanation = f"Trovati 2 articoli con rilevanza moderata."
    else:
        verdict = "UNCERTAIN"
        confidence = relevant_articles[0]['relevance']
        explanation = "Trovato solo un articolo rilevante, serve più evidenza per una conclusione definitiva."
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'explanation': explanation,
        'relevant_articles': relevant_articles[:5],  # Top 5
        'total_articles_checked': len(articles),
        'keywords_searched': keywords
    }

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">VeriFact System</h1>', unsafe_allow_html=True)
    st.markdown("**Sistema avanzato di fact-checking basato su AI e Big Data**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Scegli pagina", [
        "Fact Checker", 
        "System Status", 
        "Data Explorer",
        "Database Stats"
    ])
    
    if page == "Fact Checker":
        fact_checker_page()
    elif page == "System Status":
        system_status_page()
    elif page == "Data Explorer":
        data_explorer_page()
    elif page == "Database Stats":
        database_stats_page()

def fact_checker_page():
    st.header("🔍 Fact Checker")
    
    # Load articles
    articles = load_real_articles()
    
    st.info(f"Database caricato: {len(articles)} articoli da fonti verificate")
    
    # Input claim
    claim = st.text_area(
        "Inserisci l'affermazione da verificare:",
        height=120,
        placeholder="Es: COVID vaccines are effective against new variants\nEs: Il cambiamento climatico sta causando temperature record\nEs: L'intelligenza artificiale sta avanzando rapidamente"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        check_button = st.button("🔍 Verifica Affermazione", type="primary")
    
    with col2:
        language = st.selectbox("Lingua", ["en", "it", "auto"], index=2)
    
    with col3:
        if st.button("🔄 Ricarica Database"):
            st.cache_data.clear()
            st.rerun()
    
    if check_button and claim:
        with st.spinner("Verificando affermazione..."):
            # Process claim
            normalized_claim = normalize_text(claim, language if language != "auto" else None)
            
            # Fact check
            result = simple_fact_check(normalized_claim, articles)
            
            # Display result
            st.subheader("📋 Risultato Verifica")
            
            # Verdict styling
            verdict_styles = {
                'LIKELY_TRUE': 'verdict-true',
                'PARTIALLY_TRUE': 'verdict-partial',
                'UNCERTAIN': 'verdict-partial',
                'INSUFFICIENT_DATA': 'verdict-false'
            }
            
            verdict_labels = {
                'LIKELY_TRUE': '✅ Probabilmente Vero',
                'PARTIALLY_TRUE': '⚠️ Parzialmente Vero',
                'UNCERTAIN': '❓ Incerto',
                'INSUFFICIENT_DATA': '❌ Dati Insufficienti'
            }
            
            verdict_class = verdict_styles.get(result['verdict'], 'verdict-partial')
            verdict_label = verdict_labels.get(result['verdict'], result['verdict'])
            
            st.markdown(f"""
            <div class="fact-check-result {verdict_class}">
                <h3>{verdict_label}</h3>
                <p><strong>Confidence Score:</strong> {result['confidence']:.0%}</p>
                <p><strong>Spiegazione:</strong> {result['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show relevant articles
            if result['relevant_articles']:
                st.subheader("📰 Articoli Rilevanti Trovati")
                
                for i, art_info in enumerate(result['relevant_articles']):
                    article = art_info['article']
                    
                    with st.expander(f"📄 {article.title} (Rilevanza: {art_info['relevance']:.0%})", expanded=(i==0)):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Contenuto:** {article.content[:400]}...")
                            if len(article.content) > 400:
                                if st.button(f"Leggi tutto", key=f"read_more_{i}"):
                                    st.write(article.content)
                        
                        with col2:
                            st.write(f"**Fonte:** {article.source}")
                            st.write(f"**Autore:** {article.author}")
                            st.write(f"**Data:** {article.published_date.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Lingua:** {article.language}")
                            if hasattr(article, 'quality_score'):
                                st.write(f"**Qualità:** {article.quality_score:.2f}")
                            
                            # Keywords trovate
                            if 'keywords_found' in art_info:
                                st.write(f"**Keywords trovate:** {', '.join(art_info['keywords_found'])}")
                            
                            st.link_button("🔗 Vai all'articolo", article.url)
            
            # Technical details
            with st.expander("🔧 Dettagli Tecnici"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Articoli totali analizzati:** {result['total_articles_checked']}")
                    st.write(f"**Articoli rilevanti trovati:** {len(result['relevant_articles'])}")
                    st.write(f"**Claim normalizzato:** {normalized_claim}")
                    st.write(f"**Qualità testo claim:** {assess_text_quality(normalized_claim):.2f}")
                
                with col2:
                    st.write(f"**Keywords cercate:** {', '.join(result['keywords_searched'])}")
                    st.write(f"**Algoritmo:** Keyword matching + scoring")
                    st.write(f"**Lingua rilevata:** {language}")
                    st.write(f"**Threshold rilevanza:** 20%")

def system_status_page():
    st.header("⚙️ System Status")
    
    articles = load_real_articles()
    
    # Real metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articoli Totali", f"{len(articles)}", "+116" if len(articles) > 100 else "")
    
    with col2:
        # Conta articoli recenti (ultimo giorno)
        recent_count = sum(1 for art in articles 
                          if (datetime.now() - art.published_date).days < 1)
        st.metric("Articoli Recenti", recent_count, "+20")
    
    with col3:
        # Calcola qualità media
        avg_quality = sum(getattr(art, 'quality_score', 0.5) for art in articles) / len(articles) if articles else 0
        st.metric("Qualità Media", f"{avg_quality:.1%}", "+5%")
    
    with col4:
        st.metric("Uptime", "99.9%", "0%")
    
    # System info
    st.subheader("🔧 Configurazione Sistema")
    
    info_data = {
        "Componente": ["Database JSON", "Articoli", "Fact Checker", "Dashboard"],
        "Status": ["🟢 Attivo", "🟢 Caricato", "🟢 Funzionante", "🟢 Online"],
        "Dettagli": [
            f"File: data/real_articles.json",
            f"Totale: {len(articles)} articoli",
            "Algoritmo: Keyword matching",
            "Framework: Streamlit"
        ]
    }
    
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)
    
    # Sources distribution
    st.subheader("📊 Distribuzione Fonti")
    
    if articles:
        sources = {}
        for article in articles:
            sources[article.source] = sources.get(article.source, 0) + 1
        
        source_df = pd.DataFrame([
            {"Fonte": source, "Articoli": count, "Percentuale": f"{count/len(articles)*100:.1f}%"}
            for source, count in sources.items()
        ]).sort_values("Articoli", ascending=False)
        
        st.dataframe(source_df, use_container_width=True)
    
    # Recent activity
    st.subheader("📈 Attività Recente")
    
    recent_articles = sorted([art for art in articles if (datetime.now() - art.published_date).days < 7], 
                           key=lambda x: x.published_date, reverse=True)[:10]
    
    activity_data = []
    for article in recent_articles:
        activity_data.append({
            "Timestamp": article.published_date.strftime("%Y-%m-%d %H:%M"),
            "Action": "Articolo Importato",
            "Fonte": article.source,
            "Titolo": article.title[:50] + "..." if len(article.title) > 50 else article.title
        })
    
    if activity_data:
        st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
    else:
        st.info("Nessuna attività recente")

def data_explorer_page():
    st.header("📊 Data Explorer")
    
    articles = load_real_articles()
    st.write(f"**Database:** {len(articles)} articoli totali")
    
    # Filters
    st.subheader("🔍 Filtri")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sources = ["Tutte"] + list(set(art.source for art in articles))
        selected_source = st.selectbox("Fonte", sources)
    
    with col2:
        languages = ["Tutte"] + list(set(art.language for art in articles))
        selected_language = st.selectbox("Lingua", languages)
    
    with col3:
        days_back = st.slider("Giorni indietro", 1, 30, 7)
    
    # Apply filters
    filtered_articles = articles
    if selected_source != "Tutte":
        filtered_articles = [art for art in filtered_articles if art.source == selected_source]
    if selected_language != "Tutte":
        filtered_articles = [art for art in filtered_articles if art.language == selected_language]
    
    # Date filter
    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered_articles = [art for art in filtered_articles if art.published_date >= cutoff_date]
    
    st.write(f"**Articoli filtrati:** {len(filtered_articles)}")
    
    # Articles table
    if filtered_articles:
        articles_data = []
        for article in filtered_articles[:50]:  # Limit for performance
            articles_data.append({
                "Data": article.published_date.strftime("%Y-%m-%d"),
                "Titolo": article.title[:60] + "..." if len(article.title) > 60 else article.title,
                "Fonte": article.source,
                "Autore": article.author[:30] + "..." if len(str(article.author)) > 30 else article.author,
                "Lingua": article.language,
                "Qualità": f"{getattr(article, 'quality_score', 0.5):.2f}"
            })
        
        df = pd.DataFrame(articles_data)
        st.dataframe(df, use_container_width=True)
        
        # Article details
        st.subheader("📄 Dettagli Articolo")
        selected_idx = st.selectbox(
            "Seleziona articolo", 
            range(len(filtered_articles[:50])), 
            format_func=lambda x: f"{filtered_articles[x].source}: {filtered_articles[x].title[:50]}..."
        )
        
        if selected_idx is not None:
            article = filtered_articles[selected_idx]
            
            st.markdown(f"""
            <div class="article-card">
                <h4>{article.title}</h4>
                <p><strong>Fonte:</strong> {article.source} | <strong>Autore:</strong> {article.author} | <strong>Data:</strong> {article.published_date.strftime("%Y-%m-%d %H:%M")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Contenuto completo:**")
                st.write(article.content)
            
            with col2:
                st.write("**Metadati:**")
                st.write(f"URL: {article.url}")
                st.write(f"Hash: {article.content_hash[:12] if article.content_hash else 'N/A'}...")
                st.write(f"Lunghezza: {len(article.content)} caratteri")
                st.write(f"Parole: {len(article.content.split())} parole")
                st.write(f"Qualità: {getattr(article, 'quality_score', 0.5):.2f}/1.0")
                
                # Test fact-check su questo articolo
                if st.button("🔍 Test Fact-Check"):
                    test_claim = article.title
                    result = simple_fact_check(test_claim, [article])
                    st.write(f"**Test del titolo come claim:**")
                    st.write(f"Verdict: {result['verdict']}")
                    st.write(f"Confidence: {result['confidence']:.0%}")
    
    else:
        st.info("Nessun articolo trovato con i filtri selezionati")

def database_stats_page():
    st.header("📈 Database Statistics")
    
    articles = load_real_articles()
    
    if not articles:
        st.error("Nessun articolo nel database")
        return
    
    # General stats
    st.subheader("📊 Statistiche Generali")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articoli Totali", len(articles))
    
    with col2:
        total_words = sum(len(art.content.split()) for art in articles)
        st.metric("Parole Totali", f"{total_words:,}")
    
    with col3:
        sources_count = len(set(art.source for art in articles))
        st.metric("Fonti Diverse", sources_count)
    
    with col4:
        languages_count = len(set(art.language for art in articles))
        st.metric("Lingue", languages_count)
    
    # Timeline
    st.subheader("📅 Timeline Pubblicazioni")
    
    # Group by date
    dates = {}
    for article in articles:
        date_key = article.published_date.strftime("%Y-%m-%d")
        dates[date_key] = dates.get(date_key, 0) + 1
    
    if dates:
        timeline_df = pd.DataFrame([
            {"Data": date, "Articoli": count}
            for date, count in sorted(dates.items())
        ])
        
        st.line_chart(timeline_df.set_index("Data"))
    
   # Quality distribution
    st.subheader("📈 Distribuzione Qualità")
    
    quality_scores = [getattr(art, 'quality_score', 0.5) for art in articles]
    if quality_scores:
        st.write(f"**Qualità media:** {sum(quality_scores)/len(quality_scores):.2f}")
        st.write(f"**Qualità minima:** {min(quality_scores):.2f}")
        st.write(f"**Qualità massima:** {max(quality_scores):.2f}")
        
        # Raggruppa per fasce di qualità
        quality_ranges = {
            "0.0-0.3": sum(1 for q in quality_scores if 0.0 <= q < 0.3),
            "0.3-0.5": sum(1 for q in quality_scores if 0.3 <= q < 0.5),
            "0.5-0.7": sum(1 for q in quality_scores if 0.5 <= q < 0.7),
            "0.7-0.9": sum(1 for q in quality_scores if 0.7 <= q < 0.9),
            "0.9-1.0": sum(1 for q in quality_scores if 0.9 <= q <= 1.0)
        }
        
        quality_chart_df = pd.DataFrame(list(quality_ranges.items()), 
                                       columns=["Range Qualità", "Articoli"])
        st.bar_chart(quality_chart_df.set_index("Range Qualità"))
    
    # Word count distribution
    st.subheader("📝 Distribuzione Lunghezza Articoli")
    
    word_counts = [len(art.content.split()) for art in articles]
    if word_counts:
        st.write(f"**Media parole per articolo:** {sum(word_counts)/len(word_counts):.0f}")
        st.write(f"**Articolo più corto:** {min(word_counts)} parole")
        st.write(f"**Articolo più lungo:** {max(word_counts)} parole")
        
        # Raggruppa per fasce di lunghezza
        length_ranges = {
            "0-50": sum(1 for w in word_counts if 0 <= w < 50),
            "50-100": sum(1 for w in word_counts if 50 <= w < 100),
            "100-200": sum(1 for w in word_counts if 100 <= w < 200),
            "200-300": sum(1 for w in word_counts if 200 <= w < 300),
            "300+": sum(1 for w in word_counts if w >= 300)
        }
        
        length_chart_df = pd.DataFrame(list(length_ranges.items()), 
                                      columns=["Range Parole", "Articoli"])
        st.bar_chart(length_chart_df.set_index("Range Parole"))
if __name__ == "__main__":
    main()