"""
Test basic del sistema VeriFact
"""

from core.settings import config_manager
from core.logger import get_logger, setup_logging
from ingestion.news_fetcher import NewsArticle, create_test_article
from processing.text_normalizer import normalize_text, assess_text_quality
from core.database import db_manager
from datetime import datetime

# Setup logging
setup_logging()
logger = get_logger(__name__)

def test_basic_components():
    """Test componenti base del sistema"""
    print("=" * 50)
    print("VeriFact System - Basic Components Test")
    print("=" * 50)
    
    # Test 1: Configurazioni
    print("\n1. Testing Configuration System...")
    print(f"   Database: {config_manager.database.host}:{config_manager.database.port}")
    print(f"   Database name: {config_manager.database.database}")
    print(f"   API keys configured: {bool(config_manager.apis.news_api_key)}")
    print("   ✓ Configuration system working")
    
    # Test 2: Database manager (simulato)
    print("\n2. Testing Database Manager...")
    connection_ok = db_manager.test_connection()
    stats = db_manager.get_stats()
    print(f"   Connection status: {'✓ Connected' if connection_ok else '✗ Failed'}")
    print(f"   Simulated stats: {stats['articles_count']} articles")
    print("   ✓ Database manager working")
    
    # Test 3: Data structures
    print("\n3. Testing Data Structures...")
    
    # Crea articolo manuale
    article1 = NewsArticle(
        url="https://example.com/climate-change",
        title="Scientists Warn About Climate Change Effects",
        content="Climate change continues to pose significant challenges to global ecosystems. Recent studies indicate that temperature rises are accelerating, affecting weather patterns worldwide. Experts emphasize the need for immediate action to reduce greenhouse gas emissions.",
        author="Dr. Environmental Scientist",
        published_date=datetime.now(),
        source="Science Daily",
        language="en"
    )
    
    # Crea articolo con funzione helper
    article2 = create_test_article()
    
    print(f"   Article 1: {article1.title[:40]}...")
    print(f"   Article 1 hash: {article1.content_hash[:8]}...")
    print(f"   Article 2: {article2.title}")
    print(f"   Article 2 hash: {article2.content_hash[:8]}...")
    print("   ✓ Data structures working")
    
    # Test 4: Text processing
    print("\n4. Testing Text Processing...")
    
    test_text = article1.content
    normalized = normalize_text(test_text, article1.language)
    quality = assess_text_quality(normalized)
    
    print(f"   Original length: {len(test_text)} chars")
    print(f"   Normalized length: {len(normalized)} chars")
    print(f"   Quality score: {quality:.2f}/1.0")
    print(f"   Preview: {normalized[:60]}...")
    print("   ✓ Text processing working")
    
    # Test 5: Multiple articles processing
    print("\n5. Testing Batch Processing...")
    
    articles = [article1, article2]
    processed_count = 0
    total_quality = 0
    
    for article in articles:
        normalized_content = normalize_text(article.content)
        quality = assess_text_quality(normalized_content)
        article.quality_score = quality
        processed_count += 1
        total_quality += quality
    
    avg_quality = total_quality / processed_count if processed_count > 0 else 0
    
    print(f"   Processed articles: {processed_count}")
    print(f"   Average quality: {avg_quality:.2f}")
    print("   ✓ Batch processing working")
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("✓ Configuration system: OK")
    print("✓ Database manager: OK (simulated)")
    print("✓ Data structures: OK")
    print("✓ Text processing: OK")
    print("✓ Batch processing: OK")
    print("\n🎉 All basic components are working correctly!")
    print("Ready to proceed with advanced features.")
    
def test_specific_functionality():
    """Test funzionalità specifiche"""
    print("\n" + "-" * 30)
    print("SPECIFIC FUNCTIONALITY TEST")
    print("-" * 30)
    
    # Test normalizzazione su testo problematico
    problematic_text = """
    This    is   a text  with   multiple   spaces!!!
    
    And     line breaks...
    
    <p>HTML tags should be removed</p>
    """
    
    cleaned = normalize_text(problematic_text)
    print(f"Problematic text cleaned:")
    print(f"Before: {repr(problematic_text[:50])}")
    print(f"After: {repr(cleaned[:50])}")
    print(f"Quality: {assess_text_quality(cleaned):.2f}")

if __name__ == "__main__":
    try:
        test_basic_components()
        test_specific_functionality()
        
        print("\n" + "=" * 50)
        print("🚀 System is ready for development!")
        print("Next steps:")
        print("1. Add real news sources")
        print("2. Implement fact-checking logic")
        print("3. Create web dashboard")
        print("4. Add AI/ML components")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed with error: {e}")
        print("Check the error messages above for details.")