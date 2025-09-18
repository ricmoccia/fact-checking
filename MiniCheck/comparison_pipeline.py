import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, classification_report
from minicheck_final import MiniCheckCorrect
from groq_factchecker import GroqFactChecker

class FactCheckComparison:
    def __init__(self):
        """
        Inizializza entrambi i sistemi di fact-checking
        """
        print("Initializing comparison pipeline...")
        
        # Inizializza MiniCheck
        print("Loading MiniCheck...")
        self.minicheck = MiniCheckCorrect()
        
        # Inizializza Groq
        print("Loading Groq...")
        self.groq = GroqFactChecker()
        
        print("✅ Both systems ready!")
    
    def compare_on_sample(self, docs, claims, true_labels=None, sample_name="Test"):
        """
        Confronta MiniCheck e Groq su un set di esempi
        """
        print(f"\n=== {sample_name} Comparison ===")
        print(f"Processing {len(docs)} examples...")
        
        results = {
            'docs': docs,
            'claims': claims,
            'true_labels': true_labels if true_labels is not None else [None] * len(docs)
        }
        
        # MiniCheck predictions
        print("\nRunning MiniCheck...")
        start_time = time.time()
        minicheck_preds, minicheck_probs = self.minicheck.score(docs, claims)
        minicheck_time = time.time() - start_time
        
        results['minicheck_preds'] = minicheck_preds
        results['minicheck_probs'] = minicheck_probs
        results['minicheck_time'] = minicheck_time
        
        # Groq predictions
        print("\nRunning Groq...")
        start_time = time.time()
        groq_preds, groq_explanations, token_count = self.groq.batch_fact_check(
            docs, claims, delay=0.5  # Rate limiting
        )
        groq_time = time.time() - start_time
        
        results['groq_preds'] = groq_preds
        results['groq_explanations'] = groq_explanations
        results['groq_time'] = groq_time
        results['groq_tokens'] = token_count
        
        # Analisi risultati
        self.analyze_results(results, sample_name)
        
        return results
    
    def analyze_results(self, results, sample_name):
        """
        Analizza e mostra i risultati del confronto
        """
        print(f"\n=== {sample_name} Results Analysis ===")
        
        minicheck_preds = results['minicheck_preds']
        groq_preds = results['groq_preds']
        true_labels = results['true_labels']
        
        # Performance metrics
        if true_labels[0] is not None:
            minicheck_acc = balanced_accuracy_score(true_labels, minicheck_preds)
            groq_acc = balanced_accuracy_score(true_labels, groq_preds)
            
            print(f"📊 Accuracy Comparison:")
            print(f"  MiniCheck: {minicheck_acc:.3f}")
            print(f"  Groq:      {groq_acc:.3f}")
        
        # Agreement between systems
        agreement = sum(1 for m, g in zip(minicheck_preds, groq_preds) if m == g)
        agreement_rate = agreement / len(minicheck_preds)
        print(f"🤝 System Agreement: {agreement_rate:.3f} ({agreement}/{len(minicheck_preds)})")
        
        # Speed comparison
        print(f"⚡ Speed Comparison:")
        print(f"  MiniCheck: {results['minicheck_time']:.2f}s ({len(minicheck_preds)/results['minicheck_time']:.1f} examples/sec)")
        print(f"  Groq:      {results['groq_time']:.2f}s ({len(groq_preds)/results['groq_time']:.1f} examples/sec)")
        
        # Cost estimation (Groq)
        estimated_cost = (results['groq_tokens'] / 1000000) * 0.50  # $0.50 per M tokens (approximate)
        print(f"💰 Estimated Groq Cost: ~${estimated_cost:.4f} ({results['groq_tokens']} tokens)")
        
        # Disagreement examples
        disagreements = [(i, minicheck_preds[i], groq_preds[i]) 
                        for i in range(len(minicheck_preds)) 
                        if minicheck_preds[i] != groq_preds[i]]
        
        if disagreements:
            print(f"\n🔍 Disagreement Examples ({len(disagreements)} total):")
            for i, (idx, mini_pred, groq_pred) in enumerate(disagreements[:3]):  # Show first 3
                print(f"  Example {idx+1}:")
                print(f"    Doc: {results['docs'][idx][:100]}...")
                print(f"    Claim: {results['claims'][idx][:100]}...")
                print(f"    MiniCheck: {mini_pred}, Groq: {groq_pred}")
                if true_labels[0] is not None:
                    print(f"    True: {true_labels[idx]}")
                print()

def run_basic_test():
    """
    Test di base con esempi semplici
    """
    print("🚀 Starting Basic Test...")
    
    # Esempi di test
    test_docs = [
        "A group of students gather in the school library to study for their upcoming final exams.",
        "The weather today is sunny with a temperature of 75 degrees Fahrenheit.",
        "Apple Inc. reported record quarterly revenue of $123.9 billion for Q1 2024.",
        "The meeting is scheduled for tomorrow at 3 PM in the conference room."
    ]
    
    test_claims = [
        "The students are preparing for an examination.",
        "It is raining heavily today.",
        "Apple's Q1 2024 revenue exceeded $120 billion.",
        "The meeting will take place next week."
    ]
    
    expected_labels = [1, 0, 1, 0]  # Expected correct answers
    
    # Esegui confronto
    pipeline = FactCheckComparison()
    results = pipeline.compare_on_sample(
        test_docs, test_claims, expected_labels, 
        sample_name="Basic Test"
    )
    
    return results

def run_custom_test():
    """
    Test personalizzato con i tuoi esempi
    """
    print("\n" + "="*60)
    print("🔧 Custom Test - Add your own examples!")
    
    # Puoi aggiungere i tuoi esempi qui
    custom_docs = [
        "The restaurant serves Italian cuisine and is open from 6 PM to midnight.",
        "Python is a programming language that was first released in 1991."
    ]
    
    custom_claims = [
        "The restaurant is closed during lunch hours.",
        "Python was created in the 1980s."
    ]
    
    custom_labels = [1, 0]  # Aggiusta secondo le tue aspettative
    
    pipeline = FactCheckComparison()
    results = pipeline.compare_on_sample(
        custom_docs, custom_claims, custom_labels,
        sample_name="Custom Test"
    )
    
    return results

if __name__ == "__main__":
    # Esegui test di base
    basic_results = run_basic_test()
    
    # Esegui test personalizzato
    custom_results = run_custom_test()
    
    print("\n" + "="*60)
    print("🎉 Comparison Pipeline Complete!")
    print("\nNext steps for your paper:")
    print("1. ✅ MiniCheck replica funzionante")
    print("2. ✅ Groq API configurato")
    print("3. ✅ Pipeline di confronto pronta")
    print("4. 📝 Ora puoi processare dataset più grandi")
    print("5. 📊 Implementare metriche aggiuntive (costo, velocità, etc.)")