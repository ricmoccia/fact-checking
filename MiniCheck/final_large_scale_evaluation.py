import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, classification_report
from minicheck_final import MiniCheckCorrect
from groq_factchecker import GroqFactChecker

def load_large_dataset():
    """
    Carica il dataset da 1500 esempi
    """
    print("Loading large-scale dataset...")
    
    df = pd.read_csv('large_fact_check_dataset.csv')
    docs = df['doc'].tolist()
    claims = df['claim'].tolist()
    labels = df['label'].tolist()
    
    print(f"Dataset loaded: {len(docs)} examples")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return docs, claims, labels

def run_final_evaluation(subset_size=None):
    """
    Esegue la valutazione finale su dataset grande
    """
    print("FINAL LARGE-SCALE EVALUATION")
    print("="*60)
    
    # Carica dataset
    docs, claims, labels = load_large_dataset()
    
    # Opzione per testare su subset (per gestire costi Groq)
    if subset_size and subset_size < len(docs):
        print(f"Using subset of {subset_size} examples for cost management")
        docs = docs[:subset_size]
        claims = claims[:subset_size]
        labels = labels[:subset_size]
    
    print(f"Evaluating on {len(docs)} examples...")
    
    # Inizializza sistemi
    print("Initializing systems...")
    minicheck = MiniCheckCorrect()
    groq = GroqFactChecker()
    
    # Evaluation MiniCheck
    print("\nRunning MiniCheck evaluation...")
    start_time = time.time()
    minicheck_preds, minicheck_probs = minicheck.score(docs, claims)
    minicheck_time = time.time() - start_time
    
    # Evaluation Groq (con rate limiting)
    print(f"\nRunning Groq evaluation (this will take ~{len(docs)*0.4/60:.1f} minutes)...")
    start_time = time.time()
    groq_preds, groq_explanations, token_count = groq.batch_fact_check(
        docs, claims, delay=0.4  # Rate limiting per dataset grande
    )
    groq_time = time.time() - start_time
    
    # Calcola metriche
    minicheck_acc = balanced_accuracy_score(labels, minicheck_preds)
    groq_acc = balanced_accuracy_score(labels, groq_preds)
    
    # Agreement
    agreement = sum(1 for m, g in zip(minicheck_preds, groq_preds) if m == g)
    agreement_rate = agreement / len(minicheck_preds)
    
    # Costi
    estimated_cost = (token_count / 1000000) * 0.50
    
    # Report finale
    print("\n" + "="*60)
    print("FINAL RESULTS - PAPER QUALITY")
    print("="*60)
    
    print(f"\nDATASET:")
    print(f"  Examples: {len(docs)}")
    print(f"  Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Negative: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    print(f"\nACCURACY RESULTS:")
    print(f"  MiniCheck: {minicheck_acc:.3f}")
    print(f"  Groq (LLaMA-3.3-70B): {groq_acc:.3f}")
    if minicheck_acc > groq_acc:
        print(f"  Winner: MiniCheck (+{minicheck_acc-groq_acc:.3f})")
    else:
        print(f"  Winner: Groq (+{groq_acc-minicheck_acc:.3f})")
    
    print(f"\nSYSTEM AGREEMENT:")
    print(f"  Agreement Rate: {agreement_rate:.3f}")
    print(f"  Disagreements: {len(docs)-agreement} examples")
    
    print(f"\nPERFORMANCE:")
    print(f"  MiniCheck Time: {minicheck_time:.1f}s ({len(docs)/minicheck_time:.1f} ex/sec)")
    print(f"  Groq Time: {groq_time:.1f}s ({len(docs)/groq_time:.1f} ex/sec)")
    print(f"  Speed Ratio: {groq_time/minicheck_time:.1f}x (Groq vs MiniCheck)")
    
    print(f"\nCOST ANALYSIS:")
    print(f"  Groq Tokens: {token_count:,}")
    print(f"  Estimated Cost: ${estimated_cost:.4f}")
    print(f"  Cost per Example: ${estimated_cost/len(docs):.6f}")
    print(f"  Cost for 10K examples: ${(estimated_cost/len(docs))*10000:.2f}")
    
    # Analisi errori (campione)
    print(f"\nERROR ANALYSIS (first 5 disagreements):")
    disagreement_count = 0
    for i, (mini_pred, groq_pred) in enumerate(zip(minicheck_preds, groq_preds)):
        if mini_pred != groq_pred and disagreement_count < 5:
            print(f"\n  Example {i+1}:")
            print(f"    Doc: {docs[i][:100]}...")
            print(f"    Claim: {claims[i][:80]}...")
            print(f"    True: {labels[i]}, MiniCheck: {mini_pred}, Groq: {groq_pred}")
            disagreement_count += 1
    
    return {
        'dataset_size': len(docs),
        'minicheck_acc': minicheck_acc,
        'groq_acc': groq_acc,
        'agreement_rate': agreement_rate,
        'minicheck_time': minicheck_time,
        'groq_time': groq_time,
        'total_cost': estimated_cost
    }

if __name__ == "__main__":
    # Per gestire i costi, inizia con un subset
    print("Choose evaluation size:")
    print("1. Small test (100 examples)")
    print("2. Medium test (500 examples)")  
    print("3. Full dataset (1500 examples)")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        results = run_final_evaluation(subset_size=100)
    elif choice == "2":
        results = run_final_evaluation(subset_size=500)
    else:
        results = run_final_evaluation()  # Full dataset
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print("Results are ready for your fact-checking paper.")