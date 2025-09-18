import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from minicheck_final import MiniCheckCorrect
from groq_factchecker import GroqFactChecker
from dataset_loader import FactCheckDatasetLoader
import matplotlib.pyplot as plt
import seaborn as sns

class LargeScaleComparison:
    def __init__(self):
        print("Initializing Large Scale Comparison Pipeline...")
        
        # Inizializza sistemi
        self.minicheck = MiniCheckCorrect()
        self.groq = GroqFactChecker()
        self.loader = FactCheckDatasetLoader()
        
        print("All systems ready!")
    
    def run_comprehensive_evaluation(self):
        """
        Esegue valutazione completa su tutti i dataset
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE FACT-CHECKING EVALUATION")
        print("="*60)
        
        # Carica tutti i dataset
        all_datasets = {}
        
        # Dataset custom (piccolo ma controllato)
        docs, claims, labels, name = self.loader.load_custom_examples()
        all_datasets[name] = {'docs': docs, 'claims': claims, 'labels': labels}
        
        # Dataset sintetico FEVER (medio)
        docs, claims, labels, name = self.loader.create_synthetic_fever(40)
        all_datasets[name] = {'docs': docs, 'claims': claims, 'labels': labels}
        
        # Dataset misto grande
        docs, claims, labels, name = self.loader.create_large_mixed_dataset(60)
        all_datasets[name] = {'docs': docs, 'claims': claims, 'labels': labels}
        
        # Esegui confronti
        all_results = {}
        total_cost = 0
        total_time_minicheck = 0
        total_time_groq = 0
        
        for dataset_name, dataset in all_datasets.items():
            print(f"\n{'-'*50}")
            print(f"EVALUATING: {dataset_name}")
            print(f"Size: {len(dataset['docs'])} examples")
            
            result = self.evaluate_dataset(
                dataset['docs'], 
                dataset['claims'], 
                dataset['labels'],
                dataset_name
            )
            
            all_results[dataset_name] = result
            total_cost += result.get('groq_cost', 0)
            total_time_minicheck += result.get('minicheck_time', 0)
            total_time_groq += result.get('groq_time', 0)
        
        # Analisi aggregata
        self.generate_final_report(all_results, total_cost, total_time_minicheck, total_time_groq)
        
        return all_results
    
    def evaluate_dataset(self, docs, claims, labels, dataset_name):
        """
        Valuta un singolo dataset
        """
        results = {
            'dataset_name': dataset_name,
            'docs': docs,
            'claims': claims,
            'true_labels': labels
        }
        
        # MiniCheck evaluation
        print("Running MiniCheck...")
        start_time = time.time()
        minicheck_preds, minicheck_probs = self.minicheck.score(docs, claims)
        minicheck_time = time.time() - start_time
        
        # Groq evaluation (con rate limiting più aggressivo per dataset grandi)
        print("Running Groq...")
        delay = 0.3 if len(docs) > 20 else 0.5  # Adatta delay per dataset grandi
        start_time = time.time()
        groq_preds, groq_explanations, token_count = self.groq.batch_fact_check(
            docs, claims, delay=delay
        )
        groq_time = time.time() - start_time
        
        # Calcola metriche
        minicheck_acc = balanced_accuracy_score(labels, minicheck_preds)
        groq_acc = balanced_accuracy_score(labels, groq_preds)
        
        # Agreement tra sistemi
        agreement = sum(1 for m, g in zip(minicheck_preds, groq_preds) if m == g)
        agreement_rate = agreement / len(minicheck_preds)
        
        # Costo stimato
        estimated_cost = (token_count / 1000000) * 0.50  # $0.50 per M input tokens
        
        # Salva risultati
        results.update({
            'minicheck_preds': minicheck_preds,
            'minicheck_probs': minicheck_probs,
            'minicheck_acc': minicheck_acc,
            'minicheck_time': minicheck_time,
            'groq_preds': groq_preds,
            'groq_explanations': groq_explanations,
            'groq_acc': groq_acc,
            'groq_time': groq_time,
            'groq_tokens': token_count,
            'groq_cost': estimated_cost,
            'agreement_rate': agreement_rate,
            'size': len(docs)
        })
        
        # Report per questo dataset
        print(f"\nRESULTS for {dataset_name}:")
        print(f"  MiniCheck Accuracy: {minicheck_acc:.3f}")
        print(f"  Groq Accuracy:      {groq_acc:.3f}")
        print(f"  Agreement Rate:     {agreement_rate:.3f}")
        print(f"  MiniCheck Speed:    {len(docs)/minicheck_time:.1f} examples/sec")
        print(f"  Groq Speed:         {len(docs)/groq_time:.1f} examples/sec")
        print(f"  Estimated Cost:     ${estimated_cost:.4f}")
        
        return results
    
    def generate_final_report(self, all_results, total_cost, total_time_minicheck, total_time_groq):
        """
        Genera report finale aggregato
        """
        print("\n" + "="*60)
        print("FINAL COMPREHENSIVE REPORT")
        print("="*60)
        
        # Statistiche aggregate
        total_examples = sum(r['size'] for r in all_results.values())
        weighted_minicheck_acc = sum(r['minicheck_acc'] * r['size'] for r in all_results.values()) / total_examples
        weighted_groq_acc = sum(r['groq_acc'] * r['size'] for r in all_results.values()) / total_examples
        weighted_agreement = sum(r['agreement_rate'] * r['size'] for r in all_results.values()) / total_examples
        
        print(f"\nAGGREGATE PERFORMANCE:")
        print(f"  Total Examples Processed: {total_examples}")
        print(f"  Weighted MiniCheck Accuracy: {weighted_minicheck_acc:.3f}")
        print(f"  Weighted Groq Accuracy:      {weighted_groq_acc:.3f}")
        print(f"  Weighted Agreement Rate:     {weighted_agreement:.3f}")
        
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"  Total MiniCheck Time: {total_time_minicheck:.1f}s")
        print(f"  Total Groq Time:      {total_time_groq:.1f}s")
        print(f"  MiniCheck Throughput: {total_examples/total_time_minicheck:.1f} examples/sec")
        print(f"  Groq Throughput:      {total_examples/total_time_groq:.1f} examples/sec")
        print(f"  Speed Advantage:      {(total_time_minicheck/total_time_groq):.1f}x faster (Groq)")
        
        print(f"\nCOST ANALYSIS:")
        print(f"  Total Groq Cost:      ${total_cost:.4f}")
        print(f"  Cost per Example:     ${total_cost/total_examples:.6f}")
        print(f"  Cost per 1K Examples: ${(total_cost/total_examples)*1000:.3f}")
        
        # Analisi per dataset
        print(f"\nPER-DATASET BREAKDOWN:")
        for name, result in all_results.items():
            print(f"  {name}:")
            print(f"    Size: {result['size']} | Mini: {result['minicheck_acc']:.3f} | Groq: {result['groq_acc']:.3f} | Agreement: {result['agreement_rate']:.3f}")
        
        print(f"\n" + "="*60)
        print("PAPER INSIGHTS:")
        
        if weighted_minicheck_acc > weighted_groq_acc:
            print(f"  • MiniCheck outperforms Groq by {(weighted_minicheck_acc - weighted_groq_acc):.3f}")
        else:
            print(f"  • Groq outperforms MiniCheck by {(weighted_groq_acc - weighted_minicheck_acc):.3f}")
        
        if total_time_groq < total_time_minicheck:
            speedup = total_time_minicheck / total_time_groq
            print(f"  • Groq is {speedup:.1f}x faster than MiniCheck")
        else:
            speedup = total_time_groq / total_time_minicheck
            print(f"  • MiniCheck is {speedup:.1f}x faster than Groq")
        
        print(f"  • High agreement rate ({weighted_agreement:.3f}) suggests consistent fact-checking")
        print(f"  • Groq cost of ${total_cost:.4f} for {total_examples} examples shows scalability")
        
        return {
            'total_examples': total_examples,
            'weighted_minicheck_acc': weighted_minicheck_acc,
            'weighted_groq_acc': weighted_groq_acc,
            'weighted_agreement': weighted_agreement,
            'total_cost': total_cost,
            'speedup_factor': total_time_minicheck / total_time_groq if total_time_groq > 0 else 1
        }

def main():
    """
    Esegue la valutazione completa
    """
    print("Starting Large Scale Fact-Checking Comparison...")
    
    evaluator = LargeScaleComparison()
    results = evaluator.run_comprehensive_evaluation()
    
    print(f"\nEvaluation complete! Results ready for paper analysis.")
    print(f"Key findings available in the final report above.")
    
    return results

if __name__ == "__main__":
    main()