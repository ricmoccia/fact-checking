import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch.nn.functional as F

class MiniCheckCorrect:
    def __init__(self, model_name='lytang/MiniCheck-RoBERTa-Large', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def score(self, docs, claims):
        """
        Verifica se le claims sono supportate dai documenti
        Formato corretto: documento [SEP] claim
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for doc, claim in zip(docs, claims):
                # FORMATO CORRETTO: documento [SEP] claim
                inputs = self.tokenizer(
                    doc, claim,  # Prima doc, poi claim
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
                # Probabilità che sia supportato (classe 1)
                prob_supported = probs[0][1].cpu().item()
                pred = 1 if prob_supported > 0.5 else 0
                
                predictions.append(pred)
                probabilities.append(prob_supported)
        
        return predictions, probabilities

def test_complete():
    print("=== MiniCheck Final Test ===")
    
    # Test di base
    doc = "A group of students gather in the school library to study for their upcoming final exams."
    claim_1 = "The students are preparing for an examination."
    claim_2 = "The students are on vacation."
    
    try:
        checker = MiniCheckCorrect()
        
        docs = [doc, doc]
        claims = [claim_1, claim_2]
        
        predictions, probabilities = checker.score(docs, claims)
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Document: {doc}")
        print(f"Claim 1: '{claim_1}' -> Prediction: {predictions[0]}, Probability: {probabilities[0]:.3f}")
        print(f"Claim 2: '{claim_2}' -> Prediction: {predictions[1]}, Probability: {probabilities[1]:.3f}")
        
        # Verifica logica dei risultati
        if predictions[0] == 1 and predictions[1] == 0:
            print("\n✅ Perfect! Results are logically correct!")
            print("✅ MiniCheck setup successful!")
            return True
        else:
            print(f"\n⚠️ Results: Claim 1={predictions[0]}, Claim 2={predictions[1]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_sample():
    """Test su un campione del benchmark LLM-AggreFact"""
    print("\n=== Testing on LLM-AggreFact Sample ===")
    
    try:
        # Carica un piccolo campione del dataset
        print("Loading LLM-AggreFact dataset...")
        dataset = load_dataset("lytang/LLM-AggreFact")
        df = pd.DataFrame(dataset['test'])
        
        print(f"Total dataset size: {len(df)}")
        print(f"Datasets included: {df.dataset.unique()}")
        
        # Prendi un campione piccolo per test
        sample_size = 50
        df_sample = df.sample(n=sample_size, random_state=42)
        
        checker = MiniCheckCorrect()
        
        docs = df_sample.doc.values
        claims = df_sample.claim.values
        true_labels = df_sample.label.values
        
        print(f"Testing on {sample_size} examples...")
        predictions, probabilities = checker.score(docs, claims)
        
        # Calcola accuracy
        accuracy = balanced_accuracy_score(true_labels, predictions)
        
        print(f"\n=== BENCHMARK RESULTS ===")
        print(f"Balanced Accuracy: {accuracy:.3f}")
        print(f"Examples processed: {len(predictions)}")
        
        # Mostra alcuni esempi
        print(f"\n=== Sample Results ===")
        for i in range(min(5, len(predictions))):
            status = "✅" if predictions[i] == true_labels[i] else "❌"
            print(f"{status} Doc: {docs[i][:50]}...")
            print(f"    Claim: {claims[i][:50]}...")
            print(f"    True: {true_labels[i]}, Pred: {predictions[i]} (prob: {probabilities[i]:.3f})")
            print()
        
        return accuracy > 0.5  # Threshold ragionevole
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        return False

if __name__ == "__main__":
    # Test base
    success = test_complete()
    
    if success:
        # Test su benchmark
        test_benchmark_sample()
        
        print("\n🎉 MiniCheck è pronto per la replica del paper!")
        print("\nProssimi passi:")
        print("1. Configurare Groq API per confronti")
        print("2. Implementare pipeline di valutazione completa")
        print("3. Processare dataset completo LLM-AggreFact")
    else:
        print("\n❌ Risolvi i problemi prima di procedere")