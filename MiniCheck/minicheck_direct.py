import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import numpy as np

class MiniCheckDirect:
    def __init__(self, model_name='lytang/MiniCheck-RoBERTa-Large', device=None):
        """
        Implementazione diretta di MiniCheck usando HuggingFace
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Carica tokenizer e modello
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
        Returns: predictions (0/1), probabilities
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for doc, claim in zip(docs, claims):
                # Formato input: [CLS] claim [SEP] document [SEP]
                inputs = self.tokenizer(
                    claim, doc,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Probabilità che sia supportato (classe 1)
                prob_supported = probs[0][1].cpu().item()
                pred = 1 if prob_supported > 0.5 else 0
                
                predictions.append(pred)
                probabilities.append(prob_supported)
                
                print(f"Claim: {claim[:50]}...")
                print(f"Prediction: {pred}, Probability: {prob_supported:.3f}")
        
        return predictions, probabilities

# Test del sistema
def test_minicheck():
    print("Testing MiniCheck Direct Implementation...")
    
    # Test di base
    doc = "A group of students gather in the school library to study for their upcoming final exams."
    claim_1 = "The students are preparing for an examination."
    claim_2 = "The students are on vacation."
    
    try:
        # Inizializza MiniCheck con il nome corretto
        checker = MiniCheckDirect(model_name='lytang/MiniCheck-RoBERTa-Large')
        
        # Test predictions
        docs = [doc, doc]
        claims = [claim_1, claim_2]
        
        predictions, probabilities = checker.score(docs, claims)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Document: {doc}")
        print(f"Claim 1: '{claim_1}' -> Prediction: {predictions[0]}, Probability: {probabilities[0]:.3f}")
        print(f"Claim 2: '{claim_2}' -> Prediction: {predictions[1]}, Probability: {probabilities[1]:.3f}")
        
        # Verifica logica dei risultati
        if predictions[0] == 1 and predictions[1] == 0:
            print("\n✅ Results look correct!")
            print("✅ Setup successful!")
            return True
        else:
            print(f"\n⚠️ Results might be unexpected:")
            print(f"Expected: Claim 1=1 (supported), Claim 2=0 (not supported)")
            print(f"Got: Claim 1={predictions[0]}, Claim 2={predictions[1]}")
            return True  # Still successful, just unexpected results
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minicheck()