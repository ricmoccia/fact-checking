import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class MiniCheckDirect:
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
        
        # Debug: Mostra informazioni sul modello
        print(f"Model config: {self.model.config}")
        print(f"Number of labels: {self.model.config.num_labels}")
        
    def score_debug(self, docs, claims):
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i, (doc, claim) in enumerate(zip(docs, claims)):
                print(f"\n--- Processing pair {i+1} ---")
                
                # Proviamo diversi formati di input
                formats_to_try = [
                    # Formato 1: claim [SEP] doc
                    (claim, doc),
                    # Formato 2: doc [SEP] claim  
                    (doc, claim),
                    # Formato 3: solo concatenazione
                    (f"{claim} {doc}", None)
                ]
                
                for fmt_idx, (text1, text2) in enumerate(formats_to_try):
                    print(f"\nTrying format {fmt_idx + 1}:")
                    if text2 is not None:
                        inputs = self.tokenizer(
                            text1, text2,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                            padding=True
                        ).to(self.device)
                        print(f"Input: '{text1[:50]}...' [SEP] '{text2[:50]}...'")
                    else:
                        inputs = self.tokenizer(
                            text1,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                            padding=True
                        ).to(self.device)
                        print(f"Input: '{text1[:100]}...'")
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    
                    print(f"Logits: {logits}")
                    print(f"Probabilities: {probs}")
                    
                    if self.model.config.num_labels == 2:
                        prob_supported = probs[0][1].cpu().item()
                        pred = 1 if prob_supported > 0.5 else 0
                        print(f"Prediction: {pred}, Prob(supported): {prob_supported:.3f}")
                    else:
                        # Se ci sono più di 2 etichette
                        pred_class = torch.argmax(probs, dim=-1).cpu().item()
                        max_prob = torch.max(probs).cpu().item()
                        print(f"Prediction: {pred_class}, Max prob: {max_prob:.3f}")
                
                # Usiamo il primo formato per i risultati finali
                inputs = self.tokenizer(
                    claim, doc,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
                if self.model.config.num_labels == 2:
                    prob_supported = probs[0][1].cpu().item()
                    pred = 1 if prob_supported > 0.5 else 0
                else:
                    pred = torch.argmax(probs, dim=-1).cpu().item()
                    prob_supported = torch.max(probs).cpu().item()
                
                predictions.append(pred)
                probabilities.append(prob_supported)
        
        return predictions, probabilities

# Test con debug
def test_debug():
    print("=== MiniCheck Debug Test ===")
    
    doc = "A group of students gather in the school library to study for their upcoming final exams."
    claim_1 = "The students are preparing for an examination."
    claim_2 = "The students are on vacation."
    
    try:
        checker = MiniCheckDirect()
        
        docs = [doc, doc]
        claims = [claim_1, claim_2]
        
        predictions, probabilities = checker.score_debug(docs, claims)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Claim 1: '{claim_1}' -> Prediction: {predictions[0]}, Probability: {probabilities[0]:.3f}")
        print(f"Claim 2: '{claim_2}' -> Prediction: {predictions[1]}, Probability: {probabilities[1]:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()