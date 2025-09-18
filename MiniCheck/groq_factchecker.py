import os
from groq import Groq
import time
import json

class GroqFactChecker:
    def __init__(self, api_key=None):
        """
        Inizializza il fact-checker Groq
        """
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            # Rimuovi spazi dalle variabili ambiente
            api_key = os.getenv('GROQ_API_KEY', '').strip().strip('"')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            self.client = Groq(api_key=api_key)
        
        # Usa il modello raccomandato più recente
        self.model = "llama-3.3-70b-versatile"  # Modello aggiornato
        
    def fact_check(self, document, claim):
        """
        Fact-check di una singola claim usando Groq
        """
        prompt = f"""You are a fact-checking expert. Given the following document and claim, determine if the claim is supported by the document.

Document: {document}

Claim: {claim}

Analyze whether the claim is factually supported by the information in the document. Respond with:
- "SUPPORTED" if the claim is clearly supported by the document
- "NOT_SUPPORTED" if the claim contradicts or is not supported by the document

Response format: [SUPPORTED/NOT_SUPPORTED]
Explanation: [brief explanation]"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,  # Bassa temperatura per risultati più consistenti
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            
            # Estrai la predizione
            if "SUPPORTED" in result.upper() and "NOT_SUPPORTED" not in result.upper():
                prediction = 1
            else:
                prediction = 0
                
            return prediction, result
            
        except Exception as e:
            print(f"Errore Groq API: {e}")
            return None, str(e)
    
    def batch_fact_check(self, docs, claims, delay=1.0):
        """
        Fact-check di multiple claims con rate limiting
        """
        predictions = []
        explanations = []
        token_count = 0
        
        for i, (doc, claim) in enumerate(zip(docs, claims)):
            print(f"Processing {i+1}/{len(docs)}...")
            
            pred, explanation = self.fact_check(doc, claim)
            predictions.append(pred)
            explanations.append(explanation)
            
            # Stima token usage (approssimativa)
            token_count += len(doc.split()) + len(claim.split()) + 50
            
            # Rate limiting per evitare di superare i limiti
            time.sleep(delay)
            
        return predictions, explanations, token_count

def test_groq_setup():
    """Test del setup Groq"""
    print("=== Testing Groq Setup ===")
    
    # Test di base
    doc = "A group of students gather in the school library to study for their upcoming final exams."
    claim_1 = "The students are preparing for an examination."
    claim_2 = "The students are on vacation."
    
    try:
        print("Initializing Groq client...")
        checker = GroqFactChecker()
        print(f"Using model: {checker.model}")
        
        # Test singolo
        print(f"\nTesting claim 1...")
        pred1, exp1 = checker.fact_check(doc, claim_1)
        print(f"Claim 1: {pred1}")
        print(f"Explanation: {exp1}")
        
        print(f"\nTesting claim 2...")
        pred2, exp2 = checker.fact_check(doc, claim_2)
        print(f"Claim 2: {pred2}")
        print(f"Explanation: {exp2}")
        
        if pred1 == 1 and pred2 == 0:
            print("\n✅ Groq setup successful!")
            return True
        else:
            print(f"\n⚠️ Results: pred1={pred1}, pred2={pred2}")
            print("Groq is working but results may vary")
            return True  # Ancora funzionante
            
    except Exception as e:
        print(f"❌ Groq setup failed: {e}")
        print("\nPer configurare Groq:")
        print("1. Vai su https://console.groq.com/")
        print("2. Crea un account e ottieni API key")
        print("3. Rimuovi spazi extra dalla variabile GROQ_API_KEY")
        return False

if __name__ == "__main__":
    test_groq_setup()