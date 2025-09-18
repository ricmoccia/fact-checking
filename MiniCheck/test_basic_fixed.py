from minicheck.minicheck import MiniCheck
import os

print("Testing MiniCheck setup...")

# Test rapido
doc = "A group of students gather in the school library to study for their upcoming final exams."
claim_1 = "The students are preparing for an examination."
claim_2 = "The students are on vacation."

print(f"Document: {doc}")
print(f"Claim 1: {claim_1}")
print(f"Claim 2: {claim_2}")

try:
    # Usiamo i modelli più piccoli che funzionano su Windows
    # Proviamo prima roberta-large (non richiede vLLM)
    print("\nLoading MiniCheck model (roberta-large)...")
    scorer = MiniCheck(model_name='roberta-large', cache_dir='./ckpts')
    
    print("Running predictions...")
    pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])
    
    print(f"\nResults:")
    print(f"Claim 1 - Prediction: {pred_label[0]}, Probability: {raw_prob[0]:.3f}")
    print(f"Claim 2 - Prediction: {pred_label[1]}, Probability: {raw_prob[1]:.3f}")
    print("\n✅ Setup successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative approach...")
    
    try:
        # Proviamo DeBERTa se RoBERTa non funziona
        print("Loading MiniCheck model (deberta-v3-large)...")
        scorer = MiniCheck(model_name='deberta-v3-large', cache_dir='./ckpts')
        
        pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])
        
        print(f"\nResults:")
        print(f"Claim 1 - Prediction: {pred_label[0]}, Probability: {raw_prob[0]:.3f}")
        print(f"Claim 2 - Prediction: {pred_label[1]}, Probability: {raw_prob[1]:.3f}")
        print("\n✅ Setup successful!")
        
    except Exception as e2:
        print(f"❌ Error with DeBERTa: {e2}")