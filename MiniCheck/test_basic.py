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
    # Carica il modello più piccolo
    print("\nLoading MiniCheck model...")
    scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
    
    print("Running predictions...")
    pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])
    
    print(f"\nResults:")
    print(f"Claim 1 - Prediction: {pred_label[0]}, Probability: {raw_prob[0]:.3f}")
    print(f"Claim 2 - Prediction: {pred_label[1]}, Probability: {raw_prob[1]:.3f}")
    print("\n✅ Setup successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")