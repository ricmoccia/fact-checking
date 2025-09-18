from datasets import load_dataset
import pandas as pd
from huggingface_hub import login
import os

def download_llm_aggrefact():
    """
    Scarica il dataset LLM-AggreFact completo
    """
    print("Downloading LLM-AggreFact dataset...")
    
    try:
        # Carica il dataset (ora che hai accesso)
        dataset = load_dataset("lytang/LLM-AggreFact")
        
        print("Dataset downloaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Esplora la struttura
        for split_name, split_data in dataset.items():
            print(f"\n{split_name} split:")
            print(f"  Size: {len(split_data)}")
            print(f"  Features: {split_data.features}")
            
            # Mostra primo esempio
            if len(split_data) > 0:
                example = split_data[0]
                print(f"  Example fields: {list(example.keys())}")
                
                # Mostra contenuto delle prime righe
                for key, value in example.items():
                    if isinstance(value, str):
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you:")
        print("1. Have requested access on HuggingFace")
        print("2. Have been approved by the authors")
        print("3. Have run 'huggingface-cli login'")
        return None

def prepare_for_minicheck(dataset):
    """
    Prepara il dataset nel formato corretto per MiniCheck
    """
    print("\nPreparing dataset for MiniCheck format...")
    
    # Usa il test split
    test_data = dataset['test']
    df = pd.DataFrame(test_data)
    
    print(f"Test set size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Estrai le colonne necessarie
    docs = df['doc'].tolist()
    claims = df['claim'].tolist()
    labels = df['label'].tolist()
    
    # Controlla distribuzione labels
    print(f"Label distribution:")
    print(f"  Positive (1): {sum(labels)}")
    print(f"  Negative (0): {len(labels) - sum(labels)}")
    
    # Mostra alcuni esempi
    print(f"\nFirst 3 examples:")
    for i in range(min(3, len(docs))):
        print(f"Example {i+1}:")
        print(f"  Doc: {docs[i][:150]}...")
        print(f"  Claim: {claims[i][:100]}...")
        print(f"  Label: {labels[i]}")
        print()
    
    return docs, claims, labels

def save_dataset_sample(docs, claims, labels, sample_size=1000):
    """
    Salva un campione del dataset per test veloci
    """
    print(f"\nSaving sample of {sample_size} examples...")
    
    # Prendi campione stratificato
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    if len(docs) > sample_size:
        docs_sample, _, claims_sample, _, labels_sample, _ = train_test_split(
            docs, claims, labels, 
            train_size=sample_size, 
            stratify=labels, 
            random_state=42
        )
    else:
        docs_sample, claims_sample, labels_sample = docs, claims, labels
    
    # Salva in CSV per uso futuro
    sample_df = pd.DataFrame({
        'doc': docs_sample,
        'claim': claims_sample,
        'label': labels_sample
    })
    
    sample_df.to_csv('llm_aggrefact_sample.csv', index=False)
    print(f"Sample saved to 'llm_aggrefact_sample.csv'")
    print(f"Sample size: {len(sample_df)}")
    
    return docs_sample, claims_sample, labels_sample

if __name__ == "__main__":
    # Scarica dataset
    dataset = download_llm_aggrefact()
    
    if dataset:
        # Prepara per MiniCheck
        docs, claims, labels = prepare_for_minicheck(dataset)
        
        # Salva campione per test
        docs_sample, claims_sample, labels_sample = save_dataset_sample(
            docs, claims, labels, sample_size=500
        )
        
        print(f"\n✅ Dataset ready!")
        print(f"Full dataset: {len(docs)} examples")
        print(f"Sample dataset: {len(docs_sample)} examples")
        print(f"Now you can run your comparison pipeline on the original data!")
        
    else:
        print("❌ Dataset download failed")