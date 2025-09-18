from datasets import load_dataset
from huggingface_hub import login
import os

def try_access_llm_aggrefact():
    """
    Prova ad accedere al dataset LLM-AggreFact
    """
    print("Attempting to access LLM-AggreFact dataset...")
    
    # Metodo 1: Accesso diretto
    try:
        print("Trying direct access...")
        dataset = load_dataset("lytang/LLM-AggreFact")
        print("Success! Dataset loaded directly")
        return dataset
    except Exception as e:
        print(f"Direct access failed: {e}")
    
    # Metodo 2: Con token HuggingFace
    try:
        print("Trying with HuggingFace token...")
        # Dovrai fare login su HuggingFace prima
        # huggingface-cli login
        dataset = load_dataset("lytang/LLM-AggreFact", use_auth_token=True)
        print("Success! Dataset loaded with authentication")
        return dataset
    except Exception as e:
        print(f"Authenticated access failed: {e}")
    
    # Metodo 3: Richiesta accesso
    print("\nDataset is gated. To access:")
    print("1. Go to: https://huggingface.co/datasets/lytang/LLM-AggreFact")
    print("2. Click 'Request access'")
    print("3. Wait for approval from authors")
    print("4. Then run: huggingface-cli login")
    
    return None

def explore_dataset_structure(dataset):
    """
    Esplora la struttura del dataset LLM-AggreFact
    """
    print("\nExploring dataset structure...")
    
    print(f"Dataset splits: {list(dataset.keys())}")
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  Size: {len(split_data)}")
        print(f"  Features: {split_data.features}")
        
        # Mostra esempio
        if len(split_data) > 0:
            example = split_data[0]
            print(f"  Example:")
            for key, value in example.items():
                if isinstance(value, str):
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")

if __name__ == "__main__":
    dataset = try_access_llm_aggrefact()
    
    if dataset:
        explore_dataset_structure(dataset)
    else:
        print("Dataset access failed. See instructions above.")