from datasets import load_dataset
import pandas as pd

print("Downloading LLM-AggreFact using HuggingFace example code...")

try:
    # Usa esattamente il codice mostrato su HuggingFace
    ds = load_dataset("lytang/LLM-AggreFact")
    
    print("SUCCESS! Dataset downloaded")
    print(f"Available splits: {list(ds.keys())}")
    
    # Esplora ogni split
    for split_name, split_data in ds.items():
        print(f"\n{split_name} split:")
        print(f"  Size: {len(split_data):,} examples")
        print(f"  Features: {list(split_data.features.keys())}")
        
        # Mostra primo esempio
        if len(split_data) > 0:
            example = split_data[0]
            print(f"  Example:")
            for key, value in example.items():
                if isinstance(value, str):
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
    
    # Salva il test split per valutazione
    test_data = ds['test']
    print(f"\nConverting test split to DataFrame...")
    
    test_df = pd.DataFrame(test_data)
    print(f"Test dataframe shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    
    # Salva tutto il test set
    test_df.to_csv('llm_aggrefact_test_full.csv', index=False)
    print(f"Saved full test set: {len(test_df)} examples")
    
    # Salva anche un campione per test veloci
    sample_df = test_df.sample(n=min(500, len(test_df)), random_state=42)
    sample_df.to_csv('llm_aggrefact_test_sample.csv', index=False)
    print(f"Saved sample: {len(sample_df)} examples")
    
    # Mostra statistiche
    if 'label' in test_df.columns:
        print(f"\nLabel distribution:")
        print(test_df['label'].value_counts())
    
    if 'dataset' in test_df.columns:
        print(f"\nSource datasets:")
        print(test_df['dataset'].value_counts())
    
    print(f"\nFiles ready for evaluation:")
    print(f"  llm_aggrefact_test_full.csv ({len(test_df)} examples)")
    print(f"  llm_aggrefact_test_sample.csv ({len(sample_df)} examples)")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("Check your HuggingFace authentication status")