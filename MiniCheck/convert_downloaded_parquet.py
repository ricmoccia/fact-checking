import pandas as pd
import os

def convert_parquet_to_csv():
    """
    Converte i file .parquet scaricati da HuggingFace in CSV
    """
    print("Looking for downloaded .parquet files...")
    
    # Cerca file .parquet nella directory
    parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
    
    if not parquet_files:
        print("No .parquet files found.")
        print("Download manually from: https://huggingface.co/datasets/lytang/LLM-AggreFact")
        print("Save the files in this directory, then run this script again.")
        return
    
    print(f"Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  {f}")
    
    # Converti ogni file
    for parquet_file in parquet_files:
        print(f"\nConverting {parquet_file}...")
        
        try:
            # Leggi parquet
            df = pd.read_parquet(parquet_file)
            
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Salva come CSV
            csv_filename = parquet_file.replace('.parquet', '.csv')
            df.to_csv(csv_filename, index=False)
            
            print(f"  Saved as: {csv_filename}")
            
            # Mostra statistiche
            if 'label' in df.columns:
                print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
            
            # Salva anche un campione per test rapidi
            if len(df) > 100:
                sample_df = df.sample(n=min(500, len(df)), random_state=42)
                sample_filename = csv_filename.replace('.csv', '_sample.csv')
                sample_df.to_csv(sample_filename, index=False)
                print(f"  Sample saved as: {sample_filename}")
        
        except Exception as e:
            print(f"  Error converting {parquet_file}: {e}")
    
    print(f"\nConversion complete!")

if __name__ == "__main__":
    convert_parquet_to_csv()