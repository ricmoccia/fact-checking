import pandas as pd
from datasets import load_dataset
import numpy as np

class FactCheckDatasetLoader:
    def __init__(self):
        self.available_datasets = {
            'fever': 'fever',
            'scifact': 'allenai/scifact',
            'custom': 'custom'
        }
    
    def load_custom_examples(self):
        """
        Dataset custom per test specifici
        """
        custom_data = [
            # Esempi chiari SUPPORTATI
            ("The restaurant opens at 8 AM and serves breakfast until 11 AM.", 
             "The restaurant serves breakfast.", 1),
            ("John graduated from Harvard University in 2020 with a degree in Computer Science.", 
             "John has a computer science degree.", 1),
            ("The meeting is scheduled for Tuesday at 3 PM in room 205.", 
             "There is a meeting on Tuesday.", 1),
            
            # Esempi chiari NON SUPPORTATI
            ("The store is closed on Sundays and holidays.", 
             "The store is open every day of the week.", 0),
            ("The temperature today reached a maximum of 25 degrees Celsius.", 
             "Today was freezing cold with sub-zero temperatures.", 0),
            ("The book has 300 pages and was published in 2019.", 
             "This book was published in the 1990s.", 0),
            
            # Esempi ambigui/difficili
            ("The company reported revenue growth of 15% compared to last quarter.", 
             "The company's financial performance is improving.", 1),
            ("The study included 100 participants aged 18-65.", 
             "The research focused on elderly populations.", 0),
        ]
        
        docs = [item[0] for item in custom_data]
        claims = [item[1] for item in custom_data]
        labels = [item[2] for item in custom_data]
        
        print(f"Loaded {len(docs)} custom examples")
        return docs, claims, labels, "Custom Examples"
    
    def load_fever_sample(self, sample_size=100):
        """
        Carica un campione dal dataset FEVER
        """
        print(f"Loading FEVER dataset (sample: {sample_size})...")
        
        try:
            # Carica FEVER con trust_remote_code
            dataset = load_dataset("fever", "v1.0", split="labelled_dev", trust_remote_code=True)
            df = pd.DataFrame(dataset)
            
            # Filtra solo SUPPORTS e REFUTES
            df_filtered = df[df['label'].isin(['SUPPORTS', 'REFUTES'])].copy()
            
            # Converti labels
            df_filtered['binary_label'] = df_filtered['label'].map({
                'SUPPORTS': 1, 
                'REFUTES': 0
            })
            
            # Prendi campione
            if sample_size < len(df_filtered):
                df_sample = df_filtered.sample(n=sample_size, random_state=42)
            else:
                df_sample = df_filtered
            
            # Estrai docs dalle evidenze
            docs = []
            claims = df_sample['claim'].tolist()
            labels = df_sample['binary_label'].tolist()
            
            for _, row in df_sample.iterrows():
                # Combina le evidenze in un singolo documento
                evidence_texts = []
                if row['evidence'] and len(row['evidence']) > 0:
                    for evidence_set in row['evidence']:
                        for evidence in evidence_set:
                            if len(evidence) >= 3:  # [id, title, sentence]
                                evidence_texts.append(evidence[2])
                
                combined_doc = " ".join(evidence_texts) if evidence_texts else "No evidence available."
                docs.append(combined_doc)
            
            print(f"Loaded {len(docs)} examples from FEVER")
            return docs, claims, labels, "FEVER"
            
        except Exception as e:
            print(f"Error loading FEVER: {e}")
            print("Creating FEVER-like synthetic data...")
            return self.create_synthetic_fever(sample_size)
    
    def create_synthetic_fever(self, sample_size=100):
        """
        Crea dati sintetici in stile FEVER
        """
        fever_like_data = [
            # SUPPORTS examples
            ("Barack Obama was the 44th President of the United States and served from 2009 to 2017.", 
             "Barack Obama was the 44th President of the United States.", 1),
            ("The Earth orbits around the Sun and takes approximately 365.25 days to complete one orbit.", 
             "The Earth takes about 365 days to orbit the Sun.", 1),
            ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", 
             "Water boils at 100°C at sea level.", 1),
            ("The Great Wall of China is a series of fortifications built across northern China.", 
             "The Great Wall of China is located in northern China.", 1),
            ("Shakespeare wrote Romeo and Juliet, which was first performed in the 1590s.", 
             "Romeo and Juliet is a play by William Shakespeare.", 1),
            
            # REFUTES examples
            ("The Moon landing occurred on July 20, 1969, when Neil Armstrong first stepped on the lunar surface.", 
             "The Moon landing happened in 1968.", 0),
            ("Albert Einstein developed the theory of relativity in the early 20th century.", 
             "Einstein's theory of relativity was developed in the 19th century.", 0),
            ("The human body has 206 bones in the adult skeleton.", 
             "The human body contains over 300 bones.", 0),
            ("Paris is the capital city of France and has a population of about 2.2 million.", 
             "London is the capital of France.", 0),
            ("The Pacific Ocean is the largest ocean on Earth by both area and depth.", 
             "The Atlantic Ocean is the largest ocean on Earth.", 0),
        ]
        
        # Estendi per raggiungere sample_size
        multiplier = (sample_size // len(fever_like_data)) + 1
        extended_data = (fever_like_data * multiplier)[:sample_size]
        
        docs = [item[0] for item in extended_data]
        claims = [item[1] for item in extended_data]
        labels = [item[2] for item in extended_data]
        
        print(f"Created {len(docs)} synthetic FEVER-like examples")
        return docs, claims, labels, "Synthetic FEVER"
    
    def create_large_mixed_dataset(self, size=100):
        """
        Crea un dataset misto più grande per test robusti
        """
        print(f"Creating large mixed dataset ({size} examples)...")
        
        # Diversi tipi di esempi
        scientific_data = [
            ("Studies show that regular exercise reduces the risk of cardiovascular disease by improving heart function and circulation.", 
             "Exercise is beneficial for heart health.", 1),
            ("Research indicates that smoking tobacco increases the risk of lung cancer and other respiratory diseases.", 
             "Smoking is harmful to lung health.", 1),
            ("Clinical trials demonstrate that this medication reduces blood pressure in 85% of patients within 4 weeks.", 
             "The medication is effective for treating high blood pressure.", 1),
            ("The study found no significant difference between the treatment and control groups after 6 months.", 
             "The treatment was highly effective compared to the control.", 0),
            ("Data shows that the new vaccine has a 95% efficacy rate in preventing the target disease.", 
             "The vaccine is ineffective against the disease.", 0),
        ]
        
        news_data = [
            ("The company announced record quarterly profits of $2.5 billion, exceeding analyst expectations.", 
             "The company performed better than expected this quarter.", 1),
            ("The unemployment rate decreased to 3.8% last month, the lowest in five years.", 
             "Unemployment rates have been decreasing recently.", 1),
            ("The new policy will be implemented starting next year and affects all employees.", 
             "The policy changes take effect immediately.", 0),
            ("The merger between the two companies was completed after regulatory approval.", 
             "The merger was blocked by regulators.", 0),
        ]
        
        general_data = [
            ("The library is open Monday through Friday from 9 AM to 6 PM and closed on weekends.", 
             "The library is open during weekdays.", 1),
            ("The recipe calls for 2 cups of flour, 1 cup of sugar, and 3 eggs.", 
             "This recipe includes flour and sugar.", 1),
            ("The concert starts at 8 PM and tickets cost between $25 and $75.", 
             "Concert tickets are available for under $100.", 1),
            ("The restaurant specializes in Mediterranean cuisine and has been open for 15 years.", 
             "This is a newly opened Asian restaurant.", 0),
            ("The weather forecast predicts sunny skies with temperatures reaching 28°C.", 
             "Rain and thunderstorms are expected today.", 0),
        ]
        
        # Combina tutti i dati
        all_data = scientific_data + news_data + general_data
        
        # Estendi per raggiungere la dimensione desiderata
        multiplier = (size // len(all_data)) + 1
        extended_data = (all_data * multiplier)[:size]
        
        # Mescola per varietà
        np.random.seed(42)
        indices = np.random.permutation(len(extended_data))
        shuffled_data = [extended_data[i] for i in indices]
        
        docs = [item[0] for item in shuffled_data]
        claims = [item[1] for item in shuffled_data]
        labels = [item[2] for item in shuffled_data]
        
        print(f"Created {len(docs)} mixed examples")
        print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
        return docs, claims, labels, "Large Mixed Dataset"

def test_all_datasets():
    """
    Test tutti i dataset disponibili
    """
    print("Testing All Available Datasets...")
    loader = FactCheckDatasetLoader()
    
    results = {}
    
    # Test dataset custom
    print("\n" + "="*50)
    docs, claims, labels, name = loader.load_custom_examples()
    results[name] = {'docs': docs, 'claims': claims, 'labels': labels}
    
    # Test FEVER
    print("\n" + "="*50)
    docs, claims, labels, name = loader.load_fever_sample(30)
    if docs:
        results[name] = {'docs': docs, 'claims': claims, 'labels': labels}
    
    # Test dataset misto grande
    print("\n" + "="*50)
    docs, claims, labels, name = loader.create_large_mixed_dataset(50)
    results[name] = {'docs': docs, 'claims': claims, 'labels': labels}
    
    return results

if __name__ == "__main__":
    results = test_all_datasets()
    
    print("\n" + "="*60)
    print("DATASET SUMMARY:")
    total_examples = 0
    for name, data in results.items():
        size = len(data['docs'])
        pos = sum(data['labels'])
        neg = size - pos
        total_examples += size
        print(f"  {name}: {size} examples ({pos} positive, {neg} negative)")
    
    print(f"\nTOTAL: {total_examples} examples ready for comparison!")
    print("\nNext: Run comparison pipeline on these datasets")