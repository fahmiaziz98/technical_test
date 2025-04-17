import pandas as pd
import requests
import time

# Baca file Excel
df = pd.read_excel('evaluasi_data.xlsx')
api_url = "http://localhost:8000/ask"


for index, row in df.iterrows():
    try:

        question = row['pertanyaan']
        print(f"Processing pertanyaan {index + 1}: {question}...")
        
        response = requests.post(
            api_url,
            json={"text": question}
        )
   
        if response.status_code == 200:
            df.at[index, 'output_ai'] = response.json()['answer']
            df.to_excel('evaluasi_data.xlsx', index=False)
        
        time.sleep(3)
        
    except Exception as e:
        print(f"Skip pertanyaan {index + 1}: error")
        continue

print("Selesai!")
