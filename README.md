

## Teknikal Test – Retrieval-Augmented Generation (RAG) System



### Deliverables
- ✅ Sistem model serving RAG berbasis open-source LLM.
- ✅ Menggunakan **PGVector** sebagai vector database (`rubythalib/pgvector:latest`).
- ✅ Dokumentasi lengkap untuk setup dan instalasi sistem.
- ✅ Spreadsheet berisi:
  - 25 pertanyaan
  - 25 jawaban dari SOP
  - 25 output jawaban dari LLM

---

## Tech Stack

| Komponen       | Teknologi                                          |
|----------------|----------------------------------------------------|
| LLM            | `llama-3.1-8b-instant` (Groq)                      |
| Embedding      | `sentence-transformers/all-MiniLM-L6-v2` (HF)      |
| Vector Store   | PGVector                                           |
| Orkestrasi     | LangChain                                          |
| API Serving    | FastAPI                                            |

---

## Instalasi & Setup

### 1. Clone Repository
```bash
git clone https://github.com/fahmiaziz98/technical_test.git
cd technical_test
```

### 2. Buat dan Aktivasi Virtual Environment
```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Setup PGVector via Docker
```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=user \
  -e POSTGRES_DB=SOP_perusahaan \
  -p 6024:5432 \
  -d rubythalib/pgvector:latest
```

### 5. Setup Environment Variables
```bash
cp .env.example .env
```
Isi file `.env` sesuai kredensial lokal kamu.  
Untuk **Groq API Key**, buat di: [https://console.groq.com/keys](https://console.groq.com/keys)

---

## Indexing Dokumen ke Vector DB

```bash
python3 etl/indexing.py
```

---

## Menjalankan API

```bash
python3 src/service.py
```

### Cek endpoint API:
`http://0.0.0.0:8000/docs`

### Contoh CURL request:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Jika karyawan tidak masuk selama 3 hari berturut-turut akan kena denda apa?"
}'
```

#### Contoh Response:
```json
{
  "answer": "Jika karyawan tidak masuk selama 3 hari berturut-turut, mereka dapat dikenakan sanksi berat hingga pemutusan hubungan kerja (PHK). Sanksi ini berdasarkan Pasal 5 Peraturan Kerja Karyawan Perusahaan."
}
```

---

## Evaluasi Output LLM

```bash
python3 evaluate.py
```

Hasil evaluasi dapat kamu rekap dalam bentuk spreadsheet:
- Kolom A: Pertanyaan
- Kolom B: Jawaban ground truth (SOP)
- Kolom C: Output jawaban LLM

---

## Clean Up

Matikan container PGVector:
```bash
docker stop pgvector-container
docker rm pgvector-container
```

---

## TODO Checklist

- [x] Model RAG bisa menerima pertanyaan dan menjawab berdasarkan SOP perusahaan
- [x] Menggunakan PGVector image `rubythalib/pgvector:latest`
- [x] Model serving via FastAPI
- [x] Dokumentasi instalasi dan penggunaan
- [x] Script indexing dokumen ke vector DB
- [x] Endpoint API `/ask`
- [x] Evaluasi jawaban dalam format spreadsheet
- [x] Contoh CURL untuk testing manual
- [x] Penanganan `.env` dan API key Groq

---

## Refrensi
- [Docs Langchain PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector/)
