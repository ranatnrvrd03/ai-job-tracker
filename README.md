# AI Job Tracker

RAG (Retrieval-Augmented Generation) mimarisi kullanılarak geliştirilmiş, yerel LLM entegrasyonlu akıllı iş ilanı ve yetenek takip sistemi. Bu proje, yüklenen iş ilanları ile kullanıcı CV'sini vektörel olarak kıyaslar ve yapay zeka desteğiyle eksik beceri analizi raporu sunar.

## Teknolojiler ve Araçlar

- **Backend:** FastAPI, Pydantic, Uvicorn
- **Veri Yönetimi & Vektör Veri Tabanı:** PostgreSQL, pgvector
- **Yapay Zeka & RAG:** Sentence-Transformers (all-MiniLM-L6-v2), HuggingFace Transformers, Accelerate
- **Yerel LLM:** Qwen2.5-0.5B-Instruct
- **Geliştirme Ortamı:** Linux (Ubuntu via WSL), Python venv

## Mimari ve Çalışma Mantığı

1. **Vektörleştirme (Embedding):** Sisteme eklenen iş ilanlarının açıklama metinleri 384 boyutlu sayısal vektörlere dönüştürülür.
2. **Vektör Veri Yönetimi:** Üretilen vektörler PostgreSQL üzerinde `pgvector` eklentisi kullanılarak saklanır.
3. **Semantik Arama (Retrieval):** Kullanıcı CV metnini gönderdiğinde, kosinüs benzerliği (<=>) yöntemiyle CV'ye en uygun iş ilanları veri tabanından sorgulanır.
4. **Analiz Raporu (Generation):** En uygun iş ilanı ve CV metni yerel olarak çalışan Qwen dil modeline beslenerek adayın eksik teknik becerileri tamamen Türkçe olarak analiz edilir.

## Kurulum ve Çalıştırma

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt