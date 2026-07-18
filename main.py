from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from services.ai_service import AIService
from db.models import init_db
from db.repository import insert_job, get_similar_jobs

tags_metadata = [
    {
        "name": "Sistem Kontrolleri",
        "description": "API sağlık durumu ve temel sistem kontrolleri.",
    },
    {
        "name": "İş İlanı Yönetimi",
        "description": "Sisteme yeni iş ilanları ekleme ve vektör tabanlı saklama operasyonları.",
    },
    {
        "name": "Yapay Zeka & RAG",
        "description": "CV metni analizi, semantik arama ve yerel LLM ile eksik yetenek raporu üretimi.",
    },
]

app = FastAPI(
    title="AI Job Tracker API",
    description="RAG destekli akıllı iş ilanı ve yetenek takip sistemi",
    version="0.1.0",
    openapi_tags=tags_metadata,
    docs_url=None,
    redoc_url=None
)

@app.on_event("startup")
def on_startup():
    init_db()

ai_service = AIService()

class JobCreate(BaseModel):
    title: str
    company: str
    description: str

class CVText(BaseModel):
    text: str

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    # Orijinal ve stabil Swagger assetlerini yükliyoruz
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
    )
    
    # Tamamen kontrollü, yumuşak geçişli ve geniş boşluklu modern tasarım kodlarımız
    custom_css = """
    <style>
        body {
            background-color: #f8fafc !important;
            font-family: 'Inter', sans-serif !important;
        }
        .swagger-ui {
            max-width: 1100px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        /* Başlık Alanı Düzenlemesi */
        .swagger-ui .info {
            margin: 20px 0 40px 0 !important;
        }
        .swagger-ui .info .title {
            font-size: 32px !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        /* Kategori Başlıkları */
        .swagger-ui .opblock-tag {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #1e293b !important;
            border-bottom: 1px solid #e2e8f0 !important;
            padding-bottom: 10px !important;
            margin-bottom: 20px !important;
        }
        /* Kartlar / Butonlar Arası Net Nefes Boşluğu */
        .swagger-ui .opblock {
            margin: 0 0 20px 0 !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            background-color: #ffffff !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            transition: all 0.25s ease !important; /* İstenen yumuşak geçiş efekti */
        }
        /* Hover Durumunda Yumuşak Yükselme Efekti */
        .swagger-ui .opblock:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        }
        /* Endpoint Satır İçi İç Boşlukları (Padding) */
        .swagger-ui .opblock .opblock-summary {
            padding: 14px 20px !important;
        }
        /* HTTP Metot Butonları (GET/POST) */
        .swagger-ui .opblock .opblock-summary-method {
            border-radius: 8px !important;
            font-weight: 700 !important;
            min-width: 85px !important;
            text-align: center !important;
            padding: 6px 12px !important;
        }
        /* Şemalar Bölümü Kart Tasarımı */
        .swagger-ui .model-box {
            border-radius: 10px !important;
            padding: 15px !important;
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
        }
    </style>
    """
    
    # Yazdığımız CSS stilini HTML kodunun içerisine enjekte ediyoruz
    html_content = html.body.decode("utf-8").replace("</head>", f"{custom_css}</head>")
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/", tags=["Sistem Kontrolleri"])
def read_root():
    return {
        "status": "success",
        "message": "AI Job Tracker API başarıyla çalışıyor!",
        "version": "0.1.0"
    }

@app.post("/api/v1/jobs", tags=["İş İlanı Yönetimi"])
def add_job(payload: JobCreate):
    try:
        vector = ai_service.get_embedding(payload.description)
        job_id = insert_job(payload.title, payload.company, payload.description, vector)
        return {
            "status": "success",
            "job_id": job_id,
            "message": "İş ilanı ve vektörü başarıyla kaydedildi."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/jobs/recommend", tags=["Yapay Zeka & RAG"])
def recommend_jobs(payload: CVText):
    try:
        cv_vector = ai_service.get_embedding(payload.text)
        similar_jobs = get_similar_jobs(cv_vector, limit=1)
        
        if not similar_jobs:
            return {
                "status": "success",
                "message": "Veritabanında uygun iş ilanı bulunamadı.",
                "recommendations": []
            }
            
        best_job = similar_jobs[0]
        analysis = ai_service.analyze_missing_skills(payload.text, best_job['description'])
        
        return {
            "status": "success",
            "matched_job": {
                "title": best_job['title'],
                "company": best_job['company'],
                "similarity": best_job['similarity']
            },
            "ai_analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))