from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from services.ai_service import AIService
from db.models import init_db
from db.repository import insert_job, get_similar_jobs

# Dokümantasyon gruplarını ve açıklamalarını tanımlıyoruz
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
    docs_url=None,  # Varsayılan standart beyaz sayfayı devre dışı bırakıyoruz
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

# Özelleştirilmiş, şık temalı dokümantasyon endpoint'imiz
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        # CDN üzerinden harika bir Material/Dark teması giydiriyoruz
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-themes@3.0.1/themes/3.x/theme-material.css"
    )

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