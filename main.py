from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.ai_service import AIService
from db.models import init_db
from db.repository import insert_job, get_similar_jobs

app = FastAPI(
    title="AI Job Tracker API",
    description="RAG destekli akıllı iş ilanı ve yetenek takip sistemi",
    version="0.1.0"
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

@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "AI Job Tracker API başarıyla çalışıyor!",
        "version": "0.1.0"
    }

@app.post("/api/v1/jobs")
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

@app.post("/api/v1/jobs/recommend")
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