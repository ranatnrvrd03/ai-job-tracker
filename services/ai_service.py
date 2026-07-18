from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

class AIService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

    def get_embedding(self, text: str):
        if not text.strip():
            return []
        
        embedding = self.model.encode(text)
        return embedding.tolist()

    def calculate_similarity(self, embedding1, embedding2):
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))

    def analyze_missing_skills(self, cv_text: str, job_text: str):
        messages = [
            {"role": "system", "content": "Sen bir kariyer danışmanısın. İş ilanı ve CV'yi kıyaslayıp eksik olan teknik becerileri Türkçe listele."},
            {"role": "user", "content": f"CV:\n{cv_text}\n\nİş İlanı:\n{job_text}"}
        ]
        try:
            outputs = self.llm(messages, max_new_tokens=256, temperature=0.3)
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"Yerel LLM Analiz Hatası: {str(e)}"