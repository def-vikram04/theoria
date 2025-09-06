from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from suggestions import bias_suggestions

# Load trained model
model = joblib.load("dummy.pkl")

app = FastAPI(title="Theoria API", description="Cognitive Bias Detector", version="1.0")

# ✅ Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- ROUTES -----------------

@app.get("/")
def read_root():
    return {"message": "Hii 👋, I'm Theoria — your cognitive bias detector is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Theoria backend is healthy ✅"}

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: TextRequest):
    text = request.text
    
    # Predict bias
    bias = model.predict([text])[0]
    
    # Get suggestions
    suggestions = bias_suggestions.get(bias, ["No suggestions available."])
    
    return {
        "input": text,
        "bias_detected": bias,
        "suggestions": suggestions
    }

# ----------------- SERVER ENTRYPOINT -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
