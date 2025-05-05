import os
import logging
from typing import Dict, Any

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailRequest(BaseModel):
    email_text: str

class SpamClassifierAPI:
    def __init__(self, model_path: str = 'models/spam_classifier.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def predict(self, email_text: str) -> Dict[str, Any]:
        try:
            inputs = self.tokenizer(
                email_text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)

            return {
                'is_spam': bool(prediction.item()),
                'spam_probability': probabilities[0][1].item(),
                'confidence': float(probabilities.max())
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

app = FastAPI(title="Spam Classifier", description="Real-time Email Spam Detection")
classifier = SpamClassifierAPI()

@app.post("/predict")
async def predict_spam(request: EmailRequest):
    try:
        result = classifier.predict(request.email_text)
        return result
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        raise HTTPException(status_code=500, detail="Service error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)