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

class SpamClassificationAPI:
    def __init__(self, model_path: str = 'models/trained_models/spam_classifier'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise RuntimeError("Failed to load spam classification model")

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

            spam_probability = probabilities[0][1].item()
            is_spam = prediction.item() == 1

            return {
                'is_spam': is_spam,
                'spam_probability': spam_probability
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

app = FastAPI(title="Spam Email Classification API")
classifier = SpamClassificationAPI()

@app.post("/predict")
async def predict_spam(request: EmailRequest):
    try:
        result = classifier.predict(request.email_text)
        return result
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)