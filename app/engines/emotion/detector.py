# app/engines/emotion/detector.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os

class EmotionDetector:
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token = os.getenv("HF_TOKEN") 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.emotion_labels = [
            self.model.config.id2label[i]
            for i in sorted(self.model.config.id2label)
        ]
    
    async def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        scores = {
            label: float(probs[i])
            for i, label in enumerate(self.emotion_labels[:len(probs)])
        }
        return scores

# For real-time scoring, we also need to map to our 9 dimensions
class EmotionalAnalyzer:
    def __init__(self):
        self.detector = EmotionDetector()
        # Mapping from RoBERTa emotions to our dimensions
        self.mapping = {
            "anxiety": ["nervousness", "fear"],
            "stress": ["annoyance", "confusion", "disapproval"],
            "sadness": ["sadness", "grief", "remorse"],
            "loneliness": ["loneliness", "isolation"],  # approximate
            "burnout": ["exhaustion", "tiredness"],
            "hope": ["optimism", "joy"],
            "motivation": ["excitement", "curiosity", "admiration"],
            "emotional_exhaustion": ["exhaustion", "burnout"],
            "urgency": ["fear", "panic"]
        }
    
    async def analyze(self, text: str) -> dict:
        scores = await self.detector.predict(text)
        result = {}
        for dimension, related in self.mapping.items():
            value = sum(scores.get(e, 0) for e in related) / max(len(related), 1)
            result[dimension] = min(value, 1.0)
        result["primary_emotion"] = max(result.items(), key=lambda x: x[1])[0]
        result["intensity"] = max(value for value in result.values() if isinstance(value, (int, float)))
        return result
