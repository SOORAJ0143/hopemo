# app/engines/fine_tuning/trainer.py
import json
import openai
from datasets import Dataset
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

class FineTuner:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def prepare_data(self, conversations: List[Dict]) -> str:
        """conversations: list of {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}"""
        lines = []
        for conv in conversations:
            lines.append(json.dumps(conv))
        file_path = "fine_tuning_data.jsonl"
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return file_path
    
    def upload_and_finetune(self, file_path: str, base_model="gpt-3.5-turbo", n_epochs=3):
        # Upload file
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="fine-tune")
        file_id = response.id
        # Create fine-tuning job
        job = self.client.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model,
            hyperparameters={"n_epochs": n_epochs}
        )
        return job.id
    
    def get_status(self, job_id: str):
        return self.client.fine_tuning.jobs.retrieve(job_id)