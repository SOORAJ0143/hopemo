# scripts/fine_tune.py
from dotenv import load_dotenv
load_dotenv()

import json
import os
from app.engines.fine_tuning.trainer import FineTuner

# 1. Define the path to your JSON file (place it in the project root)
DATA_PATH = "data\my_data.json"  # <-- CHANGE THIS TO YOUR FILE PATH

# 2. Load the training data from the JSON file
with open(DATA_PATH, "r", encoding="utf-8") as f:
    training_data = json.load(f)  # Must be a list of {"messages": [...]}

# 3. Initialize the tuner
# If using the NEW trainer.py (auto-reads .env):
tuner = FineTuner(os.getenv("OPENAI_API_KEY")) 

# If using the OLD trainer.py (requires API key explicitly), use this instead:
# tuner = FineTuner(os.getenv("OPENAI_API_KEY"))

# 4. Prepare and start fine-tuning
file_path = tuner.prepare_data(training_data)
job_id = tuner.upload_and_finetune(file_path)
print(f"Fine-tuning job started: {job_id}")