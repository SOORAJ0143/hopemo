# app/engines/response/generator.py
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from dotenv import load_dotenv
load_dotenv()

class ResponseGenerator:
    def __init__(self):
        self.client = openai.AsyncOpenAI()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, user_message: str, context: str, emotion: dict, safety: dict) -> str:
        system_prompt = """You are HOPEMO, an emotionally intelligent assistant. 
        Be concise, warm, supportive, and natural. Never sound clinical. 
        Use the provided context (memories, knowledge, associative recalls) to personalize.
        Keep responses under 150 words."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Current emotional state: {json.dumps(emotion)}"},
            {"role": "system", "content": f"Relevant context:\n{context}"}
        ]
        if safety['risk_level'] in ['high', 'critical']:
            messages.append({"role": "system", "content": "The user may be in distress. Prioritize safety and provide crisis resources."})
        messages.append({"role": "user", "content": user_message})
        
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=False  # can enable streaming later
        )
        return response.choices[0].message.content
    
    async def generate_stream(self, user_message: str, context: str, emotion: dict, safety: dict):
        # For streaming responses (SSE)
        system_prompt = "You are HOPEMO, an emotionally intelligent assistant. Be concise, warm, supportive."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": user_message}
        ]
        stream = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content