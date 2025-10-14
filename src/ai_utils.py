# src/ai_utils.py

import os
from openai import OpenAI
from dotenv import load_dotenv
import ollama


# load_dotenv()  # reads .env file
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def generate_llm_summary(prompt, model="gpt-3.5-turbo", max_tokens=50):
#     """
#     Return a concise summary from LLM given a prompt.
#     """
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=max_tokens
#     )
#     return response.choices[0].message.content.strip()


 

def generate_llm_summary(prompt: str) -> str:
    """
    Return a concise summary from the local Qwen3 model via Ollama.
    
    Parameters:
        prompt (str): The text prompt for the LLM
    Returns:
        str: Generated summary text
    """
    response = ollama.generate(
        model="mistral:7b",
        prompt=prompt,
        options={
            'temperature': 0.0
        }
    )
    return response['response']  # Ollama returns the generated text directly
