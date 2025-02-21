from fastapi import FastAPI, Request
import openai
import os

pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

STANDARD_ANSWER = """
Diagnosis: Squamous cell carcinoma  
Microscopic Description: Invasive nests of atypical squamous cells with keratinization and intercellular bridges.  
Immunohistochemistry: Positive for p40 and CK5/6.  
"""

@app.post("/compare")
async def compare_text(request: Request):
    data = await request.json()
    user_response = data["text"]

    prompt = f"""
    Compare the following user response to the standard pathology report.
    
    **User Response:**  
    {user_response}

    **Correct Answer:**  
    {STANDARD_ANSWER}

    Provide feedback on accuracy, completeness, and terminology. Suggest corrections.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a pathology expert analyzing histology reports."},
                  {"role": "user", "content": prompt}]
    )

    return {"feedback": response["choices"][0]["message"]["content"]}

