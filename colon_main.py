from fastapi import FastAPI, Request
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load environment variables from .env

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Standard pathology report for comparison
STANDARD_ANSWER = """
Diagnosis: Sessile serrated lesion 
Microscopic Description: Sawtooth serrations and abundant mucin present in epithelium. Basal crypt dilation.  
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

    Provide feedback on accuracy, completeness, and terminology. Suggest corrections. Use an encouraging style.
    """

    # Call OpenAI API to get feedback
    response = client.chat.completions.create(model="gpt-4o-mini",  # Use the model you have access to
    messages=[
        {"role": "system", "content": "You are a pathology expert analyzing histology reports."},
        {"role": "user", "content": prompt}
    ])

    # Extract the relevant part of the response and return it
    feedback = response.choices[0].message.content

    return {"feedback": feedback}
