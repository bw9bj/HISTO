from fastapi import FastAPI, Request
from openai import OpenAI
import os

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load environment variables from .env

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary of standard answers keyed by module ID
STANDARD_ANSWERS = {
    "moduleLN": """
Diagnosis: Hodgkin lymphoma, nodular sclerosis type  
Microscopic Description: Dense nodules surrounded by fibrous bands. Eosinophilic infiltrate and binucleate Reed-Sternberg cells with prominent nucleoli  
IHC: Stains positive for CD30, CD15, and PAX5.  
""",
    "moduleColon": """
Diagnosis: Sessile serrated lesion  
Microscopic Description: Sawtooth serrations in the epithelium, abundant mucin, and basal crypt dilation.  
""",
    # Add more modules as needed
}

@app.post("/compare")
async def compare_text(request: Request):
    data = await request.json()
    user_response = data.get("text")
    module_id = data.get("module_id")

    if not module_id or module_id not in STANDARD_ANSWERS:
        return {"feedback": "Invalid or missing module ID."}

    standard_answer = STANDARD_ANSWERS[module_id]

    prompt = f"""
Compare the following user response to the standard pathology report.

**User Response:**  
{user_response}

**Correct Answer:**  
{standard_answer}

Provide feedback on accuracy, completeness, and terminology using an encouraging style. Suggest corrections in an encouraging style.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a pathology expert analyzing histology reports."},
            {"role": "user", "content": prompt}
        ]
    )

    feedback = response.choices[0].message.content
    return {"feedback": feedback}
