import logging
import json
from fastapi import FastAPI, Request
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()  # Load variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Add CORS middleware to allow all origins (you can specify specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Root endpoint to handle GET requests at '/'
@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

# Standard Answer for Comparison
STANDARD_ANSWER = """
Diagnosis: Squamous cell carcinoma  
Microscopic Description: Invasive nests of atypical squamous cells with keratinization and intercellular bridges.  
Immunohistochemistry: Positive for p40 and CK5/6.  
"""

# POST endpoint for comparing user response to the standard pathology report
@app.post("/compare")
async def compare_text(request: Request):
    try:
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

        # Call OpenAI API to generate a response
        response = openai.Completion.create(
            model="gpt-4",  # or another model you are using
            prompt=prompt,
            max_tokens=150
        )

        # Log the full response object as JSON string to debug
        logger.debug(f"Full OpenAI response: {json.dumps(response, indent=2)}")

        # Ensure you're accessing the response correctly
        if 'choices' in response and len(response['choices']) > 0:
            feedback = response['choices'][0].get('text', 'No feedback available')
        else:
            feedback = 'No feedback provided by the model'

        return {"feedback": feedback}

    except Exception as e:
        logger.error(f"Error: {str(e)}")  # Log any exceptions
        return {"error": str(e)}  # Return the error message in case of failure
