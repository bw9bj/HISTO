import logging
import json
from fastapi import FastAPI, Request
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Set up logging to capture all responses
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables (make sure your OPENAI_API_KEY is in your .env file)
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins (you can restrict this to specific domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Root endpoint to check if the server is running
@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

# Standard Answer for Comparison (replace this with the actual standard answer)
STANDARD_ANSWER = """
Diagnosis: Squamous cell carcinoma  
Microscopic Description: Invasive nests of atypical squamous cells with keratinization and intercellular bridges.  
Immunohistochemistry: Positive for p40 and CK5/6.  
"""

# POST endpoint to compare user input against the standard pathology report
@app.post("/compare")
async def compare_text(request: Request):
    try:
        # Parse user input from the request
        data = await request.json()
        user_response = data["text"]

        # Construct the prompt for OpenAI
        prompt = f"""
        Compare the following user response to the standard pathology report.

        **User Response:**  
        {user_response}

        **Correct Answer:**  
        {STANDARD_ANSWER}

        Provide feedback on accuracy, completeness, and terminology. Suggest corrections.
        """

        # Call OpenAI's new ChatCompletion API (1.0.0+)
        response = client.chat.completions.create(model="gpt-4",  # You can change this model depending on your use case
        messages=[
            {"role": "system", "content": "You are a pathology expert analyzing histology reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150)

        # Log the full response to inspect its structure
        logger.debug(f"Full OpenAI response: {json.dumps(response, indent=2)}")

        # Extract feedback from the OpenAI response (adjust based on response structure)
        if 'choices' in response and len(response.choices) > 0:
            feedback = response.choices[0].get('message', {}).get('content', 'No feedback available')
        else:
            feedback = 'No feedback provided by the model'

        # Return the feedback to the user
        return {"feedback": feedback}

    except Exception as e:
        # Log any error that occurs
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}  # Return the error message in case of failure
