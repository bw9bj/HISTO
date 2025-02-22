from fastapi import FastAPI, Request
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

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

        response = openai.Completion.create(
            model="gpt-4",  # or another model you are using
            prompt=prompt,
            max_tokens=150
        )

        # Debugging: print the full response object
        print(response)  # This will output the response to the logs

        # Now, ensure you're accessing the response correctly
        if 'choices' in response and len(response['choices']) > 0:
            feedback = response['choices'][0].get('text', 'No feedback available')
        else:
            feedback = 'No feedback provided by the model'

        return {"feedback": feedback}

    except Exception as e:
        return {"error": str(e)}  # This will return the error message in the response


# POST endpoint for analyzing user-provided data (can be used for other types of analysis)
@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()  # Read the JSON data sent in the POST request
    # Process the data here (for example, apply some analysis)
    return {"message": "Analysis complete", "data": data}
