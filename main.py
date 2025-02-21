from fastapi import FastAPI, Request
import openai
import os

app = FastAPI()

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    user_input = data["text"]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a pathology expert providing feedback on sample pathology reports."},
            {"role": "user", "content": user_input}
        ]
    )

    return {"feedback": response["choices"][0]["message"]["content"]}

