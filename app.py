
import os
import gradio as gr
from google import genai
from google.genai import types

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

system_prompt = """
You are a friendly AI assistant.
Answer questions in simple English.
Be polite and helpful.
Give practical examples when useful.
"""

def chatbot(message, history):
    contents = []

    for item in history:
        role = "user" if item["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": item["content"]}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=500
        )
    )

    return response.text

demo = gr.ChatInterface(
    fn=chatbot,
    title="My AI Chatbot",
    description="Ask me anything! I'm your friendly AI assistant."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
