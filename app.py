import os
import fitz
import time
import gradio as gr
from google import genai
from google.genai import types
from google.genai import errors

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

system_prompt = """
You are a friendly AI assistant.
Answer questions in simple English.
Be polite and helpful.
Give practical examples when useful.
"""


def extract_pdf_text(pdf_path):
    """
    Extract text from an uploaded PDF page by page.
    """

    extracted_pages = []

    try:
        pdf = fitz.open(pdf_path)

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")

            if page_text.strip():
                extracted_pages.append({
                    "page": page_number,
                    "text": page_text
                })

        pdf.close()

        if not extracted_pages:
            return None

        return extracted_pages

    except Exception as e:
        print("PDF extraction error:", repr(e))
        return None


def chatbot(message, history):
    contents = []

    # Convert Gradio history into Gemini format
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        content = item.get("content", "")

        # Extract text from Gradio message blocks
        if isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)

            content = "\n".join(text_parts)

        elif isinstance(content, dict):
            content = content.get("text", "")

        if isinstance(content, str) and content.strip():
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

    # Add the latest user message
    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=500
                )
            )

            return response.text

        except errors.ServerError as e:
            print("Gemini server error:", repr(e), flush=True)

            if attempt == 2:
                return "Gemini is temporarily busy. Please try again later."

            time.sleep(2 ** attempt)

        except Exception as e:
            import traceback
            print("CHATBOT ERROR:", repr(e), flush=True)
            traceback.print_exc()
            return f"Something went wrong: {type(e).__name__}"


demo = gr.ChatInterface(
    fn=chatbot,
    title="My AI Chatbot",
    description="Ask me anything! I'm your friendly AI assistant."
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
