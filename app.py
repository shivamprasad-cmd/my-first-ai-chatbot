
import os
import fitz
import time
import math
import gradio as gr

from google import genai
from google.genai import types, errors


# ----------------------------------
# 1. GEMINI CONNECTION
# ----------------------------------

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

system_prompt = """
You are a friendly AI assistant.
Answer questions in simple English.
Be polite and helpful.
Give practical examples when useful.
"""


# ----------------------------------
# 2. PDF TEXT EXTRACTION
# ----------------------------------

def extract_pdf_text(pdf_path):

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

        print("PDF extraction error:", repr(e), flush=True)
        return None


# ----------------------------------
# 3. TEXT CHUNKING
# ----------------------------------

def create_chunks(extracted_pages, chunk_size=500, overlap=100):

    chunks = []

    for page in extracted_pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "page": page["page"],
                "text": chunk_text
            })

            start += chunk_size - overlap

    return chunks


# ----------------------------------
# 4. GEMINI EMBEDDINGS
# ----------------------------------


def generate_embeddings(texts, task_type):

    all_embeddings = []

    # Smaller batches to control API usage
    batch_size = 6

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        for attempt in range(3):

            try:

                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=768
                    )
                )

                for embedding in result.embeddings:
                    all_embeddings.append(embedding.values)

                break

            except errors.ClientError as e:

                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):

                    if attempt == 2:
                        raise

                    wait_time = 60 * (attempt + 1)

                    print(
                        f"Embedding quota reached. "
                        f"Retrying in {wait_time} seconds...",
                        flush=True
                    )

                    time.sleep(wait_time)

                else:
                    raise

        # Allow the rate-limit window to recover
        if start + batch_size < len(texts):

            print(
                f"Processed {len(all_embeddings)} "
                f"of {len(texts)} chunks",
                flush=True
            )

            time.sleep(10)

    return all_embeddings



# ----------------------------------
# 5. PROCESS UPLOADED PDF
# ----------------------------------

def process_pdf(pdf_path):

    if not pdf_path:

        return None, "Please upload a PDF first."

    if not str(pdf_path).lower().endswith(".pdf"):

        return None, "Please upload a valid PDF file."

    try:

        extracted_pages = extract_pdf_text(pdf_path)

        if not extracted_pages:

            return None, (
                "No readable text found. "
                "This PDF may contain scanned images."
            )

        chunks = create_chunks(extracted_pages)

        if not chunks:

            return None, "No text chunks could be created."

        # Safety limit for the free hosting instance
        if len(chunks) > 500:

            return None, (
                "This PDF is too large for the current version. "
                "Please upload a smaller PDF."
            )

        chunk_texts = [
            chunk["text"] for chunk in chunks
        ]

        # Generate document embeddings
        embeddings = generate_embeddings(
            chunk_texts,
            "RETRIEVAL_DOCUMENT"
        )

        rag_data = {
            "chunks": chunks,
            "embeddings": embeddings
        }

        message = (
            f"PDF processed successfully! "
            f"{len(extracted_pages)} pages and "
            f"{len(chunks)} text chunks are ready. "
            f"You can now ask questions about your PDF."
        )

        return rag_data, message

    except Exception as e:

        print("PDF processing error:", repr(e), flush=True)

        return None, (
            "PDF processing failed. Please check the Render logs "
            "for details and try again."
        )


# ----------------------------------
# 6. COSINE SIMILARITY RETRIEVAL
# ----------------------------------

def retrieve_chunks(question, rag_data):

    query_result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )

    query_embedding = query_result.embeddings[0].values

    scored_chunks = []

    for index, document_embedding in enumerate(
        rag_data["embeddings"]
    ):

        # Gemini embeddings are normalized.
        # Their dot product gives cosine similarity.
        similarity = sum(
            a * b
            for a, b in zip(
                query_embedding,
                document_embedding
            )
        )

        scored_chunks.append(
            (similarity, index)
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    top_results = scored_chunks[:4]

    retrieved = []

    for score, index in top_results:

        retrieved.append({
            "score": score,
            "page": rag_data["chunks"][index]["page"],
            "text": rag_data["chunks"][index]["text"]
        })

    return retrieved


# ----------------------------------
# 7. NORMAL CHAT HISTORY
# ----------------------------------

def convert_history(history):

    contents = []

    for item in history:

        role = (
            "user"
            if item["role"] == "user"
            else "model"
        )

        content = item.get("content", "")

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

    return contents
    
def is_pdf_question(message, history):

    text = message.lower().strip()

    # Greetings and acknowledgements should always use normal chat
    small_talk = {
        "hi", "hello", "hey", "hii", "hiii",
        "ok", "okay", "okkk", "thanks", "thank you",
        "thankyou", "cool", "great", "nice",
        "bye", "good morning", "good evening"
    }

    if text in small_talk:
        return False

    # Explicit references to uploaded documents
    pdf_keywords = [
        "pdf", "document", "uploaded file",
        "uploaded document", "this file",
        "the file", "this book", "this report",
        "according to", "based on the document",
        "from the document", "in the document",
        "in this document", "in that pdf",
        "in this pdf", "from this pdf",
        "page ", "chapter ", "summarize this",
        "summarize the", "what is it about",
        "what does the document", "what does this pdf"
    ]

    if any(keyword in text for keyword in pdf_keywords):
        return True

    # Recognize short follow-up questions about the PDF
    followups = [
        "tell me more", "explain more",
        "more details", "elaborate",
        "explain that", "what about that",
        "what about it", "can you explain",
        "summarize it", "continue",
        "give me more information"
    ]

    if any(phrase in text for phrase in followups):

        recent_history = convert_history(history)[-6:]

        for item in recent_history:

            previous_message = (
                item["parts"][0]["text"].lower()
                if item.get("parts")
                else ""
            )

            if item["role"] == "user" and any(
                keyword in previous_message
                for keyword in pdf_keywords
            ):
                return True

    # Default to normal AI conversation
    return False



# ----------------------------------
# 8. MAIN CHATBOT FUNCTION
# ----------------------------------


def chatbot(message, history, rag_data):

    if not message or not message.strip():
        return "Please enter a question."

    try:

        # Decide which chatbot mode to use
        use_pdf = (
            rag_data is not None
            and is_pdf_question(message, history)
        )

        # ----------------------------------
        # MODE 1: NORMAL AI CHAT
        # ----------------------------------

        if not use_pdf:

            contents = convert_history(history)

            contents.append({
                "role": "user",
                "parts": [{"text": message}]
            })

            for attempt in range(3):

                try:

                    response = client.models.generate_content(
                        model=CHAT_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=500
                        )
                    )

                    return response.text

                except errors.ServerError:

                    if attempt == 2:
                        return (
                            "Gemini is temporarily busy. "
                            "Please try again."
                        )

                    time.sleep(2 ** attempt)

        # ----------------------------------
        # MODE 2: PDF QUESTION ANSWERING
        # ----------------------------------

        retrieved = retrieve_chunks(message, rag_data)

        context = ""

        for item in retrieved:

            context += (
                f"\n[PDF Page {item['page']}]\n"
                f"{item['text']}\n"
            )

        rag_prompt = f"""
You are a PDF document assistant.

Answer using ONLY the provided PDF context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not available, say:
   "I don't know based on the uploaded document."
3. Include relevant PDF page numbers.
4. Never invent facts or page numbers.

PDF CONTEXT:
{context}

USER QUESTION:
{message}
"""

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=rag_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.2
            )
        )

        return response.text

    except Exception as e:

        print("CHATBOT ERROR:", repr(e), flush=True)

        return (
            "Something went wrong while processing your request. "
            "Please try again."
        )


# ----------------------------------
# 9. GRADIO USER INTERFACE
# ----------------------------------

with gr.Blocks() as demo:

    gr.Markdown("# My AI Chatbot")

    gr.Markdown(
        "Chat normally or upload a PDF to ask questions "
        "about its contents."
    )

    rag_state = gr.State(value=None)

    with gr.Row():

        pdf_input = gr.File(
            label="Upload your PDF",
            file_types=[".pdf"],
            type="filepath"
        )

        process_button = gr.Button(
            "Process PDF",
            variant="primary"
        )

    pdf_status = gr.Markdown(
        "No PDF uploaded yet."
    )

    process_button.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[rag_state, pdf_status]
    )

    gr.ChatInterface(
        fn=chatbot,
        additional_inputs=[rag_state],
        title="Chat with your AI assistant",
        description=(
            "Ask general questions or ask questions "
            "about your processed PDF."
        )
    )


# ----------------------------------
# 10. START APPLICATION
# ----------------------------------

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
