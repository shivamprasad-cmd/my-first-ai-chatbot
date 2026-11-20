from dotenv import load_dotenv
import os
import fitz
import time
import gradio as gr
import json
import urllib.parse
import urllib.request

from google import genai
from google.genai import types, errors


# ==========================================
# 1. GEMINI CONNECTION
# ==========================================
load_dotenv()
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


# ==========================================
# 2. WIKIMEDIA IMAGE SEARCH
# ==========================================

def search_images(query):

    url = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 6,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 500,
        "format": "json"
    }

    try:

        encoded_params = urllib.parse.urlencode(params)

        request = urllib.request.Request(
            f"{url}?{encoded_params}",
            headers={
                "User-Agent": "MyFirstAIChatbot/1.0"
            }
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})

        results = []

        for page in pages.values():

            image_info = page.get("imageinfo", [])

            if not image_info:
                continue

            info = image_info[0]
            metadata = info.get("extmetadata", {})

            results.append({
                "title": page.get("title", "Untitled"),
                "image_url": info.get("thumburl", info.get("url")),
                "source_url": info.get("descriptionurl", ""),
                "license": metadata.get(
                    "LicenseShortName", {}
                ).get("value", "License not specified"),
                "artist": metadata.get(
                    "Artist", {}
                ).get("value", "Unknown author")
            })

        return results

    except Exception as e:

        print("Image search error:", e)
        return []
    
# ==========================================
# IMAGE SEARCH REQUEST DETECTION
# ==========================================

def is_image_search_request(message):

    keywords = [
        "show me images",
        "show images",
        "show me pictures",
        "show pictures",
        "show me photos",
        "show photos",
        "find images",
        "find pictures",
        "find photos",
        "search images",
        "search for images",
        "search pictures",
        "display images",
        "display pictures",
        "get images",
        "get photos"
    ]

    message = message.lower().strip()

    return any(keyword in message for keyword in keywords)

# ==========================================
# EXTRACT IMAGE SEARCH QUERY
# ==========================================

def extract_image_query(message):

    prefixes = [
        "show me images of",
        "show images of",
        "show me pictures of",
        "show pictures of",
        "show me photos of",
        "show photos of",
        "find images of",
        "find pictures of",
        "find photos of",
        "search images for",
        "search for images of",
        "search pictures of",
        "display images of",
        "display pictures of",
        "get images of",
        "get photos of"
    ]

    message = message.strip()

    for prefix in prefixes:

        if message.lower().startswith(prefix):
            return message[len(prefix):].strip()

    return message


# ==========================================
# FORMAT IMAGE SEARCH RESULTS
# ==========================================

def format_image_results(results, query):

    if not results:
        return f"Sorry, I couldn't find images for **{query}**."

    answer = f"## Images found for: {query}\n\n"

    for index, image in enumerate(results, start=1):

        answer += f"### {index}. {image['title']}\n\n"

        answer += f"![{image['title']}]({image['image_url']})\n\n"

        answer += (
            f"**Author:** {image['artist']}\n\n"
            f"**License:** {image['license']}\n\n"
            f"[View original image and licence]({image['source_url']})\n\n"
            "---\n\n"
        )

    return answer


# ==========================================
# 2. PDF TEXT EXTRACTION
# ==========================================

def extract_pdf_text(pdf_path):

    extracted_pages = []
    filename = os.path.basename(pdf_path)

    try:

        pdf = fitz.open(pdf_path)

        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text")

            if page_text.strip():

                extracted_pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": page_text
                })

        pdf.close()

        return extracted_pages if extracted_pages else None

    except Exception as e:

        print("PDF extraction error:", repr(e), flush=True)
        return None


# ==========================================
# 3. TEXT CHUNKING
# ==========================================

def create_chunks(extracted_pages, chunk_size=500, overlap=100):

    chunks = []

    for page in extracted_pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": chunk_text
            })

            start += chunk_size - overlap

    return chunks


# ==========================================
# 4. GEMINI EMBEDDINGS
# ==========================================

def generate_embeddings(texts, task_type):

    all_embeddings = []
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

        if start + batch_size < len(texts):

            print(
                f"Processed {len(all_embeddings)} "
                f"of {len(texts)} chunks",
                flush=True
            )

            time.sleep(10)

    return all_embeddings


# ==========================================
# 5. PROCESS MULTIPLE PDF FILES
# ==========================================

def process_pdfs(pdf_paths):

    if not pdf_paths:
        return None, "No PDF files were uploaded."

    all_pages = []

    try:

        for pdf_path in pdf_paths:

            if not pdf_path.lower().endswith(".pdf"):
                continue

            extracted_pages = extract_pdf_text(pdf_path)

            if extracted_pages:
                all_pages.extend(extracted_pages)

        if not all_pages:

            return None, (
                "No readable PDF text found. "
                "Scanned PDFs may require OCR."
            )

        chunks = create_chunks(all_pages)

        if not chunks:
            return None, "No text chunks could be created."

        # Limit for the current free hosting version
        if len(chunks) > 500:

            return None, (
                "These PDFs contain too much text. "
                "Please upload fewer or smaller PDFs."
            )

        chunk_texts = [
            chunk["text"] for chunk in chunks
        ]

        embeddings = generate_embeddings(
            chunk_texts,
            "RETRIEVAL_DOCUMENT"
        )

        rag_data = {
            "chunks": chunks,
            "embeddings": embeddings
        }

        filenames = list(dict.fromkeys(
            page["source"] for page in all_pages
        ))

        status = (
            f"Successfully processed {len(filenames)} PDF(s), "
            f"{len(all_pages)} readable pages and "
            f"{len(chunks)} chunks.\n\n"
            f"Documents: {', '.join(filenames)}"
        )

        return rag_data, status

    except Exception as e:

        print("PDF processing error:", repr(e), flush=True)

        return None, (
            "PDF processing failed. Please try again later."
        )


# ==========================================
# 6. COSINE SIMILARITY RETRIEVAL
# ==========================================

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

        similarity = sum(
            a * b
            for a, b in zip(
                query_embedding,
                document_embedding
            )
        )

        scored_chunks.append((similarity, index))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    top_results = scored_chunks[:4]

    retrieved = []

    for score, index in top_results:

        retrieved.append({
            "score": score,
            "source": rag_data["chunks"][index]["source"],
            "page": rag_data["chunks"][index]["page"],
            "text": rag_data["chunks"][index]["text"]
        })

    return retrieved


# ==========================================
# 7. CONVERT CHAT HISTORY
# ==========================================

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


# ==========================================
# 8. INTELLIGENT CHAT ROUTING
# ==========================================

def is_pdf_question(message, history):

    text = message.lower().strip()

    small_talk = {
        "hi", "hello", "hey", "hii", "hiii",
        "ok", "okay", "okkk", "thanks",
        "thank you", "thankyou", "cool",
        "great", "nice", "bye",
        "good morning", "good evening"
    }

    if text in small_talk:
        return False

    pdf_keywords = [
        "pdf", "document", "uploaded file",
        "uploaded document", "this file",
        "the file", "this book", "this report",
        "according to", "based on the document",
        "from the document", "in the document",
        "in this document", "in that pdf",
        "in this pdf", "from this pdf",
        "page ", "chapter ", "summarize",
        "what is it about", "what does the document",
        "what does this pdf"
    ]

    if any(keyword in text for keyword in pdf_keywords):
        return True

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

    return False


# ==========================================
# 9. EXTRACT UPLOADED FILE PATHS
# ==========================================

def get_uploaded_paths(files):

    paths = []

    for file_item in files or []:

        if isinstance(file_item, str):

            path = file_item

        elif isinstance(file_item, dict):

            path = (
                file_item.get("path")
                or file_item.get("name")
            )

        else:

            path = getattr(file_item, "path", None)

        if path and str(path).lower().endswith(".pdf"):
            paths.append(str(path))

    return paths


# ==========================================
# 10. MAIN MULTIMODAL CHATBOT
# ==========================================

def chatbot(message, history, rag_data):

    # MultimodalTextbox supplies a dictionary
    if isinstance(message, dict):

        user_text = (message.get("text") or "").strip()

        uploaded_files = get_uploaded_paths(
            message.get("files", [])
        )

    else:

        user_text = str(message or "").strip()
        uploaded_files = []

    updated_rag_data = rag_data

    try:

        # Process newly attached PDFs
        if uploaded_files:

            new_rag_data, status = process_pdfs(uploaded_files)

            if new_rag_data is None:

                return status, rag_data

            updated_rag_data = new_rag_data

            print(status, flush=True)

            # If the user only uploads a PDF
            # without asking a question
            if not user_text:

                return status, updated_rag_data

        # Empty message and no attachment
        if not user_text:

            return (
                "Please enter a message or attach a PDF.",
                updated_rag_data
            )
        
        # ----------------------------------
        # IMAGE SEARCH
        # ----------------------------------

        if is_image_search_request(user_text):

            image_query = extract_image_query(user_text)

            if not image_query:

                return (
                    "Please specify what images you want to search for.",
                    updated_rag_data
                )

            results = search_images(image_query)

            answer = format_image_results(
                results,
                image_query
            )

            return answer, updated_rag_data

        

        # Decide whether this is a PDF question
        use_pdf = (
            updated_rag_data is not None
            and (
                bool(uploaded_files)
                and user_text.lower() not in {
                    "hi", "hello", "hey", "ok",
                    "okay", "thanks", "thank you"
                }
                or is_pdf_question(user_text, history)
            )
        )

        # ----------------------------------
        # NORMAL AI CHAT
        # ----------------------------------

        if not use_pdf:

            contents = convert_history(history)

            contents.append({
                "role": "user",
                "parts": [{"text": user_text}]
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

                    return response.text, updated_rag_data

                except errors.ServerError:

                    if attempt == 2:

                        return (
                            "Gemini is temporarily busy. "
                            "Please try again.",
                            updated_rag_data
                        )

                    time.sleep(2 ** attempt)

        # ----------------------------------
        # PDF QUESTION ANSWERING
        # ----------------------------------

        retrieved = retrieve_chunks(
            user_text,
            updated_rag_data
        )

        context = ""

        for item in retrieved:

            context += (
                f"\n[Document: {item['source']} | "
                f"Page: {item['page']}]\n"
                f"{item['text']}\n"
            )

        rag_prompt = f"""
You are a PDF document assistant.

Answer using ONLY the provided PDF context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not available, say:
   "I don't know based on the uploaded document."
3. Include the document filename and relevant page numbers.
4. Never invent facts or page numbers.

PDF CONTEXT:
{context}

USER QUESTION:
{user_text}
"""

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=rag_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.2
            )
        )

        return response.text, updated_rag_data

    except Exception as e:

        print("CHATBOT ERROR:", repr(e), flush=True)

        return (
            "Something went wrong. Please try again.",
            updated_rag_data
        )

        

# ==========================================
# 11. GRADIO CHAT INTERFACE
# ==========================================

with gr.Blocks() as demo:

    gr.Markdown("# My AI Chatbot")

    gr.Markdown(
        "Ask anything or attach one or more PDFs "
        "using the + button."
    )

    # Per-browser-session document storage
    rag_state = gr.State(value=None)

    gr.ChatInterface(
        fn=chatbot,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_types=[".pdf"],
            file_count="multiple",
            sources=["upload"],
            placeholder="Message your AI assistant..."
        ),
        additional_inputs=[rag_state],
        additional_outputs=[rag_state],
        title="Chat with your AI assistant",
        description=(
            "Chat normally or upload PDFs using the + button "
            "to ask questions about your documents."
        )
    )


# ==========================================
# 12. START APPLICATION
# ==========================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
