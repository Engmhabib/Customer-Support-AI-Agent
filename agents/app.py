from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import os
import numpy as np
import faiss  # For vector search
import time
from docx import Document  # Import for handling Word files

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)  # Enable CORS

# Global variables
context_chunks = []
index = None
chunk_size = 500  # Adjust the chunk size as needed

# Logging function
def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def chunk_text(text, chunk_size=chunk_size):
    """Split the text into manageable chunks."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text_chunk):
    """Generate an embedding for a text chunk."""
    response = openai.Embedding.create(
        input=text_chunk,
        model="text-embedding-ada-002"  # Example embedding model
    )
    return np.array(response['data'][0]['embedding'])

def create_vector_index(chunks):
    """Create and return a FAISS index from text chunks."""
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def save_faiss_index(index, file_path):
    """Save the FAISS index to a file."""
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    """Load the FAISS index from a file."""
    return faiss.read_index(file_path)

@app.before_first_request
def load_initial_document():
    global context_chunks, index
    initial_file_path = 'uploads/combined.docx'  # Updated to .docx
    faiss_index_path = 'faiss_index.index'
    chunks_file_path = 'context_chunks.npy'  # Save the chunks as a numpy array

    if os.path.exists(faiss_index_path) and os.path.exists(chunks_file_path):
        log("Loading FAISS index and context chunks from disk...")
        index = load_faiss_index(faiss_index_path)
        context_chunks = np.load(chunks_file_path, allow_pickle=True).tolist()
    else:
        if os.path.exists(initial_file_path):
            log("Extracting text from Word document...")
            full_text = extract_text_from_docx(initial_file_path)  # Updated function call
            log("Text extraction completed.")

            log("Chunking text...")
            context_chunks = chunk_text(full_text)
            log(f"Text chunked into {len(context_chunks)} chunks.")

            log("Creating FAISS index...")
            index, _ = create_vector_index(context_chunks)
            log("FAISS index created.")

            # Save FAISS index and chunks to disk
            save_faiss_index(index, faiss_index_path)
            np.save(chunks_file_path, np.array(context_chunks))
        else:
            log("Initial document not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/question', methods=['POST'])
def handle_question():
    global context_chunks, index
    data = request.json
    question = data.get("question")

    if not context_chunks:
        return jsonify({"error": "No document uploaded."}), 400

    log(f"Received question: {question}")
    
    # Generate embedding for the question
    log("Generating embedding for the question...")
    question_embedding = get_embedding(question)
    
    # Search for relevant chunks
    log("Searching for relevant chunks...")
    k = 5  # Number of top relevant chunks to retrieve
    distances, indices = index.search(np.array([question_embedding]).astype('float32'), k)
    relevant_chunks = [context_chunks[i] for i in indices[0]]

    log(f"Retrieved {len(relevant_chunks)} relevant chunks.")

    # Combine relevant chunks to form the context
    context_text = "\n\n".join(relevant_chunks)
    
    # Construct the final prompt
    formatted_prompt = f"""
    Given the following information from the uploaded document:

    {context_text}

    ---

    Please answer the following question:

    {question}
    """
    log("Sending prompt to GPT-4 Turbo...")
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a AI tutor."},
            {"role": "user", "content": formatted_prompt}
        ]
    )
    answer = response.choices[0].message['content'].strip()
    log("Response received from GPT-4 Turbo.")
    return jsonify({"answer": answer})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
