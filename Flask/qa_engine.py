from sentence_transformers import SentenceTransformer, util
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Load transformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + " "
    return text.strip()


def get_answer(pdf_path, question):
    # Extract text from PDF
    text = read_pdf(pdf_path)

    if not text:
        return "❌ No readable text found in the document. Please upload a text-based PDF."

    # Split into sentences
    sentences = sent_tokenize(text)

    # Remove very short sentences
    sentences = [s for s in sentences if len(s.strip()) > 20]

    if len(sentences) == 0:
        return "❌ Document does not contain meaningful content."

    # Create embeddings
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute similarity
    scores = util.cos_sim(question_embedding, sentence_embeddings)[0]

    # Get top 3 relevant sentences
    top_k = min(3, len(sentences))
    top_results = scores.topk(top_k)

    answers = [sentences[int(idx)] for idx in top_results.indices]

    # Combine answers
    final_answer = " ".join(answers)
    return final_answer
