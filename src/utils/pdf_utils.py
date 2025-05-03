# src/utils/pdf_utils.py
import fitz  # PyMuPDF

def extract_prompts_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return [line.strip() for line in text.split("\n") if line.strip()]
