import os
import fitz  # PyMuPDF
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="multimodal_store")

# Paths to images and documents
img_folder = "images"
doc_folder = "documents_for_testing"

# === Index Images ===
for img_file in tqdm(os.listdir(img_folder), desc="Indexing Images"):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(img_folder, img_file)
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding = embedding[0] / embedding.norm()
    collection.add(
        documents=[f"Image file: {img_file}"],
        embeddings=[embedding.tolist()],
        ids=[f"img_{img_file}"]
    )

# === Index PDFs ===
for pdf_file in tqdm(os.listdir(doc_folder), desc="Indexing PDFs"):
    if not pdf_file.lower().endswith(".pdf"):
        continue
    pdf_path = os.path.join(doc_folder, pdf_file)
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    doc.close()
    
    if text.strip():
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
        embedding = embedding[0] / embedding.norm()
        collection.add(
            documents=[text[:1000]],  # Optionally truncate long docs
            embeddings=[embedding.tolist()],
            ids=[f"pdf_{pdf_file}"]
        )

print("Indexing complete.")