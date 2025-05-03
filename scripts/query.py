from transformers import CLIPProcessor, CLIPModel
import torch
from chromadb import PersistentClient

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Connect to the persistent ChromaDB
chroma_client = PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="multimodal_store")

# Get user query
query = input("Enter your text query: ")

# Embed the query
inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77)
with torch.no_grad():
    query_embedding = clip_model.get_text_features(**inputs)
query_embedding = query_embedding[0] / query_embedding.norm()

# Query the collection
results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)

# Print results
for doc, id_ in zip(results['documents'][0], results['ids'][0]):
    print(f"\nID: {id_}")
    print(f"Content: {doc}")