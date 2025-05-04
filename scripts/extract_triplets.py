import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

# Load REBEL model and tokenizer
model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_triples(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=10,
            early_stopping=False,
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\n[DEBUG] Raw model output:\n{decoded_output}\n")
    
    triples = []
    
    # Remove special tokens
    cleaned_output = re.sub(r"<s>|</s>", "", decoded_output).strip()
    
    # Extract triplet content
    triplet_info = cleaned_output.split("<triplet> ")[1:]
    triplets = []
    for i in triplet_info:
        split_version = i.split("<subj> ")
        print(split_version)
        subject = split_version[0]
        for i in range(1,len(split_version)):
            obj_rel = split_version[i]
            obj, rel = obj_rel.split("<obj> ")[:2]
            triplets.append({
            "subject": subject,
            "object": obj,
            "relation": rel
            })
    
    return triplets

if __name__ == "__main__":
    text = "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021."
    results = extract_triples(text)
    print(json.dumps(results, indent=2))
else:
    while True:
        text = input("Enter a sentence to extract triples (or type 'exit'): ")
        if text.lower() == "exit":
            break
        
        results = extract_triples(text)
        
        if results:
            print(json.dumps(results, indent=2))
        else:
            print("No triples extracted.")