import os
import sys
import logging
import tempfile
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
import json
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import networkx as nx
from src.interaction.knowledge_graphs import BERTKGTransformer
from src.model.model_catalogue import ModelType, ModelCatalogue, Providers
from src.model.wrappers import ChatModelWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

# Add project root to path to import the Psycore modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Import Psycore modules
from src.data.common_types import AttachmentTypes
from src.data.attachments import Attachment, FailedExtraction
from src.data.pdf_extractor import PDFExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("s3_pdf_processor")

# Set up output directory
OUTPUT_DIR = os.path.join(os.getcwd(), "processed_pdfs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Setup completed. Output directory is:", OUTPUT_DIR)

# Configure bucket names
SOURCE_BUCKET = "psycore-documents-445644858344"
IMAGES_BUCKET = "psycore-document-images-445644858344"
GRAPHS_BUCKET = "psycore-document-graphs-445644858344"

# Initialize model for image analysis
def get_model(model_type=None):
    """
    Get a model instance based on the specified type.
    If no type is specified, defaults to Bakllava 7B.
    """
    if model_type is None:
        model_type = ModelCatalogue._models["bakllava_7b"]  # Using Bakllava 7B as default for local processing
    return ChatModelWrapper(model_type)

def analyze_image_with_llm(image_path: str, image_s3_url: str, original_doc_s3_url: str, model=None) -> dict:
    """
    Analyze an image using a language model and return the analysis.
    
    Args:
        image_path: Path to the image file
        image_s3_url: S3 URL where the image is stored
        original_doc_s3_url: S3 URL of the original PDF document
        model: Optional model instance to use (if None, uses default model)
    """
    try:
        if model is None:
            model = get_model()
            
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Create message format based on model type
        if model.model_type.provider == Providers.OLLAMA:
            # Ollama format requires image_url type with base64 data URL
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""Analyze this image and provide a detailed description. Consider:
1. What is shown in the image?
2. What is the main purpose or message of this image?
3. How does it relate to the document it's from?
4. Are there any key data points, figures, or text visible?

Original document: {original_doc_s3_url}
Image location: {image_s3_url}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_data}"
                    }
                ]
            )
        else:
            # Default format for other providers (Claude, Gemini, etc.)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""Analyze this image and provide a detailed description. Consider:
1. What is shown in the image?
2. What is the main purpose or message of this image?
3. How does it relate to the document it's from?
4. Are there any key data points, figures, or text visible?

Original document: {original_doc_s3_url}
Image location: {image_s3_url}"""
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    }
                ]
            )
        
        # Use LangChain's chat interface with a single message
        response = model.model.invoke([message])
        
        # Extract the response text
        analysis_text = str(response.content) if hasattr(response, 'content') else str(response)
        
        return {
            "analysis": analysis_text,
            "image_s3_url": image_s3_url,
            "original_doc_s3_url": original_doc_s3_url,
            "model_type": str(model.model_type)
        }
    except Exception as e:
        logger.error(f"Failed to analyze image with model: {str(e)}", exc_info=True)  # Added exc_info for better error tracking
        return {
            "error": str(e),
            "image_s3_url": image_s3_url,
            "original_doc_s3_url": original_doc_s3_url,
            "model_type": str(model.model_type) if model else "unknown"
        }

def extract_knowledge_graph(text: str, image_analyses: list) -> nx.DiGraph:
    """
    Extract a knowledge graph from text and image analyses.
    """
    # Initialize BERT transformer with custom entity types
    kg_transformer = BERTKGTransformer(
        allowed_nodes=['PERSON', 'ORG', 'LOC', 'DATE', 'FIGURE', 'CONCEPT', 'DATA', 'IMAGE', 'TOPIC', 'KEY_POINT'],
        allowed_relationships=[
            ('PERSON', 'works_for', 'ORG'),
            ('PERSON', 'located_in', 'LOC'),
            ('ORG', 'based_in', 'LOC'),
            ('CONCEPT', 'related_to', 'CONCEPT'),
            ('DATA', 'supports', 'CONCEPT'),
            ('FIGURE', 'illustrates', 'CONCEPT'),
            ('IMAGE', 'contains', 'CONCEPT'),
            ('IMAGE', 'shows', 'FIGURE'),
            ('TOPIC', 'includes', 'CONCEPT'),
            ('KEY_POINT', 'supports', 'TOPIC'),
            ('DATA', 'part_of', 'FIGURE'),
            ('CONCEPT', 'mentioned_in', 'IMAGE'),
            ('FIGURE', 'appears_in', 'IMAGE'),
            ('TOPIC', 'discussed_in', 'IMAGE')
        ]
    )
    
    # Create a new graph
    graph = nx.DiGraph()
    
    # Process main document text in chunks to handle large documents
    chunk_size = 1000  # Process 1000 characters at a time
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    for chunk in text_chunks:
        if not chunk.strip():  # Skip empty chunks
            continue
        # Process chunk and merge with main graph
        chunk_graph = kg_transformer.process_text(chunk)
        graph = nx.compose(graph, chunk_graph)
    
    # Process image analyses
    for analysis in image_analyses:
        if 'analysis' in analysis and 'image_s3_url' in analysis:
            # Add image node
            image_node = analysis['image_s3_url']
            graph.add_node(image_node, type='IMAGE')
            
            # Process image analysis text
            analysis_text = analysis['analysis']
            if isinstance(analysis_text, str) and analysis_text.strip():
                # Extract key concepts from the analysis
                img_graph = kg_transformer.process_text(analysis_text)
                
                # Add edges between image and extracted concepts
                for node, data in img_graph.nodes(data=True):
                    if data.get('type') in ['CONCEPT', 'FIGURE', 'TOPIC', 'KEY_POINT']:
                        # Add the node and its data to the main graph
                        graph.add_node(node, **data)
                        # Connect image to the concept
                        if data.get('type') == 'CONCEPT':
                            graph.add_edge(image_node, node, relationship='contains')
                        elif data.get('type') == 'FIGURE':
                            graph.add_edge(image_node, node, relationship='shows')
                        elif data.get('type') == 'TOPIC':
                            graph.add_edge(image_node, node, relationship='discusses')
                        elif data.get('type') == 'KEY_POINT':
                            graph.add_edge(image_node, node, relationship='illustrates')
                
                # Add relationships between concepts
                for u, v, data in img_graph.edges(data=True):
                    if u in graph and v in graph:  # Only add edges if both nodes exist
                        graph.add_edge(u, v, **data)
    
    # Ensure all nodes have a type
    for node in graph.nodes():
        if 'type' not in graph.nodes[node]:
            if any(node.startswith(prefix) for prefix in ['s3://', 'http://', 'https://']):
                graph.nodes[node]['type'] = 'IMAGE'
            else:
                graph.nodes[node]['type'] = 'CONCEPT'
    
    # Ensure all edges have a relationship
    for u, v in graph.edges():
        if 'relationship' not in graph.edges[u, v]:
            graph.edges[u, v]['relationship'] = 'related_to'
    
    return graph

# S3 Configuration
def configure_s3_client(aws_access_key_id=None, aws_secret_access_key=None, region_name='eu-west-2'):
    """
    Configure and return an S3 client
    If no credentials are provided, boto3 will use the default credential chain
    """
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

def list_pdfs_in_s3(s3_client, bucket_name, prefix=''):
    """
    List all PDF files in the specified S3 bucket and prefix
    
    Returns:
        list: List of dictionaries containing file info
    """
    pdf_files = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].lower().endswith('.pdf'):
                        size_mb = obj['Size'] / (1024 * 1024)  # Convert to MB
                        pdf_files.append({
                            'key': obj['Key'],
                            'size': size_mb,
                            'last_modified': obj['LastModified']
                        })
    except ClientError as e:
        logger.error(f"Error listing PDFs in S3: {str(e)}")
    
    return pdf_files

def download_from_s3(s3_client, bucket_name, s3_key, local_path):
    """
    Download a file from S3
    """
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        logger.error(f"Failed to download {s3_key} from S3: {str(e)}")
        return False

def upload_to_s3(s3_client, file_path, bucket_name, s3_key):
    """
    Upload a file to S3
    """
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return True
    except ClientError as e:
        logger.error(f"Failed to upload {file_path} to S3: {str(e)}")
        return False

def process_pdf(pdf_path, output_dir=None):
    """
    Process a PDF file using the PDFExtractor and save results.
    """
    result = {
        "filename": os.path.basename(pdf_path),
        "success": False,
        "text_extracted": False,
        "image_count": 0,
        "error": None,
        "output_dir": None,
        "image_analyses": []
    }
    
    try:
        # Create output directory if specified
        pdf_basename = os.path.basename(pdf_path).replace(".", "_")
        
        if output_dir:
            result_dir = os.path.join(output_dir, pdf_basename)
            os.makedirs(result_dir, exist_ok=True)
            result["output_dir"] = result_dir
        
        # Create attachment and process it
        attachment = Attachment(AttachmentTypes.FILE, pdf_path, True)
        attachment.extract()
        
        # Get extracted images
        extracted_images = attachment.pop_extra_attachments()
        result["image_count"] = len(extracted_images)
        
        # Get extracted text
        extracted_text = attachment.attachment_data
        result["text_extracted"] = len(extracted_text) > 0
        
        # Save results if output directory is specified
        if output_dir:
            # Save text content
            if result["text_extracted"]:
                text_path = os.path.join(result_dir, "extracted_text.txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
            
            # Save extracted images and analyze them
            images_dir = os.path.join(result_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            for i, img in enumerate(extracted_images):
                try:
                    # Convert base64 to image
                    img_data = img.attachment_data
                    img_bytes = base64.b64decode(img_data)
                    
                    # Save the image
                    img_path = os.path.join(images_dir, f"image_{i+1}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                        
                    # Store image analysis
                    result["image_analyses"].append({
                        "image_path": img_path,
                        "page_number": img.metadata.get("page_number", "unknown"),
                        "page_text": img.metadata.get("page_text", "")
                    })
                except Exception as e:
                    logger.warning(f"Failed to save image {i+1}: {str(e)}")
            
            # Save metadata
            metadata = {
                "filename": os.path.basename(pdf_path),
                "text_length": len(extracted_text),
                "image_count": len(extracted_images),
                "processed_at": str(datetime.now())
            }
            
            metadata_path = os.path.join(result_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        result["success"] = True
        
    except FailedExtraction as e:
        result["success"] = False
        result["error"] = str(e)
        logger.error(f"Failed extraction for {pdf_path}: {str(e)}")
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        logger.error(f"Error processing {pdf_path}: {str(e)}")
    
    return result

def process_pdf_from_s3(s3_client, bucket_name, s3_key, output_dir, model=None):
    """
    Download a PDF from S3, process it, and upload results back to S3
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: Source bucket name
        s3_key: Key of the PDF file in S3
        output_dir: Local directory to store processed files
        model: Optional model instance to use for image analysis
    """
    # Create a temporary directory for downloading
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download PDF from S3
        local_pdf_path = os.path.join(temp_dir, os.path.basename(s3_key))
        if download_from_s3(s3_client, bucket_name, s3_key, local_pdf_path):
            # Process the PDF
            start_time = datetime.now()
            result = process_pdf(local_pdf_path, output_dir)
            processing_time = datetime.now() - start_time
            
            print(f"\nProcessing completed in {processing_time.total_seconds():.2f} seconds")
            print(f"Success: {result['success']}")
            print(f"Extracted text: {'Yes' if result['text_extracted'] else 'No'}")
            print(f"Images extracted: {result['image_count']}")
            
            if result['output_dir']:
                print(f"Results saved to: {result['output_dir']}")
                
                # Upload results back to S3
                pdf_name = os.path.basename(s3_key).replace('.pdf', '')
                results_prefix = f"processed/{pdf_name}"
                doc_s3_url = f"s3://{bucket_name}/{s3_key}"
                
                # Upload extracted text
                if result['text_extracted']:
                    text_path = os.path.join(result['output_dir'], "extracted_text.txt")
                    upload_to_s3(s3_client, text_path, bucket_name, f"{results_prefix}/extracted_text.txt")
                
                # Upload metadata
                metadata_path = os.path.join(result['output_dir'], "metadata.json")
                upload_to_s3(s3_client, metadata_path, bucket_name, f"{results_prefix}/metadata.json")
                
                # Process and upload images
                images_dir = os.path.join(result['output_dir'], "images")
                image_analyses = []
                
                if os.path.exists(images_dir):
                    for img_file in os.listdir(images_dir):
                        img_path = os.path.join(images_dir, img_file)
                        # Upload to the dedicated images bucket with PDF name as prefix
                        image_key = f"{pdf_name}/{img_file}"
                        image_s3_url = f"s3://{IMAGES_BUCKET}/{image_key}"
                        
                        if upload_to_s3(s3_client, img_path, IMAGES_BUCKET, image_key):
                            print(f"Uploaded image to: {image_s3_url}")
                            
                            # Analyze image with model
                            analysis = analyze_image_with_llm(img_path, image_s3_url, doc_s3_url, model)
                            image_analyses.append(analysis)
                
                # Save image analyses
                analyses_path = os.path.join(result['output_dir'], "image_analyses.json")
                with open(analyses_path, "w") as f:
                    json.dump(image_analyses, f, indent=2)
                upload_to_s3(s3_client, analyses_path, bucket_name, f"{results_prefix}/image_analyses.json")
                
                # Extract knowledge graph
                if result['text_extracted']:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        doc_text = f.read()
                    
                    # Create knowledge graph
                    graph = extract_knowledge_graph(doc_text, image_analyses)
                    
                    # Save graph data
                    graph_data = {
                        "nodes": [
                            {
                                "id": n,
                                "type": graph.nodes[n].get('type', 'UNKNOWN'),
                                "label": n.split('/')[-1] if '/' in n else n  # Use last part of URL for images
                            } for n in graph.nodes()
                        ],
                        "edges": [
                            {
                                "source": u,
                                "target": v,
                                "relationship": graph.edges[u, v].get('relationship', 'related_to'),
                                "weight": 1.0
                            } for u, v in graph.edges()
                        ],
                        "metadata": {
                            "model_used": str(model.model_type) if model else "default",
                            "processed_at": str(datetime.now()),
                            "source_document": doc_s3_url,
                            "node_count": len(graph.nodes()),
                            "edge_count": len(graph.edges()),
                            "node_types": list(set(data.get('type', 'UNKNOWN') 
                                                 for _, data in graph.nodes(data=True))),
                            "relationship_types": list(set(data.get('relationship', 'related_to') 
                                                        for _, _, data in graph.edges(data=True)))
                        }
                    }
                    
                    graph_path = os.path.join(result['output_dir'], "knowledge_graph.json")
                    with open(graph_path, "w") as f:
                        json.dump(graph_data, f, indent=2)
                    
                    # Upload to graphs bucket
                    upload_to_s3(s3_client, graph_path, GRAPHS_BUCKET, f"{pdf_name}/knowledge_graph.json")
                    print(f"Uploaded knowledge graph to: s3://{GRAPHS_BUCKET}/{pdf_name}/knowledge_graph.json")
            
            return result
        else:
            print(f"Failed to download PDF from S3: {s3_key}")
            return None

# Initialize S3 client
s3_client = configure_s3_client()

# List PDFs in S3
pdf_files = list_pdfs_in_s3(s3_client, SOURCE_BUCKET)

if not pdf_files:
    print("No PDF files found in S3 bucket:", SOURCE_BUCKET)
else:
    print(f"Found {len(pdf_files)} PDF files in S3:")
    for i, pdf_file in enumerate(pdf_files):
        print(f"{i+1}. {pdf_file['key']} ({pdf_file['size']:.2f} MB)")
        print(f"   Last modified: {pdf_file['last_modified']}")
    
    # Initialize model once for all processing
    model = get_model()
    print(f"\nUsing model: {model.model_type}")
    
    print("\nStarting batch processing of all PDFs...")
    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing PDF {i+1}/{len(pdf_files)}: {pdf_file['key']}")
        try:
            result = process_pdf_from_s3(s3_client, SOURCE_BUCKET, pdf_file['key'], OUTPUT_DIR, model)
            if result and result['success']:
                print(f"✓ Successfully processed {pdf_file['key']}")
                if result['image_count'] > 0:
                    print(f"  - {result['image_count']} images uploaded to {IMAGES_BUCKET}")
            else:
                print(f"✗ Failed to process {pdf_file['key']}")
        except Exception as e:
            print(f"✗ Error processing {pdf_file['key']}: {str(e)}")
    
    print("\nBatch processing completed!") 