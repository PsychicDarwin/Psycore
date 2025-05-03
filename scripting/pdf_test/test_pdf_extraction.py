
import os
import sys
import json
import shutil
from datetime import datetime
import tempfile

# Add the correct path to the data directory
import sys
sys.path.append(r"C:\Users\anast\Documents\Psycore\src\data")

# Print debugging information
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Try importing the module
try:
    # Import the Attachment class 
    from attachments import Attachment, AttachmentTypes, FailedExtraction
    print("Successfully imported Attachment classes")
except ImportError as e:
    print(f"Import error: {e}")
    print("Detailed error:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Import necessary libraries
try:
    import fitz  # PyMuPDF
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError as e:
    print(f"Error importing required library: {e}")
    print("Please install required libraries with:")
    print("pip install pymupdf pillow numpy")
    sys.exit(1)

# Define paths
PDF_PATH = r"C:\Users\anast\Documents\Psycore\scripting\jupyter_testing\22-036458-01_GIS_early_process_evaluation_Accessible_CLIENT_USE.pdf"
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "document-images")
VECTORS_DIR = os.path.join(os.path.dirname(__file__), "document-vectors")
HIGH_RES_DIR = os.path.join(os.path.dirname(__file__), "high-res-pages")

def create_placeholder_image(path, width=400, height=300, text="Placeholder Image"):
    """Create a proper PNG image with text"""
    try:
        # Create a light gray image
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Draw a border
        draw.rectangle([0, 0, width-1, height-1], outline=(200, 200, 200), width=2)
        
        # Try to use a font if available, otherwise use default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = None
        
        # Draw the text centered
        draw.text((width//2 - 100, height//2 - 10), text, fill=(0, 0, 0), font=font)
        
        # Save the actual PNG file
        img.save(path, "PNG")
        return True
    except Exception as e:
        print(f"Error creating placeholder image: {e}")
        return False

def render_pdf_page_as_image(pdf_path, page_num, output_path, dpi=300):
    """Render a PDF page as a high-quality image"""
    try:
        # Open the PDF
        pdf = fitz.open(pdf_path)
        
        # Check page range
        if page_num < 1 or page_num > len(pdf):
            print(f"Error: Page {page_num} is out of range (1-{len(pdf)})")
            return False
        
        # Get the page (adjusting for 0-based indexing)
        page = pdf[page_num - 1]
        
        # Calculate zoom factor based on DPI
        zoom = dpi / 72  # 72 is the default DPI for PDFs
        
        # Create a matrix for the specified DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render page to a pixmap
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        # Save the image
        pixmap.save(output_path)
        
        # Close the PDF
        pdf.close()
        
        print(f"Rendered high-resolution image of page {page_num} at {dpi} DPI to {output_path}")
        return True
    except Exception as e:
        print(f"Error rendering page: {e}")
        return False

def render_pdf_page_to_svg(pdf_path, page_num, output_path):
    """
    Render a PDF page to SVG format using PyMuPDF's built-in SVG converter
    This is more reliable than our custom drawing conversion
    """
    try:
        # Open the PDF
        pdf = fitz.open(pdf_path)
        
        # Check page range
        if page_num < 1 or page_num > len(pdf):
            print(f"Error: Page {page_num} is out of range (1-{len(pdf)})")
            return False
        
        # Get the page (adjusting for 0-based indexing)
        page = pdf[page_num - 1]
        
        # Convert page to SVG
        svg_data = page.get_svg_image(text_as_path=True)
        
        # Write SVG to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_data)
        
        # Close the PDF
        pdf.close()
        
        print(f"Converted page {page_num} to SVG: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting page to SVG: {e}")
        return False

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extract embedded images from PDF using PyMuPDF
    Saves them as actual image files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the PDF
        pdf = fitz.open(pdf_path)
        
        # Track extracted images
        extracted_images = {}
        
        # Process each page
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            
            # Get images from page
            image_list = page.get_images(full=True)
            
            # Process each image
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Extract image
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create output path
                    img_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index}.{image_ext}")
                    
                    # Save image
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Add to tracking
                    if page_num + 1 not in extracted_images:
                        extracted_images[page_num + 1] = []
                    
                    extracted_images[page_num + 1].append({
                        "path": img_path,
                        "width": base_image.get("width", 0),
                        "height": base_image.get("height", 0),
                        "format": image_ext
                    })
                    
                    print(f"Extracted image from page {page_num+1}: {img_path}")
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num+1}: {e}")
        
        # Close the PDF
        pdf.close()
        
        return extracted_images
    except Exception as e:
        print(f"Error extracting images: {e}")
        return {}

def test_pdf_extraction():
    print(f"Testing enhanced PDF extraction from: {PDF_PATH}")
    
    # Create required directories
    for directory in [DOCUMENTS_DIR, IMAGES_DIR, VECTORS_DIR, HIGH_RES_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Use standard attachment extraction first
    attachment_type = AttachmentTypes.FILE
    attachment = Attachment(attachment_type, PDF_PATH, True)
    
    try:
        # Extract data from the PDF using the Attachment class
        print("Extracting data from PDF using Attachment class...")
        attachment.extract()
        
        # Extract document structure
        doc_structure = None
        doc_id = None
        
        if "document_structure" in attachment.additional_data:
            doc_structure = attachment.additional_data["document_structure"]
            doc_id = doc_structure["document_id"]
            print(f"Document ID: {doc_id}")
        else:
            print("Warning: No document structure found in extracted data")
            doc_id = "pdf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Now extract images directly using PyMuPDF for better quality
        print("\nExtracting images directly using PyMuPDF...")
        images_dir = os.path.join(IMAGES_DIR, doc_id)
        os.makedirs(images_dir, exist_ok=True)
        
        extracted_images = extract_images_from_pdf(PDF_PATH, images_dir)
        print(f"Extracted {sum(len(imgs) for imgs in extracted_images.values())} images from {len(extracted_images)} pages")
        
        # Create vector graphics (SVG) for each page
        print("\nCreating SVG versions of pages...")
        vectors_dir = os.path.join(VECTORS_DIR, doc_id)
        os.makedirs(vectors_dir, exist_ok=True)
        
        # Open the PDF to get page count
        pdf = fitz.open(PDF_PATH)
        page_count = len(pdf)
        pdf.close()
        
        # Process each page
        for page_num in range(1, page_count + 1):
            svg_path = os.path.join(vectors_dir, f"page_{page_num}.svg")
            render_pdf_page_to_svg(PDF_PATH, page_num, svg_path)
        
        # Also create high-resolution PNG images for important pages
        # These are particularly useful for pages with graphs/charts
        important_pages = []
        
        # Ask user if they want to render specific pages
        print("\nWhich pages would you like to render at high resolution?")
        print("Enter page numbers separated by commas, or 'all' for all pages, or 'none' to skip:")
        page_input = input()
        
        if page_input.lower() == 'all':
            important_pages = list(range(1, page_count + 1))
        elif page_input.lower() != 'none':
            important_pages = [int(p.strip()) for p in page_input.split(',') if p.strip().isdigit()]
        
        if important_pages:
            print(f"Rendering {len(important_pages)} pages at high resolution...")
            
            for page_num in important_pages:
                if 1 <= page_num <= page_count:
                    png_path = os.path.join(HIGH_RES_DIR, f"page_{page_num}.png")
                    render_pdf_page_as_image(PDF_PATH, page_num, png_path, dpi=300)
        
        print("\n=== PDF EXTRACTION COMPLETED ===")
        print(f"Document ID: {doc_id}")
        print(f"Images extracted to: {images_dir}")
        print(f"Vector graphics (SVG) saved to: {vectors_dir}")
        print(f"High-resolution pages saved to: {HIGH_RES_DIR}")
        
    except Exception as e:
        print(f"Error in PDF extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_extraction()
