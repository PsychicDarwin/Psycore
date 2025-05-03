
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime
import traceback
import argparse

# Try importing required libraries
try:
    import fitz  # PyMuPDF
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install required libraries with: pip install pymupdf pillow")
    sys.exit(1)

class PDFExtractor:
    """PDF content extraction tool with support for images and vector graphics"""
    
    def __init__(self, pdf_path, output_dir=None):
        """
        Initialize the PDF extractor
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted content (default: auto-generated based on PDF name)
        """
        self.pdf_path = pdf_path
        
        # Verify the PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output directory if not specified
        if output_dir is None:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            self.output_dir = f"pdf_output_{pdf_name}"
        else:
            self.output_dir = output_dir
            
        # Create output directories
        self.images_dir = os.path.join(self.output_dir, "images")
        self.svg_dir = os.path.join(self.output_dir, "svg")
        self.highres_dir = os.path.join(self.output_dir, "highres")
        
        # Check PDF
        try:
            pdf = fitz.open(pdf_path)
            self.page_count = len(pdf)
            pdf.close()
        except Exception as e:
            raise ValueError(f"Failed to open PDF file: {e}")
            
    def create_output_dirs(self):
        """Create required output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.svg_dir, exist_ok=True)
        os.makedirs(self.highres_dir, exist_ok=True)
        
    def extract_images(self):
        """Extract embedded images from the PDF"""
        extracted_images = {}
        
        # Use a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Open the PDF
            pdf = fitz.open(self.pdf_path)
            
            # Process each page
            for page_num in range(self.page_count):
                page = pdf[page_num]
                image_list = page.get_images(full=True)
                
                if not image_list:
                    continue
                    
                # Create page directory for this page's images
                page_dir = os.path.join(self.images_dir, f"page_{page_num+1}")
                os.makedirs(page_dir, exist_ok=True)
                
                # Track images for this page
                page_images = []
                
                # Process each image
                for img_index, img_info in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img_info[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Create temp file
                        temp_path = os.path.join(temp_dir, f"img_{img_index}.{image_ext}")
                        with open(temp_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Only save images larger than 100x100 pixels to filter out tiny icons
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        
                        if width >= 100 and height >= 100:
                            # Save to output directory
                            output_path = os.path.join(page_dir, f"img_{img_index}.{image_ext}")
                            shutil.copy2(temp_path, output_path)
                            
                            # Track image
                            page_images.append({
                                "path": output_path,
                                "width": width,
                                "height": height,
                                "format": image_ext
                            })
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num+1}: {e}")
                
                if page_images:
                    extracted_images[page_num + 1] = page_images
            
            # Close the PDF
            pdf.close()
        
        return extracted_images
    
    def extract_svg(self, pages=None):
        """
        Extract pages as SVG files (vector graphics)
        
        Args:
            pages: List of page numbers to extract, or None for all pages
        """
        if pages is None:
            pages = list(range(1, self.page_count + 1))
            
        extracted_svg = {}
        
        # Use a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Open the PDF
            pdf = fitz.open(self.pdf_path)
            
            # Process each page
            for page_num in pages:
                if page_num < 1 or page_num > self.page_count:
                    print(f"Warning: Page {page_num} is out of range (1-{self.page_count})")
                    continue
                    
                try:
                    # Get the page (0-based indexing)
                    page = pdf[page_num - 1]
                    
                    # Create temporary SVG file
                    temp_path = os.path.join(temp_dir, f"page_{page_num}.svg")
                    
                    # Convert page to SVG
                    svg_data = page.get_svg_image(text_as_path=True)
                    
                    # Save to temp file
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(svg_data)
                    
                    # Copy to output directory
                    output_path = os.path.join(self.svg_dir, f"page_{page_num}.svg")
                    shutil.copy2(temp_path, output_path)
                    
                    # Track SVG file
                    extracted_svg[page_num] = output_path
                except Exception as e:
                    print(f"Error extracting SVG from page {page_num}: {e}")
            
            # Close the PDF
            pdf.close()
        
        return extracted_svg
    
    def render_high_resolution(self, pages=None, dpi=300):
        """
        Render high-resolution images of PDF pages
        
        Args:
            pages: List of page numbers to render, or None for all pages
            dpi: DPI for rendering (higher = better quality but larger files)
        """
        if pages is None:
            pages = list(range(1, self.page_count + 1))
            
        rendered_pages = {}
        
        # Use a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Open the PDF
            pdf = fitz.open(self.pdf_path)
            
            # Process each page
            for page_num in pages:
                if page_num < 1 or page_num > self.page_count:
                    print(f"Warning: Page {page_num} is out of range (1-{self.page_count})")
                    continue
                    
                try:
                    # Get the page (0-based indexing)
                    page = pdf[page_num - 1]
                    
                    # Calculate zoom factor based on DPI
                    zoom = dpi / 72  # 72 is the default DPI for PDFs
                    
                    # Create a matrix for the specified DPI
                    matrix = fitz.Matrix(zoom, zoom)
                    
                    # Create temporary PNG file
                    temp_path = os.path.join(temp_dir, f"page_{page_num}.png")
                    
                    # Render page to a pixmap
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    pixmap.save(temp_path)
                    
                    # Copy to output directory
                    output_path = os.path.join(self.highres_dir, f"page_{page_num}.png")
                    shutil.copy2(temp_path, output_path)
                    
                    # Track rendered page
                    rendered_pages[page_num] = {
                        "path": output_path,
                        "dpi": dpi,
                        "width": pixmap.width,
                        "height": pixmap.height
                    }
                except Exception as e:
                    print(f"Error rendering page {page_num}: {e}")
            
            # Close the PDF
            pdf.close()
        
        return rendered_pages
    
    def extract_all(self, extract_images=True, extract_svg=True, 
                    render_highres=None, dpi=300):
        """
        Extract all content from the PDF
        
        Args:
            extract_images: Whether to extract embedded images
            extract_svg: Whether to extract pages as SVG
            render_highres: List of page numbers to render in high resolution, or None for none
            dpi: DPI for high-resolution rendering
            
        Returns:
            Dictionary with extraction results
        """
        # Create output directories
        self.create_output_dirs()
        
        # Initialize results
        results = {
            "pdf_path": self.pdf_path,
            "output_dir": self.output_dir,
            "page_count": self.page_count,
            "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Extract images if requested
        if extract_images:
            print(f"Extracting images from PDF...")
            extracted_images = self.extract_images()
            
            # Count images
            image_count = sum(len(images) for images in extracted_images.values())
            page_count = len(extracted_images)
            
            results["images"] = {
                "count": image_count,
                "pages": list(extracted_images.keys()),
                "directory": self.images_dir
            }
            
            print(f"Extracted {image_count} images from {page_count} pages")
        
        # Extract SVG if requested
        if extract_svg:
            print(f"Extracting vector graphics (SVG) from PDF...")
            extracted_svg = self.extract_svg()
            
            results["svg"] = {
                "count": len(extracted_svg),
                "pages": list(extracted_svg.keys()),
                "directory": self.svg_dir
            }
            
            print(f"Extracted SVG from {len(extracted_svg)} pages")
        
        # Render high-resolution images if requested
        if render_highres:
            print(f"Rendering high-resolution images of selected pages...")
            rendered_pages = self.render_high_resolution(pages=render_highres, dpi=dpi)
            
            results["highres"] = {
                "count": len(rendered_pages),
                "pages": list(rendered_pages.keys()),
                "directory": self.highres_dir,
                "dpi": dpi
            }
            
            print(f"Rendered {len(rendered_pages)} pages at {dpi} DPI")
        
        # Save extraction report
        report_path = os.path.join(self.output_dir, "extraction_report.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Extraction complete. Results saved to {self.output_dir}")
        print(f"Report saved to {report_path}")
        
        return results
    
    def analyze_content(self):
        """Analyze PDF content and suggest pages with graphs/charts"""
        suggested_pages = []
        
        # Open the PDF
        pdf = fitz.open(self.pdf_path)
        
        # Look for pages with likely vector graphics content
        for page_num in range(self.page_count):
            page = pdf[page_num]
            
            # Get vector content (drawings)
            drawings = page.get_drawings()
            
            # Count text blocks
            text_blocks = page.get_text("blocks")
            
            # If page has significant drawings and moderate text, it might contain graphs/charts
            if len(drawings) > 10 and len(text_blocks) < 20:
                suggested_pages.append(page_num + 1)  # Convert to 1-based page numbers
        
        # Close the PDF
        pdf.close()
        
        return suggested_pages

def create_html_report(results, output_path):
    """Create an HTML report of extraction results"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PDF Extraction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
        .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background: #f9f9f9; }}
        .card img {{ max-width: 100%; height: auto; }}
        .metadata {{ font-size: 0.9em; color: #7f8c8d; margin-top: 10px; }}
        .button {{ display: inline-block; background: #3498db; color: white; padding: 8px 16px; 
                 border-radius: 4px; text-decoration: none; margin-top: 10px; }}
        .button:hover {{ background: #2980b9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PDF Extraction Report</h1>
        
        <div class="section">
            <h2>Overview</h2>
            <p>File: {os.path.basename(results['pdf_path'])}</p>
            <p>Total Pages: {results['page_count']}</p>
            <p>Extraction Date: {results['extraction_time']}</p>
        </div>
"""
    
    # Add images section if available
    if "images" in results:
        image_count = results["images"]["count"]
        image_pages = results["images"]["pages"]
        
        html += f"""
        <div class="section">
            <h2>Extracted Images ({image_count})</h2>
            <p>Images found on {len(image_pages)} pages: {', '.join(map(str, sorted(image_pages)))}</p>
            
            <div class="grid">
"""
        
        # Add card for each page with images
        image_dir = results["images"]["directory"]
        for page_num in sorted(image_pages):
            page_dir = os.path.join(image_dir, f"page_{page_num}")
            
            # Find the first image for this page to show as preview
            image_files = [f for f in os.listdir(page_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if image_files:
                preview_image = os.path.join(f"page_{page_num}", image_files[0])
                html += f"""
                <div class="card">
                    <h3>Page {page_num}</h3>
                    <img src="images/{preview_image}" alt="Preview image from page {page_num}">
                    <div class="metadata">{len(image_files)} image(s) found</div>
                    <a href="images/page_{page_num}" class="button">View All</a>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add SVG section if available
    if "svg" in results:
        svg_count = results["svg"]["count"]
        svg_pages = results["svg"]["pages"]
        
        html += f"""
        <div class="section">
            <h2>Vector Graphics (SVG) ({svg_count})</h2>
            <p>Vector content extracted from {svg_count} pages</p>
            
            <div class="grid">
"""
        
        # Add card for each SVG page
        for page_num in sorted(svg_pages):
            html += f"""
                <div class="card">
                    <h3>Page {page_num}</h3>
                    <embed src="svg/page_{page_num}.svg" type="image/svg+xml" width="230" height="200">
                    <div class="metadata">Vector content from page {page_num}</div>
                    <a href="svg/page_{page_num}.svg" class="button">View Full SVG</a>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add high-resolution section if available
    if "highres" in results:
        highres_count = results["highres"]["count"]
        highres_pages = results["highres"]["pages"]
        dpi = results["highres"]["dpi"]
        
        html += f"""
        <div class="section">
            <h2>High-Resolution Pages ({highres_count})</h2>
            <p>Pages rendered at {dpi} DPI for detailed view</p>
            
            <div class="grid">
"""
        
        # Add card for each high-res page
        for page_num in sorted(highres_pages):
            html += f"""
                <div class="card">
                    <h3>Page {page_num}</h3>
                    <img src="highres/page_{page_num}.png" alt="High-resolution image of page {page_num}" style="max-width: 230px;">
                    <div class="metadata">{dpi} DPI rendering</div>
                    <a href="highres/page_{page_num}.png" class="button">View Full Size</a>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Complete HTML document
    html += """
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html)
    
    return output_path

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Extract content from PDF files including images and vector graphics.')
    parser.add_argument('pdf_path', nargs='?', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output directory (default: auto-generated)')
    parser.add_argument('--images', '-i', action='store_true', help='Extract embedded images')
    parser.add_argument('--svg', '-s', action='store_true', help='Extract vector graphics as SVG')
    parser.add_argument('--highres', '-r', help='Pages to render at high resolution (comma-separated, "all", or "auto")')
    parser.add_argument('--dpi', '-d', type=int, default=300, help='DPI for high-resolution rendering (default: 300)')
    parser.add_argument('--all', '-a', action='store_true', help='Extract all content types')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get PDF path from arguments or prompt
    pdf_path = args.pdf_path
    if not pdf_path:
        pdf_path = input("Enter path to PDF file: ").strip()
        
        # Check if path is valid
        if not pdf_path or not os.path.exists(pdf_path):
            print("Error: Invalid or missing PDF file path")
            return
    
    try:
        # Initialize extractor
        extractor = PDFExtractor(pdf_path, args.output)
        
        # If no specific extraction option is chosen, prompt user
        if not (args.images or args.svg or args.highres or args.all):
            print("\nExtraction options:")
            extract_images = input("Extract embedded images? (y/n, default: y): ").lower() != 'n'
            extract_svg = input("Extract vector graphics as SVG? (y/n, default: y): ").lower() != 'n'
            
            highres_input = input("Pages to render at high resolution (comma-separated, 'all', 'auto', or 'none', default: auto): ")
            if not highres_input or highres_input.lower() == 'auto':
                # Automatically suggest pages that might contain graphs/charts
                suggested_pages = extractor.analyze_content()
                if suggested_pages:
                    print(f"Suggesting pages that might contain graphs/charts: {', '.join(map(str, suggested_pages))}")
                    render_highres = suggested_pages
                else:
                    print("No pages with likely graphs/charts detected")
                    render_highres = None
            elif highres_input.lower() == 'all':
                render_highres = list(range(1, extractor.page_count + 1))
            elif highres_input.lower() != 'none':
                render_highres = [int(p.strip()) for p in highres_input.split(',') if p.strip().isdigit()]
            else:
                render_highres = None
            
            generate_report = input("Generate HTML report? (y/n, default: y): ").lower() != 'n'
        else:
            # Use command-line arguments
            extract_images = args.images or args.all
            extract_svg = args.svg or args.all
            
            if args.highres:
                if args.highres.lower() == 'all':
                    render_highres = list(range(1, extractor.page_count + 1))
                elif args.highres.lower() == 'auto':
                    render_highres = extractor.analyze_content()
                else:
                    render_highres = [int(p.strip()) for p in args.highres.split(',') if p.strip().isdigit()]
            else:
                render_highres = None
            
            generate_report = args.report
        
        # Extract content
        results = extractor.extract_all(
            extract_images=extract_images,
            extract_svg=extract_svg,
            render_highres=render_highres,
            dpi=args.dpi
        )
        
        # Generate HTML report if requested
        if generate_report:
            report_path = os.path.join(extractor.output_dir, "report.html")
            create_html_report(results, report_path)
            print(f"HTML report generated: {report_path}")
            
            # Try to open the report in the default browser
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(report_path))
                print("Opening report in web browser...")
            except:
                print("Could not open browser automatically. Please open the report manually.")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
