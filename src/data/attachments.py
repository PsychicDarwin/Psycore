from enum import Enum
import base64
from PIL import Image
from io import BytesIO
import fitz  # PyMuPDF
import ffmpeg  
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt
# import whisper  # Import commented out to avoid dependency issue

import os
import uuid
import hashlib
import numpy as np
from datetime import datetime

MAX_LLM_IMAGE_PIXELS = 512 
TEMPFILE = "tempfile"


class AttachmentTypes(Enum):
    IMAGE = 1
    AUDIO = 2
    VIDEO = 3
    FILE =  4

    @staticmethod
    def from_filename(filename: str) -> 'AttachmentTypes':
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            return AttachmentTypes.IMAGE
        elif filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".flac"):
            return AttachmentTypes.AUDIO
        elif filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
            return AttachmentTypes.VIDEO
        else:
            return AttachmentTypes.FILE
        
class Attachment:
    def __init__(self, attachment_type: AttachmentTypes, attachment_data: str, needsExtraction: bool = False):
        self.attachment_type = attachment_type
        self.attachment_data = attachment_data
        self.needsExtraction = needsExtraction  
        self.prompt_mapping = None # This is the mapping name, that goes through prompt template, it is defined when processed by a 
        self.additional_data = {} # This is the additional data that is passed to the llm, like the metadata of the image, etc.
        self.metadata = None

    def extract(self):
        if self.needsExtraction:
            try:
                self.needsExtraction = False  # Successfully extracted
                if self.attachment_type == AttachmentTypes.IMAGE:
                    self._process_image()
                elif self.attachment_type == AttachmentTypes.AUDIO:
                    self._process_audio()
                elif self.attachment_type == AttachmentTypes.VIDEO:
                    self._process_video()
                elif self.attachment_type == AttachmentTypes.FILE:
                    self._process_file()
                else:
                    self.needsExtraction = True  # Unsupported type
                    return
            except Exception as e:
                self.needsExtraction = True  # Keep extraction status in case of failure
                raise FailedExtraction(self, str(e))

    def _process_image(self):
        try: 
            with Image.open(self.attachment_data) as img:
                img = img.convert("RGB")
                if (MAX_LLM_IMAGE_PIXELS != None):
                    img = img.resize((MAX_LLM_IMAGE_PIXELS, MAX_LLM_IMAGE_PIXELS))
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                self.attachment_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                self.metadata = {"width": img.width, "height": img.height}
        except Exception as e:
            raise FailedExtraction(self, str(e))
    

    def _process_graph(prompt):
        # Ensure necessary NLTK downloads

        """Convert a string into a JSON-based knowledge graph."""
        words = word_tokenize(prompt.lower())
        stop_words = set(stopwords.words("english"))

        # Filter meaningful words
        key_words = [word for word in words if word.isalnum() and word not in stop_words]

        # Construct graph data structure
        graph = {
            "nodes": [{"id": word} for word in key_words],
            "edges": [{"source": key_words[i], "target": key_words[i + 1]} for i in range(len(key_words) - 1)]
        }

            # Create graph
        visual_graph = nx.Graph()
        
        # Add words as nodes
        for word in key_words:
            visual_graph.add_node(word)

        # Connect words based on proximity
        for i in range(len(key_words) - 1):
            visual_graph.add_edge(key_words[i], key_words[i + 1])
        
        print(prompt)
        return json.dumps(graph, indent=4), visual_graph
    
        
    def _process_audio(self):
        # Get file type
        try:
            file_type = self.attachment_data.split('.')[-1]
            filepath = self.attachment_data
            if file_type != "wav":
                # Convert to wav
                ffmpeg.input(self.attachment_data).output(f"{TEMPFILE}.wav").run()
                with open(f"{TEMPFILE}.wav", "rb") as f:
                    self.attachment_data = base64.b64encode(f.read()).decode('utf-8')
                filepath = f"{TEMPFILE}.wav"
            # Get duration
            duration = ffmpeg.probe(f"{filepath}")['format']['duration']
            self.metadata = {"duration": duration}
        except Exception as e:
            raise FailedExtraction(self, str(e))
    
    def _convert_audio_to_text(self):
        """Transcribes audio to text using OpenAI's Whisper."""
        if self.attachment_type != AttachmentTypes.AUDIO:
            raise ValueError("Audio-to-text conversion is only supported for audio attachments.")
        
        # This functionality is commented out to avoid dependency on whisper
        raise NotImplementedError("Audio transcription is currently disabled due to missing whisper dependency")

    def _process_video(self):
        raise NotImplementedError("Video processing not yet implemented")
        
    def _process_file(self):
        """
        Process different file types, with special handling for PDFs
        
        For PDFs, this extracts:  
        - Document structure and metadata
        - Embedded images (raster graphics)
        - Vector graphics (drawings)
        """
        if self.attachment_data.lower().endswith('.pdf'):
            try:
                # Initialize document structure following the schema
                document_id = str(uuid.uuid4())
                created_at = int(datetime.now().timestamp())
                
                document_structure = {
                    "document_id": document_id,
                    "created_at": created_at,
                    "document_s3_link": "",  # Will be populated elsewhere
                    "text_summary_s3_link": "",  # Will be populated elsewhere
                    "graph_s3_link": "",  # Will be populated elsewhere
                    "images": [],
                    "metadata": {
                        "title": os.path.basename(self.attachment_data),
                        "author": "Unknown",  # Can be updated later
                        "created_date": datetime.now().strftime("%Y-%m-%d")
                    }
                }
                
                # Extract and process images from PDF
                self._extract_images_from_pdf(self.attachment_data, document_structure)
                
                # Store the document structure in additional_data for use by other team members
                self.additional_data["document_structure"] = document_structure
                
                # Update metadata with document structure
                if not self.metadata:
                    self.metadata = {}
                    
                self.metadata["document_structure"] = document_structure
                
            except Exception as e:
                raise FailedExtraction(self, f"Failed to process PDF: {str(e)}")
        else:
          # Handle csv, txt, etc.
          pass
            
    def _hash_image(self, image_data, hash_algorithm='sha256'):
        """
        Generate a hash for image data.
        
        Args:
            image_data: Binary image data
            hash_algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', etc.)
        
        Returns:
            Hash string of the image
        """
        try:
            if hash_algorithm == 'md5':
                hash_obj = hashlib.md5(image_data)
            elif hash_algorithm == 'sha1':
                hash_obj = hashlib.sha1(image_data)
            elif hash_algorithm == 'sha256':
                hash_obj = hashlib.sha256(image_data)
            else:
                hash_obj = hashlib.sha256(image_data)  # Default to sha256
            
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"Error calculating hash: {str(e)}")
            return f"error_hash_{uuid.uuid4()}"

    def _extract_images_from_pdf(self, pdf_path, document_structure):
        """
        Extract images and vector graphics from PDF while avoiding duplicates, and update document structure.
        Uses PyMuPDF (fitz) to extract images and vector graphics directly from the PDF structure.
        
        Args:
            pdf_path: Path to the PDF file
            document_structure: Dictionary containing document structure (will be modified)
        """
        try:
            # Dictionary to store image hashes to avoid duplicates
            hash_to_images = {}
            
            # Open the PDF
            pdf = fitz.open(pdf_path)
            
            # Create a temporary directory for images
            doc_id = document_structure["document_id"]
            temp_dir = f"temp_images_{doc_id}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Process each page
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                image_list = page.get_images()
                
                # Process each image on the page
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Hash the image data
                        img_hash = self._hash_image(image_bytes)
                        
                        # Check if this is a duplicate
                        is_duplicate = img_hash in hash_to_images
                        
                        # Create image info with page number (1-based)
                        image_info = {
                            "page_number": page_num + 1,
                            "image_index": img_index,
                            "hash": img_hash,
                            "is_duplicate": is_duplicate,
                            "format": image_ext.upper()
                        }
                        
                        if is_duplicate:
                            # This is a duplicate - reference the original image
                            original_info = hash_to_images[img_hash]
                            original_page = original_info["page_number"]
                            
                            # Find the original image in the document structure
                            for entry in document_structure["images"]:
                                if entry["page_number"] == original_page:
                                    # Add a new entry referencing the same S3 links
                                    document_structure["images"].append({
                                        "page_number": page_num + 1,
                                        "image_s3_link": entry["image_s3_link"],
                                        "image_text_summary_s3_link": entry["image_text_summary_s3_link"]
                                    })
                                    break
                            
                            # Add to additional_data for LLM
                            self.additional_data[f"image_{page_num+1}_{img_index}"] = {
                                "page_number": page_num + 1,
                                "image_index": img_index,
                                "is_duplicate": True,
                                "original_page": original_page,
                                "hash": img_hash,
                                "format": image_ext.upper()
                            }
                        else:
                            # This is a new image
                            hash_to_images[img_hash] = image_info
                            
                            # Save the image to a temporary file
                            img_temp_path = os.path.join(temp_dir, f"page_{page_num+1}_img_{img_index}.{image_ext}")
                            with open(img_temp_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            
                            # Create standardized placeholders for S3 links
                            image_s3_link = f"__S3_IMAGE_LINK__{doc_id}__{page_num+1}_{img_index}__"
                            summary_s3_link = f"__S3_SUMMARY_LINK__{doc_id}__{page_num+1}_{img_index}__"
                            
                            # Add to document structure
                            document_structure["images"].append({
                                "page_number": page_num + 1,
                                "image_s3_link": image_s3_link,
                                "image_text_summary_s3_link": summary_s3_link
                            })
                            
                            # Add to additional_data for LLM
                            width = base_image.get("width", 0)
                            height = base_image.get("height", 0)
                            self.additional_data[f"image_{page_num+1}_{img_index}"] = {
                                "page_number": page_num + 1,
                                "image_index": img_index,
                                "dimensions": f"{width}x{height}",
                                "is_duplicate": False,
                                "hash": img_hash,
                                "format": image_ext.upper(),
                                "temp_path": img_temp_path
                            }
                    except Exception as e:
                        print(f"Error processing image {img_index} on page {page_num+1}: {str(e)}")
                        continue
            
            # Store duplicate information in additional_data
            duplicate_groups = {}
            for hash_val, info in hash_to_images.items():
                # Find all instances of this hash
                instances = []
                for key, img_info in self.additional_data.items():
                    if isinstance(img_info, dict) and img_info.get("hash") == hash_val:
                        instances.append(img_info["page_number"])
                
                if len(instances) > 1:
                    duplicate_groups[hash_val] = instances
            
            if duplicate_groups:
                self.additional_data["duplicate_images"] = duplicate_groups
            
            # Extract vector graphics if available
            vector_graphics_dir = f"temp_vectors_{doc_id}"
            os.makedirs(vector_graphics_dir, exist_ok=True)
            
            vector_count = self._extract_vector_graphics(pdf, vector_graphics_dir, document_structure)
            
            # Store summary stats
            self.additional_data["image_stats"] = {
                "total_pages": len(pdf),
                "total_images": len(hash_to_images) + sum(1 for img in self.additional_data.values() 
                                                          if isinstance(img, dict) and img.get("is_duplicate", False)),
                "unique_images": len(hash_to_images),
                "vector_graphics": vector_count
            }
            
            # Close the PDF
            pdf.close()
            
            # Clean up the temporary directories after processing
            # Comment these out if you want to keep the extracted images/vectors
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(vector_graphics_dir, ignore_errors=True)
            
        except Exception as e:
            raise FailedExtraction(self, f"Failed to extract images from PDF: {str(e)}")
            
    def _extract_vector_graphics(self, pdf, output_dir, document_structure):
        """
        Extract vector graphics (drawings) from PDF and save them as SVG or JSON.
        
        Args:
            pdf: Open PyMuPDF document
            output_dir: Directory to save the vector graphics
            document_structure: Document structure dictionary to update
        
        Returns:
            Total count of vector graphics extracted
        """
        try:
            # Add vector_graphics array to document structure if it doesn't exist
            if "vector_graphics" not in document_structure:
                document_structure["vector_graphics"] = []
                
            vector_count = 0
            doc_id = document_structure["document_id"]
            
            # Process each page
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                
                # Extract vector drawings
                try:
                    drawings = page.get_drawings()
                    if drawings:
                        # Save the vector data as JSON
                        json_path = os.path.join(output_dir, f"page_{page_num+1}_vectors.json")
                        with open(json_path, "w") as f:
                            import json
                            json.dump(drawings, f, indent=2)
                        
                        # Try to convert to SVG if possible
                        svg_path = os.path.join(output_dir, f"page_{page_num+1}_vectors.svg")
                        self._convert_drawings_to_svg(drawings, svg_path, page.rect.width, page.rect.height)
                        
                        # Create standardized placeholders for S3 links
                        json_s3_link = f"__S3_VECTOR_JSON_LINK__{doc_id}__{page_num+1}__"
                        svg_s3_link = f"__S3_VECTOR_SVG_LINK__{doc_id}__{page_num+1}__"
                        
                        # Add to document structure
                        document_structure["vector_graphics"].append({
                            "page_number": page_num + 1,
                            "vector_json_s3_link": json_s3_link,
                            "vector_svg_s3_link": svg_s3_link,
                            "vector_count": len(drawings)
                        })
                        
                        # Add to additional_data
                        self.additional_data[f"vector_{page_num+1}"] = {
                            "page_number": page_num + 1,
                            "vector_count": len(drawings),
                            "temp_json_path": json_path,
                            "temp_svg_path": svg_path
                        }
                        
                        vector_count += len(drawings)
                except Exception as e:
                    print(f"Error extracting vector graphics from page {page_num+1}: {str(e)}")
            
            return vector_count
        except Exception as e:
            print(f"Error in vector graphics extraction: {str(e)}")
            return 0
            
    def _convert_drawings_to_svg(self, drawings, output_path, width, height):
        """
        Convert PyMuPDF drawings to SVG format
        
        Args:
            drawings: List of drawing dictionaries from page.get_drawings()
            output_path: Path to save the SVG file
            width: Page width
            height: Page height
        """
        try:
            # Basic SVG structure
            svg = f'<?xml version="1.0" encoding="UTF-8"?>\n'
            svg += f'<svg xmlns="http://www.w3.org/2000/svg" '
            svg += f'width="{width}pt" height="{height}pt" '
            svg += f'viewBox="0 0 {width} {height}">\n'
            
            # Process each drawing
            for i, drawing in enumerate(drawings):
                svg += f'  <g id="drawing_{i}">{self._convert_drawing_to_svg_path(drawing)}</g>\n'
            
            svg += '</svg>'
            
            # Write SVG to file
            with open(output_path, "w") as f:
                f.write(svg)
                
            return True
        except Exception as e:
            print(f"Error converting drawings to SVG: {str(e)}")
            return False
            
    def _convert_drawing_to_svg_path(self, drawing):
        """
        Convert a single PyMuPDF drawing to SVG path element
        
        Args:
            drawing: Dictionary containing drawing information
        
        Returns:
            SVG path element as string
        """
        try:
            # Extract color information
            stroke_color = drawing.get("stroke_color")
            fill_color = drawing.get("fill_color")
            
            # Convert colors to SVG format
            stroke = "none"
            if stroke_color:
                r, g, b = stroke_color
                stroke = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                
            fill = "none"
            if fill_color:
                r, g, b = fill_color
                fill = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            
            # Create path string from commands
            path_data = ""
            for item in drawing.get("items", []):
                cmd = item[0]
                points = item[1:]
                
                if cmd == "m":  # moveto
                    path_data += f"M{points[0]} {points[1]} "
                elif cmd == "l":  # lineto
                    path_data += f"L{points[0]} {points[1]} "
                elif cmd == "c":  # curveto (cubic bezier)
                    x1, y1 = points[0], points[1]
                    x2, y2 = points[2], points[3]
                    x3, y3 = points[4], points[5]
                    path_data += f"C{x1} {y1} {x2} {y2} {x3} {y3} "
                elif cmd == "f":  # close path
                    path_data += "Z "
            
            # Create SVG path element
            svg_path = f'<path d="{path_data}" fill="{fill}" stroke="{stroke}" '
            
            # Add stroke width if available
            if "width" in drawing:
                svg_path += f'stroke-width="{drawing["width"]}" '
            
            # Close the path tag
            svg_path += '/>'  
            
            return svg_path
        except Exception as e:
            print(f"Error converting drawing to SVG path: {str(e)}")
            return f'<text x="10" y="20">Error: {str(e)}</text>'

    @staticmethod
    def attachmentListMapping(attachments: list, attachment_constant: str = "attachment") -> dict:
        mappings = {}
        for i, attachment in enumerate(attachments):
            attachment.prompt_mapping = f"{attachment_constant}{i}"
            mappings[attachment.prompt_mapping] = attachment.attachment_data
        return mappings


    @staticmethod
    def extractAttachmentList(attachments: list):
        i = 0
        while i < len(attachments):
            attachment = attachments[i]
            try:
                attachment.extract()
            except FailedExtraction:
                # If we can't extract the attachment, remove it from the list as we can't use it
                attachments.pop(i)
                continue 
            # This shouldn't falsely trigger as continue prevents it from being called if failed extraction
            if attachment.needsExtraction:
                attachments.pop(i)
            else:
                i += 1
        return attachments

    

class FailedExtraction(Exception):
    def __init__(self, attachment: Attachment, message: str):
        super().__init__(f"""
Failed to extract attachment
Attachment type: {attachment.attachment_type.name}
Error: {message}
        """)
        self.attachment = attachment
        

