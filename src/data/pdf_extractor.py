from enum import Enum
import hashlib
import fitz  # PyMuPDF
from src.data.common_types import AttachmentTypes
from PIL import Image
from io import BytesIO

class HashAlgorithms(Enum):
    SHA256 = 1
    SHA1 = 2
    MD5 = 3

class PDFExtractor():
    def __init__(self, attachment):
        # Imports S3Attachment after the class is initialized to avoid circular dependency
        from src.data.attachments import S3Attachment 
        self.attachment = attachment
        if isinstance(attachment, S3Attachment):
            self.local = False
        else:
            self.local = True
        self.pdf = fitz.open(self.attachment.attachment_data)
        self.image_hashes = {}
        self.image_count = 0

    def close(self):
        """Close the PDF file to release system resources"""
        if hasattr(self, 'pdf') and self.pdf:
            self.pdf.close()

    @staticmethod
    def hash_image(image_data, hash_algorithm: HashAlgorithms = HashAlgorithms.MD5):
        if hash_algorithm == HashAlgorithms.SHA256:
            hash_obj = hashlib.sha256(image_data)
        elif hash_algorithm == HashAlgorithms.SHA1:
            hash_obj = hashlib.sha1(image_data)
        elif hash_algorithm == HashAlgorithms.MD5:
            hash_obj = hashlib.md5(image_data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
        return hash_obj.hexdigest()

    def process_pages(self):
        from src.data.attachments import Attachment, S3Attachment  # Import here to break circular dependency
        
        try:
            text = ""
            extracted_images = []
            page_texts = []  # Stores text from each page
            
            # Processes each page for text and images
            for page_num in range(self.pdf.page_count):
                page = self.pdf.load_page(page_num)
                page_text = page.get_text()
                text += page_text
                page_texts.append(page_text)  # Store text for this page
                
                image_list = page.get_images(full=True)
                
                for img_i, img in enumerate(image_list):
                    xref = img[0]
                    base_image = self.pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_hash = PDFExtractor.hash_image(image_bytes)
                    
                    if img_hash not in self.image_hashes:
                        # Store both image bytes and page number
                        self.image_hashes[img_hash] = {
                            "bytes": image_bytes,
                            "page_num": page_num
                        }
                        self.image_count += 1
            
            # Creates attachments for each unique image
            for img_hash, img_data in self.image_hashes.items():
                try:
                    image_bytes = img_data["bytes"]
                    page_num = img_data["page_num"]
                    
                    # Create proper attachment objects for each image
                    if not self.local:
                        img_attachment = S3Attachment(
                            attachment_data=image_bytes,  # First argument should be attachment_data
                            needsExtraction=False,
                            s3_key=self.attachment.s3_key,
                            bucket_name=self.attachment.bucket_name
                        )
                    else:
                        img_attachment = Attachment(
                            attachment_type=AttachmentTypes.IMAGE,
                            attachment_data=image_bytes,
                            needsExtraction=False
                        )
                    
                    # Process the image bytes into base64
                    img = Image.open(BytesIO(image_bytes))
                    img_attachment.attachment_data, img_attachment.metadata = img_attachment.imageFileToBase64(img)
                    
                    # Add page text to metadata
                    if img_attachment.metadata is None:
                        img_attachment.metadata = {}
                    img_attachment.metadata.update({
                        "page_number": page_num + 1,  # Convert to 1-based page numbers
                        "page_text": page_texts[page_num]
                    })
                    
                    extracted_images.append(img_attachment)
                except Exception as e:
                    print(f"Failed to process image: {str(e)}")
                    continue
            
            # Sets the extracted images as extra attachments
            self.attachment.extra_attachments = extracted_images
            
            # Sets the PDF text as the main attachment data
            self.attachment.attachment_data = text
            
            return {"text": text, "image_count": self.image_count}
        finally:
            # Always close the PDF file when done
            self.close()
