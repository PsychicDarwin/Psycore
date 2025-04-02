from enum import Enum
import hashlib
import fitz  # PyMuPDF
from src.data.common_types import AttachmentTypes

class HashAlgorithms(Enum):
    SHA256 = 1
    SHA1 = 2
    MD5 = 3

class PDFExtractor():
    def __init__(self, attachment):
        self.attachment = attachment
        from src.data.attachments import S3Attachment  # Import here to break circular dependency
        if isinstance(attachment, S3Attachment):
            self.local = False
        else:
            self.local = True
        self.pdf = fitz.open(self.attachment.attachment_data)
        self.image_hashes = {}
        self.image_count = 0

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
        from PIL import Image
        from io import BytesIO
        
        text = ""
        extracted_images = []
        
        # Process each page for text and images
        for page_num in range(self.pdf.page_count):
            page = self.pdf.load_page(page_num)
            text += page.get_text()
            image_list = page.get_images(full=True)
            
            for img_i, img in enumerate(image_list):
                xref = img[0]
                base_image = self.pdf.extract_image(xref)
                image_bytes = base_image["image"]
                img_hash = PDFExtractor.hash_image(image_bytes)
                
                if img_hash not in self.image_hashes:
                    self.image_hashes[img_hash] = image_bytes
                    self.image_count += 1
        
        # Create attachments for each unique image
        for image_bytes in self.image_hashes.values():
            try:
                # Create proper attachment objects for each image
                if not self.local:
                    img_attachment = S3Attachment(AttachmentTypes.IMAGE, image_bytes, False, self.attachment.s3_key)
                else:
                    img_attachment = Attachment(AttachmentTypes.IMAGE, image_bytes, False)
                
                # Process the image bytes into base64
                img = Image.open(BytesIO(image_bytes))
                img_attachment.attachment_data, img_attachment.metadata = img_attachment.imageFileToBase64(img)
                extracted_images.append(img_attachment)
            except Exception as e:
                print(f"Failed to process image: {str(e)}")
                continue
        
        # Set the extracted images as extra attachments
        self.attachment.extra_attachments = extracted_images
        
        # Set the PDF text as the main attachment data
        self.attachment.attachment_data = text
        
        return {"text": text, "image_count": self.image_count}
