from enum import Enum
import hashlib
import fitz # PyMuPDF
from src.data.attachments import Attachment, AttachmentTypes, S3Attachment
class HashAlgorithms(Enum):
    SHA256 = 1
    SHA1 = 2
    MD5 = 3

class PDFExtractor():
    def __init__(self, attachment: Attachment):
        self.attachment = attachment
        if type(attachment) == S3Attachment:
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
        text = ""
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
        for image in self.image_hashes.values():
            if not self.local:
                i = S3Attachment(AttachmentTypes.IMAGE, image, False, self.attachment.s3_key)
            else:
                i = Attachment(AttachmentTypes.IMAGE, image, False)
            i.attachment_data, i.metadata = i.imageFileToBase64(i.attachment_data)
        self.attachment.extra_attachments = list(self.image_hashes.values())
        # Extract all text from the PDF
        self.attachment.attachment_data = text
