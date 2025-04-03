from enum import Enum
import base64
from PIL import Image, ImageFile
from io import BytesIO
from pdf2image import convert_from_path
import ffmpeg  # Install with: pip install imageio[ffmpeg]
from src.data.common_types import AttachmentTypes, MAX_LLM_IMAGE_PIXELS, TEMPFILE
import os
from src.data.s3_manager import S3Manager

class Attachment:
    def __init__(self, attachment_type: AttachmentTypes, attachment_data: str, needsExtraction: bool = False):
        self.attachment_type = attachment_type
        self.attachment_data = attachment_data
        self.needsExtraction = needsExtraction
        self.prompt_mapping = None # This is the mapping name, that goes through prompt template, it is defined when processed by a 
        self.metadata = None
        self.extra_attachments = None

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
            
    def pop_extra_attachments(self):
        if self.extra_attachments is None:
            return []
        attachments = self.extra_attachments
        self.extra_attachments = None
        return attachments
        
    def _process_image(self):
        try: 
            with Image.open(self.attachment_data) as img:
                self.attachment_data, self.metadata = self.imageFileToBase64(img)
        except Exception as e:
            raise FailedExtraction(self, str(e))
        
    def imageFileToBase64(self, image: ImageFile) -> str:
        img = image.convert("RGB")
        if (MAX_LLM_IMAGE_PIXELS != None):
            img = img.resize((MAX_LLM_IMAGE_PIXELS, MAX_LLM_IMAGE_PIXELS))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return (base64.b64encode(buffer.getvalue()).decode('utf-8'), {"width": img.width, "height": img.height})
        
    def _process_audio(self):
        # Gets file type
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

    def _process_video(self):
        raise NotImplementedError("Video processing not yet implemented")
        
    def _process_file(self):
        # Import here to break circular dependency
        from src.data.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(self)
        extractor.process_pages()

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
        i = 0
        while i < len(attachments):
            extra_attachments = attachments[i].pop_extra_attachments()
            if extra_attachments:
                attachments.extend(extra_attachments)
            i += 1
        return attachments


class S3Attachment(Attachment):
    
    def __init__(self, attachment_data: str, needsExtraction: bool = False, s3_key: str = None, bucket_name: str = None, attachment_type: AttachmentTypes = None,):
        self.s3_key = s3_key
        self.bucket_name = bucket_name
        self.temp_file_path = None
        
        # If this is an S3 file, download it temporarily
        if s3_key and bucket_name:
            self._download_from_s3()
        else:
            self.temp_file_path = attachment_data

        # Determines the attachment type from the file path
        if attachment_type is None:
            self.attachment_type = AttachmentTypes.from_filename(self.temp_file_path)
        else:
            self.attachment_type = attachment_type
            
        # Initializes a parent class with the downloaded file path
        super().__init__(attachment_type, self.temp_file_path, needsExtraction)
        
    def _download_from_s3(self):
        """Download the file from S3 to a temporary location"""
        try:
            # Creates temporary directory if it doesn't exist
            os.makedirs(os.path.dirname(TEMPFILE), exist_ok=True)
            
            # Generates a unique temporary file path
            self.temp_file_path = f"{TEMPFILE}_{os.path.basename(self.s3_key)}"
            
            # Download the file
            s3_manager = S3Manager(bucket_name=self.bucket_name)
            if not s3_manager.download_file(self.s3_key, self.temp_file_path):
                raise FailedExtraction(self, f"Failed to download file {self.s3_key} from S3")
                
            # Verifies that the file exists and is readable
            if not os.path.exists(self.temp_file_path):
                raise FailedExtraction(self, f"Downloaded file {self.temp_file_path} does not exist")
                
        except Exception as e:
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                try:
                    os.remove(self.temp_file_path)
                except:
                    pass
            raise FailedExtraction(self, f"Failed to download file from S3: {str(e)}")
            
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.remove(self.temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file {self.temp_file_path}: {str(e)}")


class FailedExtraction(Exception):
    def __init__(self, attachment: Attachment, message: str):
        super().__init__(f"""
Failed to extract attachment
Attachment type: {attachment.attachment_type.name}
Error: {message}
        """)
        self.attachment = attachment
