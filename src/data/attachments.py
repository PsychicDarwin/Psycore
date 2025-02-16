from enum import Enum
import base64
from PIL import Image
from io import BytesIO
import ffmpeg  # Install with: pip install imageio[ffmpeg]


MAX_LLM_IMAGE_PIXELS = 512 
TEMPFILE = "tempfile"


class AttachmentTypes(Enum):
    IMAGE = 1
    AUDIO = 2
    VIDEO = 3
    FILE =  4

class Attachment:
    def __init__(self, attachment_type: AttachmentTypes, attachment_data: str, needsExtraction: bool = False):
        self.attachment_type = attachment_type
        self.attachment_data = attachment_data
        self.needsExtraction = needsExtraction
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
                print(f"Error extracting attachment: {e}")
                self.needsExtraction = True  # Keep extraction status in case of failure

    def _process_image(self):
        with Image.open(self.attachment_data) as img:
            img = img.convert("RGB")
            if (MAX_LLM_IMAGE_PIXELS != None):
                img = img.resize((MAX_LLM_IMAGE_PIXELS, MAX_LLM_IMAGE_PIXELS))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            self.attachment_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            self.metadata = {"width": img.width, "height": img.height}
        
    def _process_audio(self):
        # Get file type
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
        pass

    def _process_video(self):
        raise NotImplementedError("Video processing not yet implemented")
        

    def _process_file(self):
        # This will need work for different file types, like pdfs, csvs, etc.
        pass
