from enum import Enum
import base64
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path
import ffmpeg  # Install with: pip install imageio[ffmpeg]
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt


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

    def _process_video(self):
        raise NotImplementedError("Video processing not yet implemented")
        

    def _process_file(self):
        # This will need work for different file types, like pdfs, csvs, etc.
        pass

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
        

