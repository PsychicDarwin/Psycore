from enum import Enum
import base64
from PIL import Image, ImageFile
from io import BytesIO
import os
import tempfile

MAX_LLM_IMAGE_PIXELS = 512 
TEMPFILE = os.path.join(tempfile.gettempdir(), "psycore_temp")

class AttachmentTypes(Enum):
    IMAGE = 1
    AUDIO = 2
    VIDEO = 3
    FILE = 4

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
