import pytest
from src.data.attachments import Attachment, AttachmentTypes, MAX_LLM_IMAGE_PIXELS
from PIL import Image
import base64
from io import BytesIO
import os

@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()

@pytest.fixture
def temp_image_file(sample_image, tmp_path):
    """Save the sample image to a temporary file."""
    img_path = tmp_path / "test.jpg"
    with open(img_path, "wb") as f:
        f.write(sample_image)
    return str(img_path)

def test_attachment_types_enum():
    """Test that all attachment types are correctly defined."""
    assert AttachmentTypes.IMAGE.value == 1
    assert AttachmentTypes.AUDIO.value == 2
    assert AttachmentTypes.VIDEO.value == 3
    assert AttachmentTypes.FILE.value == 4

def test_attachment_creation():
    """Test basic attachment creation."""
    attachment = Attachment(AttachmentTypes.IMAGE, "test_data")
    assert attachment.attachment_type == AttachmentTypes.IMAGE
    assert attachment.attachment_data == "test_data"
    assert attachment.needsExtraction is False
    assert attachment.metadata is None

def test_image_processing(temp_image_file):
    """Test image attachment processing."""
    attachment = Attachment(AttachmentTypes.IMAGE, temp_image_file, True)
    attachment.extract()
    
    assert attachment.needsExtraction is False
    assert attachment.metadata is not None
    assert "width" in attachment.metadata
    assert "height" in attachment.metadata
    
    # Test if the image was resized to MAX_LLM_IMAGE_PIXELS
    assert attachment.metadata["width"] == MAX_LLM_IMAGE_PIXELS
    assert attachment.metadata["height"] == MAX_LLM_IMAGE_PIXELS
    
    # Test if the image data is base64 encoded
    try:
        decoded = base64.b64decode(attachment.attachment_data)
        Image.open(BytesIO(decoded))  # Should not raise an error
    except Exception:
        pytest.fail("Image data is not properly base64 encoded")

def test_invalid_image_processing():
    """Test handling of invalid image data."""
    attachment = Attachment(AttachmentTypes.IMAGE.value, "invalid_path.jpg", True)
    attachment.extract()
    assert attachment.needsExtraction is True  # Should fail to extract

@pytest.mark.skipif(not os.path.exists("path/to/test.wav"), reason="Test audio file not available")
def test_audio_processing():
    """Test audio attachment processing."""
    test_audio = "path/to/test.wav"
    attachment = Attachment(AttachmentTypes.AUDIO, test_audio, True)
    attachment.extract()
    
    assert attachment.needsExtraction is False
    assert attachment.metadata is not None
    assert "duration" in attachment.metadata
    
    # Test if the audio data is base64 encoded
    try:
        base64.b64decode(attachment.attachment_data)
    except Exception:
        pytest.fail("Audio data is not properly base64 encoded")

@pytest.mark.skip(reason="Video processing not yet implemented")
def test_video_processing():
    """Test that video processing raises NotImplementedError."""
    attachment = Attachment(AttachmentTypes.VIDEO, "test.mp4", True)
    attachment.extract()

    assert attachment.needsExtraction is True  # Should remain True for videos as not implemented

@pytest.mark.skip(reason="File processing not yet implemented")
def test_file_processing():
    """Test basic file attachment processing."""
    attachment = Attachment(AttachmentTypes.FILE, "test.txt", True)
    attachment.extract()
    assert attachment.needsExtraction is False

def test_unknown_type_processing():
    """Test handling of unknown attachment type."""
    attachment = Attachment(8, "test_data", True)
    attachment.extract()
    assert attachment.needsExtraction is True  

@pytest.mark.parametrize("attachment_type,data,needs_extraction", [
    (AttachmentTypes.IMAGE.value, "test.jpg", True),
    (AttachmentTypes.AUDIO.value, "test.mp3", True),
    (AttachmentTypes.VIDEO.value, "test.mp4", True),
    (AttachmentTypes.FILE.value, "test.txt", True),
])
def test_attachment_initialization_variations(attachment_type, data, needs_extraction):
    """Test attachment creation with various types and configurations."""
    attachment = Attachment(attachment_type, data, needs_extraction)
    assert attachment.attachment_type == attachment_type
    assert attachment.attachment_data == data
    assert attachment.needsExtraction == needs_extraction
