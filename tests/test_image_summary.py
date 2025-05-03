from src.data.attachments import Attachment, AttachmentTypes
from pathlib import Path

def test_image_summary():
    # Step 1: Use a sample image file path
    image_path = "/Users/arun/Desktop/Darwin/Psycore/jupyter_testing/404.jpg"  # Replace with a real path

    # Step 2: Create Attachment with needsExtraction=True
    attachment = Attachment(
        attachment_type=AttachmentTypes.IMAGE,
        attachment_data=image_path,
        needsExtraction=True
    )

    # Step 3: Extract and process the image
    attachment.extract()

    # Step 4: Generate the text summary
    summary = attachment._text_summary()

    # Step 5: Output the result
    print("=== Image Summary ===")
    print(summary)
    print("=====================")

if __name__ == "__main__":
    test_image_summary()
