# PDF Extraction Test

This is a test script for the PDF image extraction functionality in the Psycore application. It demonstrates how the `Attachment` class processes PDF files, extracts images, identifies duplicates, and prepares metadata for storage in an S3-like structure.

## Prerequisites

Before running the test, make sure you have the necessary dependencies installed:

```
pip install PyMuPDF pillow
```

## How to Run

1. Ensure the PDF file path in the script is correct:
   ```python
   PDF_PATH = "C:\\Users\\anast\\Documents\\Psycore\\scripting\\jupyter_testing\\22-036458-01_GIS_early_process_evaluation_Accessible_CLIENT_USE.pdf"
   ```

2. Run the test script:
   ```
   python test_pdf_extraction.py
   ```

## What the Test Does

1. Creates an `Attachment` instance for the PDF file
2. Imports the original `Attachment` class from the data folder (`C:\Users\anast\Documents\Darwin\Psycore\src\data\attachments.py`)
3. Extracts images from the PDF (using the implementation in the original file)
4. Detects duplicate images
5. Creates a document structure with metadata and placeholders for S3 links
6. Simulates S3 storage by copying files to local directories:
   - `documents/{document_id}/` for the PDF and associated metadata
   - `document-images/{document_id}/` for extracted images and their text summaries
7. Prints out the metadata, additional data, and updated document structure

## Directory Structure

- `documents/`: Simulated S3 bucket for storing original PDFs and metadata
- `document-images/`: Simulated S3 bucket for storing extracted images and their text summaries

## Example Output

The script will output:
- The metadata and additional data extracted from the PDF
- Information about each image including page number
- Details about duplicate images if any are found
- The structure of the simulated S3 storage

## Integration with Psycore

This test demonstrates how the `Attachment` class processes PDF files and extracts images for storage in S3. The `additional_data` field contains all the necessary information for other components of the Psycore application to use.
