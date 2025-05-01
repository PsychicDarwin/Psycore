# Document Preprocessing Pipeline

This module provides a skeleton for the document preprocessing pipeline for the Psycore project. It handles listing files from S3 buckets, with placeholders for document processing functionality that will be implemented later.

## Features

- List files in S3 buckets
- Download documents from S3
- Basic placeholder processing structure
- Upload processed results to S3
- Integration with DynamoDB for document tracking

## Architecture

The preprocessing pipeline leverages the following components from the main Psycore codebase:

- `src.data.s3_handler.S3Handler` - For all S3 operations (listing, downloading, uploading)
- `src.data.db_handler.DynamoHandler` - For storing document metadata and relationships

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure your AWS credentials are configured in your environment or `.env` file.

3. Configure the environment variables:
   - `DOCUMENTS_BUCKET_NAME`: Name of the S3 bucket containing documents
   - `DOCUMENT_TEXT_BUCKET_NAME`: Name of the S3 bucket for storing extracted text
   - `DOCUMENT_IMAGES_BUCKET_NAME`: Name of the S3 bucket for storing images
   - `DOCUMENT_GRAPHS_BUCKET_NAME`: Name of the S3 bucket for storing graphs
   - `DYNAMODB_DOCUMENT_RELATIONSHIPS_TABLE`: Name of the DynamoDB table for document relationships (default: psycore-document-relationships)

## Database Structure

The preprocessing pipeline integrates with DynamoDB to store document information with the following structure:

```json
{
  "document_id": "string",
  "created_at": number,
  "document_s3_link": "string",
  "text_summary_s3_link": "string",
  "graph_s3_link": "string",
  "images": [
    {
      "page_number": number,
      "image_s3_link": "string",
      "text_summary": "string"
    }
  ],
  "metadata": {
    "title": "string",
    "author": "string",
    "created_date": "string"
  }
}
```

## Usage

Run the pipeline with the default settings:
```
python -m preprocessing.main
```

Run the pipeline with custom settings:
```
python -m preprocessing.main --bucket my-bucket --prefix documents/ --limit 10 --clean --verbose
```

### Command-line Arguments

- `--bucket`: Specify the S3 bucket to process (defaults to `DOCUMENTS_BUCKET_NAME` env var)
- `--prefix`: Key prefix for filtering files in the bucket
- `--limit`: Maximum number of files to process
- `--clean`: Clean output buckets (text, images, graphs) and DynamoDB table before processing
- `--verbose`: Enable verbose logging

## Module Structure

- `__init__.py`: Package initialization
- `main.py`: Command-line interface and entry point
- `pipeline.py`: Main preprocessing pipeline implementation
- `processor.py`: Basic document processing skeleton (to be expanded) 