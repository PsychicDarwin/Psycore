# S3 Handler Tests

This directory contains tests for the S3Handler class which is responsible for handling document storage in S3 buckets.

## Test Types

There are two types of tests for the S3Handler:

1. **Mock Tests** (`test_s3_handler.py`): These tests use mocked S3 client with unittest.mock to test functionality without actual AWS interactions. They run quickly and don't require AWS credentials.

2. **Real AWS Tests** (`test_s3_handler_real.py`): These tests interact with actual AWS S3 services to verify real-world functionality. They require valid AWS credentials and bucket configuration.

## Running the Tests

### Mock Tests

To run the mock tests (which don't require AWS credentials):

```bash
pytest tests/data/test_s3_handler.py -v
```

### Real AWS Tests

To run tests that interact with real AWS services, you need:

1. Valid AWS credentials in your environment or `.env` file
2. Properly configured bucket names

Set the following environment variables in your `.env` file:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=your_region

DOCUMENTS_BUCKET_NAME=your-documents-bucket
DOCUMENT_TEXT_BUCKET_NAME=your-text-bucket
DOCUMENT_IMAGES_BUCKET_NAME=your-images-bucket
DOCUMENT_GRAPHS_BUCKET_NAME=your-graphs-bucket
```

Then run:

```bash
pytest tests/data/test_s3_handler_real.py -v --real-aws
```

The `--real-aws` flag is required to run tests marked with `@pytest.mark.real_aws`.

## Test Structure

Both test suites test similar functionality, but with different approaches:

- Document upload/download
- Text content upload/download
- Image upload/download
- Graph upload/download
- Listing document files
- Deleting document files
- Checking if files exist
- Generating presigned URLs

## Notes about Real AWS Tests

The real AWS tests:

1. Will create buckets if they don't exist
2. Will generate random document IDs to avoid conflicts
3. Will clean up test files after each test
4. Will only run if explicitly enabled with `--real-aws` flag
5. Are skipped by default to avoid accidental AWS charges

**CAUTION**: Running real AWS tests will create and delete objects in your S3 buckets and may incur AWS charges. 