"""
Preprocessing Pipeline Main Entry Point

This script serves as the entry point for the document preprocessing pipeline.
It handles command-line arguments and executes the pipeline.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocessing')

# Add parent directory to path to allow local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.pipeline import PreprocessingPipeline

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Preprocessing Pipeline")
    
    parser.add_argument(
        "--bucket",
        help="S3 bucket to process (defaults to DOCUMENTS_BUCKET_NAME env var)",
        default=None
    )
    
    parser.add_argument(
        "--prefix",
        help="Key prefix for filtering files",
        default=""
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of files to process",
        default=None
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output buckets before processing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the preprocessing pipeline."""
    args = parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    try:
        # Initialize the pipeline
        pipeline = PreprocessingPipeline()
        
        logger.info("Starting preprocessing pipeline")
        logger.info(f"Bucket: {args.bucket or 'Using default from env vars'}")
        logger.info(f"Prefix: {args.prefix or 'No prefix'}")
        logger.info(f"Limit: {args.limit or 'No limit'}")
        if args.clean:
            logger.info("Will clean output buckets before processing")
        
        # Check DynamoDB integration status
        if hasattr(pipeline, 'db_handler') and pipeline.db_handler:
            logger.info("DynamoDB integration: ENABLED")
            logger.info(f"DynamoDB table: {pipeline.db_handler.table_name}")
        else:
            logger.warning("DynamoDB integration: DISABLED")
        
        # Process documents
        results = pipeline.process_documents(
            bucket_name=args.bucket,
            prefix=args.prefix,
            limit=args.limit,
            clean_output=args.clean
        )
        
        # Summarize results
        successful = sum(1 for r in results if r.get('status') == 'success')
        partial = sum(1 for r in results if r.get('status') == 'partial')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        dynamodb_entries = sum(1 for r in results if r.get('dynamodb_entry'))
        
        logger.info("Preprocessing pipeline completed")
        logger.info(f"Processed {len(results)} files")
        logger.info(f"Success: {successful}, Partial: {partial}, Failed: {failed}")
        if dynamodb_entries:
            logger.info(f"Added {dynamodb_entries} entries to DynamoDB")
        
        # Display some sample results
        if results:
            logger.info("\nSample results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                status = result.get('status', 'unknown')
                key = result.get('original_key', 'unknown')
                text_uri = result.get('text_uri', 'N/A')
                error = result.get('error', 'None')
                dynamodb = "✓" if result.get('dynamodb_entry') else "✗"
                
                logger.info(f"Result {i+1}: {key} - Status: {status} - DynamoDB: {dynamodb}")
                if text_uri != 'N/A':
                    logger.info(f"  Text URI: {text_uri}")
                if error != 'None':
                    logger.info(f"  Error: {error}")
            
            if len(results) > 3:
                logger.info(f"... and {len(results) - 3} more results")
                
        return 0
            
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 