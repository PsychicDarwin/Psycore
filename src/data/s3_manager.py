"""
S3 Manager for handling file operations with AWS S3.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from src.data.common_types import AttachmentTypes
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class S3Manager:
    """
    Class for handling S3 operations to download and upload files.
    """
    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None, aws_region: Optional[str] = None):
        """
        Initialize S3Manager with credentials.
        
        Args:
            bucket_name: Name of the S3 bucket
            aws_access_key_id: AWS access key ID (optional - falls back to environment variables)
            aws_secret_access_key: AWS secret access key (optional - falls back to environment variables)
            aws_region: AWS region (optional - falls back to environment variables)
        """
        self.bucket_name = bucket_name
        
        # Use provided credentials or fall back to environment variables
        credentials = {}
        if aws_access_key_id:
            credentials['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            credentials['aws_secret_access_key'] = aws_secret_access_key
        if aws_region:
            credentials['region_name'] = aws_region
            
        self.s3_client = boto3.client('s3', **credentials)
        self.s3_resource = boto3.resource('s3', **credentials)
        self.bucket = self.s3_resource.Bucket(bucket_name)
        
        logger.info(f"Initialized S3Manager for bucket: {bucket_name}")
    
    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List all files in a bucket with the given prefix.
        
        Args:
            prefix: S3 prefix to filter objects
            
        Returns:
            List of dictionaries containing file information
        """
        try:
            objects = []
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            
            if 'Contents' not in response:
                logger.info(f"No objects found in bucket {self.bucket_name} with prefix '{prefix}'")
                return []
                
            for obj in response['Contents']:
                # Skip folders (objects ending with '/')
                if not obj['Key'].endswith('/'):
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag']
                    })
            
            logger.debug(f"Found {len(objects)} objects in bucket {self.bucket_name} with prefix '{prefix}'")
            return objects
            
        except ClientError as e:
            logger.error(f"Error listing objects in S3: {str(e)}")
            raise
    
    def download_file(self, key: str, local_path: str) -> bool:
        """
        Download a file from S3 to a local path.
        
        Args:
            key: S3 object key
            local_path: Local path to save the file
            
        Returns:
            Boolean indicating if download was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            self.s3_client.download_file(self.bucket_name, key, local_path)
            logger.debug(f"Successfully downloaded {key} to {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading {key} from S3: {str(e)}")
            return False
            
    def upload_file(self, local_path: str, key: str, extra_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local path of file to upload
            key: S3 object key to use
            extra_args: Extra arguments for S3 upload (e.g., ContentType)
            
        Returns:
            Boolean indicating if upload was successful
        """
        try:
            # Set default extra_args if None
            if extra_args is None:
                extra_args = {}
            
            # If Content-Type not specified, try to determine from file extension
            if 'ContentType' not in extra_args:
                content_type = self._get_content_type(local_path)
                if content_type:
                    extra_args['ContentType'] = content_type
            
            # Upload the file
            self.s3_client.upload_file(local_path, self.bucket_name, key, ExtraArgs=extra_args)
            logger.debug(f"Successfully uploaded {local_path} to {key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading {local_path} to S3: {str(e)}")
            return False
            
    def download_files_by_prefix(self, prefix: str, local_dir: str) -> List[str]:
        """
        Download all files with a certain prefix to a local directory.
        
        Args:
            prefix: S3 key prefix
            local_dir: Local directory to save files
            
        Returns:
            List of local file paths that were downloaded
        """
        files = self.list_files(prefix)
        downloaded_files = []
        
        for file_info in files:
            key = file_info['key']
            local_path = os.path.join(local_dir, os.path.basename(key))
            
            if self.download_file(key, local_path):
                downloaded_files.append(local_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {local_dir}")
        return downloaded_files
    
    def s3_process_file(self, key: str, local_path: str):
        """
        Process a file from S3 and return an S3Attachment object.
        
        Args:
            key: S3 object key
            local_path: Local path to save the file
            
        Returns:
            S3Attachment object
        """
        # Import here to break circular dependency
        from src.data.attachments import S3Attachment
        
        if self.download_file(key, local_path):
            attachment = S3Attachment(AttachmentTypes.from_filename(local_path), local_path, True, key, self.bucket_name)
            attachment.extract()
            if attachment.needsExtraction:
                raise Exception(f"Failed to process file {key} from S3")
            else:
                os.remove(local_path)
            return attachment
        else:
            raise Exception(f"Failed to download file {key} from S3")
    
    def _get_content_type(self, file_path: str) -> Optional[str]:
        """
        Determine content type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type as string or None if can't be determined
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        # MIME type mapping
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.json': 'application/json',
            '.mp3': 'audio/mpeg',
            '.mp4': 'video/mp4',
            '.wav': 'audio/wav',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        }
        
        return content_types.get(extension)
