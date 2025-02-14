"""
Main entry point for the application.
"""
import os
import logging
from dotenv import load_dotenv

# Load .env variables only if they don't exist in system environment
load_dotenv(override=False)

# Environment configuration - system env takes precedence naturally now
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Program constants
if DEBUG:
    PROGRAM_MODE = "development"
    logger.info(f"Running in {PROGRAM_MODE} mode with log level: {LOG_LEVEL}")
    logger.debug("Debug mode is enabled")
else:
    PROGRAM_MODE = "production"
    logger.info(f"Running in {PROGRAM_MODE} mode with log level: {LOG_LEVEL}")

def main():
    """Run a simple example of the calculator."""
    try:
        
        if DEBUG:
            logger.debug("Debug information:")
            logger.debug(f"Current working directory: {os.getcwd()}")
            env_vars = {key: '[SET]' if val else '[NOT SET]' 
                       for key, val in os.environ.items() 
                       if key in ['DEBUG', 'LOG_LEVEL']}
            logger.debug(f"Environment variables status: {env_vars}")
            
    except Exception as e:
        if DEBUG:
            logger.exception("An error occurred in debug mode:")
            raise
        else:
            logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()