"""
Demo script for RAG with Knowledge Graph functionality.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
from src.data.document_loader import DocumentProcessor
from src.model.model_catalogue import ModelCatalogue
from src.model.wrappers import ChatModelWrapper, EmbeddingWrapper
from src.interaction.entity_extraction import EntityExtractor
from src.interaction.rag_knowledge_graph import RAGKnowledgeGraph
from src.flow.rag_process import RAGProcess

def setup_neo4j_connection():
    """
    Set up Neo4j connection parameters.
    
    Returns:
        Tuple of (uri, username, password)
    """
    # Get Neo4j connection parameters from environment variables or use defaults
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j+ssc://54c74f06.databases.neo4j.io")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "Vab-d8acjfnWiwYVo3PblfNYZZAqd7-P0CyGh6Iomgs")
    
    logger.info(f"Using Neo4j connection: {neo4j_uri}")
    
    return neo4j_uri, neo4j_username, neo4j_password

def main():
    """
    Run a demonstration of RAG with Knowledge Graph.
    """
    try:
        # Step 1: Set up Neo4j connection
        neo4j_uri, neo4j_username, neo4j_password = setup_neo4j_connection()
        
        # Step 2: Get model and embedding types
        model_type = ModelCatalogue._models.get("oai_3.5_final")  # Using GPT-3.5 Turbo
        embedding_type = ModelCatalogue._embeddings.get("oai_text_3_large")  # Using OpenAI embeddings
        
        if not model_type or not embedding_type:
            logger.error("Model or embedding type not found in ModelCatalogue")
            return
        
        # Step 3: Create model and embedding wrappers
        model_wrapper = ChatModelWrapper(model_type)
        embedding_wrapper = EmbeddingWrapper(embedding_type)
        
        # Step 4: Create entity extractor
        entity_extractor = EntityExtractor(model_wrapper)
        
        # Step 5: Create RAG Knowledge Graph
        rag_kg = RAGKnowledgeGraph(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            model_wrapper=model_wrapper,
            embedding_wrapper=embedding_wrapper,
            entity_extractor=entity_extractor
        )

        # Ask the user whether to reuse existing graph and vector index
        reuse_existing = input("Do you want to reuse an existing knowledge graph and vector index? (y/n): ").strip().lower() == "y"

        if not reuse_existing:
            # Step 6: Load and process documents
            pdf_path = input("Enter the path to a PDF document: ").strip().strip("'\"")
            document_processor = DocumentProcessor(chunk_size=512, chunk_overlap=24)
            documents = document_processor.process_pdf(pdf_path)
            
            logger.info(f"Loaded and processed {len(documents)} document chunks")

            # Step 7: Add documents to knowledge graph
            logger.info("Converting documents to graph documents and adding to Neo4j...")
            rag_kg.add_documents_to_graph(documents)

            # Step 8: Create vector index
            logger.info("Creating vector index...")
            rag_kg.create_vector_index()
        else:
            logger.info("Skipping document ingestion and vector index creation.")
            rag_kg.create_vector_index()  # still need this to initialize the wrapper

        
        # # Step 6: Load and process documents
        # # pdf_path = input("Enter the path to a PDF document: ")
        # pdf_path = input("Enter the path to a PDF document: ").strip().strip("'\"")

        
        # document_processor = DocumentProcessor(chunk_size=512, chunk_overlap=24)
        # documents = document_processor.process_pdf(pdf_path)
        
        # logger.info(f"Loaded and processed {len(documents)} document chunks")
        
        # # Step 7: Add documents to knowledge graph
        # logger.info("Converting documents to graph documents and adding to Neo4j...")
        # rag_kg.add_documents_to_graph(documents)
        
        # # Step 8: Create vector index
        # logger.info("Creating vector index...")
        # rag_kg.create_vector_index()
        
        # Step 9: Create RAG process
        rag_process = RAGProcess(rag_kg, model_wrapper)
        
        # Step 10: Interactive Q&A loop
        chat_history = []
        
        print("\nRAG with Knowledge Graph Q&A System")
        print("Type 'exit' to quit\n")
        
        while True:
            question = input("\nEnter your question: ")
            
            if question.lower() in ["exit", "quit", "q"]:
                break
            
            # Process the question
            result = rag_process.run({
                "question": question,
                "chat_history": chat_history
            })
            
            # Display the answer
            # print(f"\nAnswer: {result['answer']}")

            print(f"\nAnswer: {result['answer']}")
            print("\nGraph Evidence Used:")
            for line in result.get("graph_evidence", []):
                print(f"  - {line}")

            
            # Update chat history
            chat_history.append((question, result["answer"]))
            
            # Keep chat history limited to last 5 exchanges
            if len(chat_history) > 5:
                chat_history = chat_history[-5:]
        
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
