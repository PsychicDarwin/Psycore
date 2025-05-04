# """
# RAG with Knowledge Graph module for integrating Neo4j with LLMs.
# """
# from typing import List, Dict, Any, Optional
# from langchain.schema import Document
# from langchain_community.graphs import Neo4jGraph
# from langchain_community.vectorstores import Neo4jVector
# from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
# from langchain_openai import OpenAIEmbeddings
# from src.model.wrappers import ChatModelWrapper, EmbeddingWrapper
# from src.model.model_catalogue import ModelType, EmbeddingType
# from src.interaction.knowledge_graphs import ModelKGTransformer
# from src.interaction.entity_extraction import EntityExtractor
# from src.credential_manager.LocalCredentials import LocalCredentials

# class RAGKnowledgeGraph:
#     """
#     A class for implementing RAG with Neo4j Knowledge Graph.
#     """
    
#     def __init__(
#         self, 
#         neo4j_uri: str = "bolt://localhost:7687",
#         neo4j_username: str = "neo4j",
#         neo4j_password: str = "password",
#         model_wrapper: Optional[ChatModelWrapper] = None,
#         embedding_wrapper: Optional[EmbeddingWrapper] = None,
#         entity_extractor: Optional[EntityExtractor] = None
#     ):
#         """
#         Initialize the RAGKnowledgeGraph.
        
#         Args:
#             neo4j_uri: URI for Neo4j connection
#             neo4j_username: Username for Neo4j connection
#             neo4j_password: Password for Neo4j connection
#             model_wrapper: ChatModelWrapper instance for the LLM
#             embedding_wrapper: EmbeddingWrapper instance for embeddings
#             entity_extractor: EntityExtractor instance for entity extraction
#         """
#         # Set Neo4j connection parameters
#         self.neo4j_uri = LocalCredentials.get_credential('NEO4J_URI').secret_key
#         neo4j_credentials = LocalCredentials.get_credential('NEO4J_CREDENTIALS')
#         self.neo4j_username = neo4j_credentials.user_key
#         self.neo4j_password = neo4j_credentials.secret_key
        
#         # Initialize Neo4j graph
#         self.graph = Neo4jGraph(
#             url=self.neo4j_uri,
#             username=self.neo4j_username,
#             password=self.neo4j_password
#         )
        
#         # Set model and embedding wrappers
#         self.model_wrapper = model_wrapper
#         self.embedding_wrapper = embedding_wrapper
        
#         # Set entity extractor
#         self.entity_extractor = entity_extractor
        
#         # Initialize vector index to None (will be created after adding documents)
#         self.vector_index = None
        
#         # Initialize graph transformer to None (will be created when needed)
#         self.graph_transformer = None
    
#     @classmethod
#     def from_model_types(
#         cls,
#         model_type: ModelType,
#         embedding_type: EmbeddingType,
#         neo4j_uri: str = "bolt://localhost:7687",
#         neo4j_username: str = "neo4j",
#         neo4j_password: str = "password"
#     ) -> 'RAGKnowledgeGraph':
#         """
#         Create a RAGKnowledgeGraph from ModelType and EmbeddingType.
        
#         Args:
#             model_type: ModelType instance
#             embedding_type: EmbeddingType instance
            
#         Returns:
#             RAGKnowledgeGraph instance
#         """
#         from src.model.wrappers import ChatModelWrapper, EmbeddingWrapper
#         from src.interaction.entity_extraction import EntityExtractor
        
#         model_wrapper = ChatModelWrapper(model_type)
#         embedding_wrapper = EmbeddingWrapper(embedding_type)
#         entity_extractor = EntityExtractor(model_wrapper)
        
#         return cls(
#             neo4j_uri=neo4j_uri,
#             neo4j_username=neo4j_username,
#             neo4j_password=neo4j_password,
#             model_wrapper=model_wrapper,
#             embedding_wrapper=embedding_wrapper,
#             entity_extractor=entity_extractor
#         )
    
#     def _ensure_graph_transformer(self):
#         """
#         Ensure that the graph transformer is initialized.
#         """
#         if self.graph_transformer is None and self.model_wrapper is not None:
#             self.graph_transformer = ModelKGTransformer(self.model_wrapper.model_type)

    
#     def convert_to_graph_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
#         """
#         Convert documents to graph documents.
        
#         Args:
#             documents: List of documents to convert
            
#         Returns:
#             List of graph documents
#         """
#         self._ensure_graph_transformer()
#         return self.graph_transformer.convert_to_graph_documents(documents)
    
#     def add_documents_to_graph(
#         self, 
#         documents: List[Document],
#         base_entity_label: bool = True,
#         include_source: bool = True
#     ) -> None:
#         """
#         Add documents to the knowledge graph.
        
#         Args:
#             documents: List of documents to add
#             base_entity_label: Whether to use base entity label
#             include_source: Whether to include source information
#         """
#         # Convert documents to graph documents
#         graph_documents = self.convert_to_graph_documents(documents)
        
#         # Add graph documents to Neo4j
#         self.graph.add_graph_documents(
#             graph_documents,
#             baseEntityLabel=base_entity_label,
#             include_source=include_source
#         )
        
#         # Create fulltext index for entities
#         self.graph.query(
#             "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
#         )
    
#     def create_vector_index(
#         self,
#         node_label: str = "Document",
#         text_node_properties: List[str] = ["text"],
#         embedding_node_property: str = "embedding",
#         search_type: str = "hybrid"
#     ) -> None:
#         """
#         Create a vector index for the knowledge graph.
        
#         Args:
#             node_label: Label of nodes to index
#             text_node_properties: Properties containing text to embed
#             embedding_node_property: Property to store embeddings
#             search_type: Type of search to use
#         """
#         # Use OpenAIEmbeddings if embedding_wrapper is not provided
#         embedding_model = (
#             self.embedding_wrapper.embedding 
#             if self.embedding_wrapper is not None 
#             else OpenAIEmbeddings()
#         )
        
#         # Create vector index
#         self.vector_index = Neo4jVector.from_existing_graph(
#             embedding=embedding_model,
#             search_type=search_type,
#             node_label=node_label,
#             text_node_properties=text_node_properties,
#             embedding_node_property=embedding_node_property,
#             url=self.neo4j_uri,
#             username=self.neo4j_username,
#             password=self.neo4j_password
#         )
    
#     def generate_full_text_query(self, input_text: str) -> str:
#         """
#         Generate a full-text query for Neo4j.
        
#         Args:
#             input_text: Input text to generate query from
            
#         Returns:
#             Full-text query string
#         """
#         full_text_query = ""
#         words = [el for el in remove_lucene_chars(input_text).split() if el]
        
#         for word in words[:-1]:
#             full_text_query += f" {word}~2 AND"
        
#         full_text_query += f" {words[-1]}~2"
        
#         return full_text_query.strip()
    
#     # def structured_retrieval(self, question: str) -> str:
#     #     """
#     #     Perform structured retrieval using the knowledge graph.
        
#     #     Args:
#     #         question: Question to retrieve information for
            
#     #     Returns:
#     #         Retrieved information as a string
#     #     """
#     #     result = ""
        
#     #     # Extract entities from the question
#     #     entities = self.entity_extractor.extract_entities(question)
        
#     #     # Query the knowledge graph for each entity
#     #     for entity in entities:
#     #         response = self.graph.query(
#     #             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
#     #             YIELD node,score
#     #             CALL {
#     #               WITH node
#     #               MATCH (node)-[r:!MENTIONS]->(neighbor)
#     #               RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
#     #               UNION ALL
#     #               WITH node
#     #               MATCH (node)<-[r:!MENTIONS]-(neighbor)
#     #               RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
#     #             }
#     #             RETURN output LIMIT 50""",
#     #             {"query": self.generate_full_text_query(entity)},
#     #         )
            
#     #         result += "\n".join([el['output'] for el in response])
        
#     #     return result

#     def structured_retrieval(self, question: str) -> Dict[str, Any]:
#         """
#         Perform structured retrieval using the knowledge graph.
#         Args:
#             question: Question to retrieve information for
#         Returns:
#             Dictionary containing formatted string + raw graph outputs
#         """
#         structured_text = ""
#         graph_outputs = []

#         entities = self.entity_extractor.extract_entities(question)

#         for entity in entities:
#             if not entity.strip():
#                 continue
#             query = self.generate_full_text_query(entity)
#             response = self.graph.query(
#                 """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
#                 YIELD node,score
#                 CALL {
#                 WITH node
#                 MATCH (node)-[r:!MENTIONS]->(neighbor)
#                 RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
#                 UNION ALL
#                 WITH node
#                 MATCH (node)<-[r:!MENTIONS]-(neighbor)
#                 RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
#                 }
#                 RETURN output LIMIT 50""",
#                 {"query": query}
#             )

#             for el in response:
#                 line = el['output']
#                 structured_text += line + "\n"
#                 graph_outputs.append(line)

#         return {
#             "structured_text": structured_text.strip(),
#             "graph_evidence": graph_outputs
#         }

    
#     def vector_retrieval(self, question: str, k: int = 4) -> List[str]:
#         """
#         Perform vector retrieval using the vector index.
        
#         Args:
#             question: Question to retrieve information for
#             k: Number of documents to retrieve
            
#         Returns:
#             List of retrieved document contents
#         """
#         if self.vector_index is None:
#             raise ValueError("Vector index not created. Call create_vector_index() first.")
        
#         results = self.vector_index.similarity_search(question, k=k)
#         return [doc.page_content for doc in results]
    
#     # def hybrid_retrieval(self, question: str, k: int = 4) -> str:
#     #     """
#     #     Perform hybrid retrieval using both structured and vector retrieval.
        
#     #     Args:
#     #         question: Question to retrieve information for
#     #         k: Number of documents to retrieve from vector search
            
#     #     Returns:
#     #         Combined retrieval results as a string
#     #     """
#     #     print(f"Search query: {question}")
        
#     #     # Get structured data
#     #     structured_data = self.structured_retrieval(question)
        
#     #     # Get unstructured data
#     #     unstructured_data = self.vector_retrieval(question, k=k)
        
#     #     # Combine results
#     #     final_data = f"""Structured data:
#     #     {structured_data}
        
#     #     Unstructured data:
#     #     {"#Document ". join(unstructured_data)}
#     #     """
        
#     #     return final_data

#     def hybrid_retrieval(self, question: str, k: int = 4) -> Dict[str, Any]:
#         print(f"Search query: {question}")

#         structured_data = self.structured_retrieval(question)
#         unstructured_docs = self.vector_retrieval(question, k=k)

#         formatted_context = f"""Structured data:
#         {structured_data['structured_text']}

#         Unstructured data:
#         {"#Document ".join(unstructured_docs)}
#         """

#         return {
#             "context": formatted_context.strip(),
#             "graph_evidence": structured_data["graph_evidence"],
#             "documents": unstructured_docs
#         }


"""
RAG with Knowledge Graph module for integrating Neo4j with LLMs.
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_openai import OpenAIEmbeddings
from src.model.wrappers import ChatModelWrapper, EmbeddingWrapper
from src.model.model_catalogue import ModelType, EmbeddingType
from src.interaction.knowledge_graphs import ModelKGTransformer
from src.interaction.entity_extraction import EntityExtractor

class RAGKnowledgeGraph:
    """
    A class for implementing RAG with Neo4j Knowledge Graph.
    """
    
    def __init__(
        self, 
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        model_wrapper: Optional[ChatModelWrapper] = None,
        embedding_wrapper: Optional[EmbeddingWrapper] = None,
        entity_extractor: Optional[EntityExtractor] = None
    ):
        """
        Initialize the RAGKnowledgeGraph.
        
        Args:
            neo4j_uri: URI for Neo4j connection
            neo4j_username: Username for Neo4j connection
            neo4j_password: Password for Neo4j connection
            model_wrapper: ChatModelWrapper instance for the LLM
            embedding_wrapper: EmbeddingWrapper instance for embeddings
            entity_extractor: EntityExtractor instance for entity extraction
        """
        # Set Neo4j connection parameters
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        # Initialize Neo4j graph
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Set model and embedding wrappers
        self.model_wrapper = model_wrapper
        self.embedding_wrapper = embedding_wrapper
        
        # Set entity extractor
        self.entity_extractor = entity_extractor
        
        # Initialize vector index to None (will be created after adding documents)
        self.vector_index = None
        
        # Initialize graph transformer to None (will be created when needed)
        self.graph_transformer = None
    
    @classmethod
    def from_model_types(
        cls,
        model_type: ModelType,
        embedding_type: EmbeddingType,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password"
    ) -> 'RAGKnowledgeGraph':
        """
        Create a RAGKnowledgeGraph from ModelType and EmbeddingType.
        
        Args:
            model_type: ModelType instance
            embedding_type: EmbeddingType instance
            neo4j_uri: URI for Neo4j connection
            neo4j_username: Username for Neo4j connection
            neo4j_password: Password for Neo4j connection
            
        Returns:
            RAGKnowledgeGraph instance
        """
        from src.model.wrappers import ChatModelWrapper, EmbeddingWrapper
        from src.interaction.entity_extraction import EntityExtractor
        
        model_wrapper = ChatModelWrapper(model_type)
        embedding_wrapper = EmbeddingWrapper(embedding_type)
        entity_extractor = EntityExtractor(model_wrapper)
        
        return cls(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            model_wrapper=model_wrapper,
            embedding_wrapper=embedding_wrapper,
            entity_extractor=entity_extractor
        )
    
    def _ensure_graph_transformer(self):
        """
        Ensure that the graph transformer is initialized.
        """
        if self.graph_transformer is None and self.model_wrapper is not None:
            self.graph_transformer = ModelKGTransformer(self.model_wrapper.model_type)

    
    def convert_to_graph_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Convert documents to graph documents.
        
        Args:
            documents: List of documents to convert
            
        Returns:
            List of graph documents
        """
        self._ensure_graph_transformer()
        return self.graph_transformer.convert_to_graph_documents(documents)
    
    # def add_documents_to_graph(
    #     self, 
    #     documents: List[Document],
    #     base_entity_label: bool = True,
    #     include_source: bool = True
    # ) -> None:
    #     """
    #     Add documents to the knowledge graph.
        
    #     Args:
    #         documents: List of documents to add
    #         base_entity_label: Whether to use base entity label
    #         include_source: Whether to include source information
    #     """
    #     # Convert documents to graph documents
    #     graph_documents = self.convert_to_graph_documents(documents)
        
    #     # Add graph documents to Neo4j
    #     self.graph.add_graph_documents(
    #         graph_documents,
    #         baseEntityLabel=base_entity_label,
    #         include_source=include_source
    #     )
        
    #     # Create fulltext index for entities
    #     self.graph.query(
    #         "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    #     )

    def add_documents_to_graph(
        self, 
        documents: List[Document],
        base_entity_label: bool = True,
        include_source: bool = True,
        return_graph_json: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Add documents to the knowledge graph and optionally return the graph as JSON.

        Args:
            documents: List of documents to add
            base_entity_label: Whether to use base entity label
            include_source: Whether to include source information
            return_graph_json: If True, returns a JSON of the graph

        Returns:
            Optional JSON dictionary of graph content
        """
        graph_documents = self.convert_to_graph_documents(documents)
        
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=base_entity_label,
            include_source=include_source
        )
        
        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )

        if return_graph_json:
            return self.graph_docs_to_json(graph_documents)
    
    def create_vector_index(
        self,
        node_label: str = "Document",
        text_node_properties: List[str] = ["text"],
        embedding_node_property: str = "embedding",
        search_type: str = "hybrid"
    ) -> None:
        """
        Create a vector index for the knowledge graph.
        
        Args:
            node_label: Label of nodes to index
            text_node_properties: Properties containing text to embed
            embedding_node_property: Property to store embeddings
            search_type: Type of search to use
        """
        # Use OpenAIEmbeddings if embedding_wrapper is not provided
        embedding_model = (
            self.embedding_wrapper.embedding 
            if self.embedding_wrapper is not None 
            else OpenAIEmbeddings()
        )
        
        # Create vector index
        self.vector_index = Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            search_type=search_type,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
    
    def generate_full_text_query(self, input_text: str) -> str:
        """
        Generate a full-text query for Neo4j.
        
        Args:
            input_text: Input text to generate query from
            
        Returns:
            Full-text query string
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input_text).split() if el]
        
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        
        full_text_query += f" {words[-1]}~2"
        
        return full_text_query.strip()
    
    # def structured_retrieval(self, question: str) -> str:
    #     """
    #     Perform structured retrieval using the knowledge graph.
        
    #     Args:
    #         question: Question to retrieve information for
            
    #     Returns:
    #         Retrieved information as a string
    #     """
    #     result = ""
        
    #     # Extract entities from the question
    #     entities = self.entity_extractor.extract_entities(question)
        
    #     # Query the knowledge graph for each entity
    #     for entity in entities:
    #         response = self.graph.query(
    #             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
    #             YIELD node,score
    #             CALL {
    #               WITH node
    #               MATCH (node)-[r:!MENTIONS]->(neighbor)
    #               RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
    #               UNION ALL
    #               WITH node
    #               MATCH (node)<-[r:!MENTIONS]-(neighbor)
    #               RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
    #             }
    #             RETURN output LIMIT 50""",
    #             {"query": self.generate_full_text_query(entity)},
    #         )
            
    #         result += "\n".join([el['output'] for el in response])
        
    #     return result

    def structured_retrieval(self, question: str) -> Dict[str, Any]:
        """
        Perform structured retrieval using the knowledge graph.
        Args:
            question: Question to retrieve information for
        Returns:
            Dictionary containing formatted string + raw graph outputs
        """
        structured_text = ""
        graph_outputs = []

        entities = self.entity_extractor.extract_entities(question)

        for entity in entities:
            if not entity.strip():
                continue
            query = self.generate_full_text_query(entity)
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50""",
                {"query": query}
            )

            for el in response:
                line = el['output']
                structured_text += line + "\n"
                graph_outputs.append(line)

        return {
            "structured_text": structured_text.strip(),
            "graph_evidence": graph_outputs
        }

    
    def vector_retrieval(self, question: str, k: int = 4) -> List[str]:
        """
        Perform vector retrieval using the vector index.
        
        Args:
            question: Question to retrieve information for
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document contents
        """
        if self.vector_index is None:
            raise ValueError("Vector index not created. Call create_vector_index() first.")
        
        results = self.vector_index.similarity_search(question, k=k)
        return [doc.page_content for doc in results]
    
    # def hybrid_retrieval(self, question: str, k: int = 4) -> str:
    #     """
    #     Perform hybrid retrieval using both structured and vector retrieval.
        
    #     Args:
    #         question: Question to retrieve information for
    #         k: Number of documents to retrieve from vector search
            
    #     Returns:
    #         Combined retrieval results as a string
    #     """
    #     print(f"Search query: {question}")
        
    #     # Get structured data
    #     structured_data = self.structured_retrieval(question)
        
    #     # Get unstructured data
    #     unstructured_data = self.vector_retrieval(question, k=k)
        
    #     # Combine results
    #     final_data = f"""Structured data:
    #     {structured_data}
        
    #     Unstructured data:
    #     {"#Document ". join(unstructured_data)}
    #     """
        
    #     return final_data

    def hybrid_retrieval(self, question: str, k: int = 4) -> Dict[str, Any]:
        print(f"Search query: {question}")

        structured_data = self.structured_retrieval(question)
        unstructured_docs = self.vector_retrieval(question, k=k)

        formatted_context = f"""Structured data:
        {structured_data['structured_text']}

        Unstructured data:
        {"#Document ".join(unstructured_docs)}
        """

        return {
            "context": formatted_context.strip(),
            "graph_evidence": structured_data["graph_evidence"],
            "documents": unstructured_docs
        }
    
    def graph_docs_to_json(self, graph_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert graph documents to JSON-friendly format (nodes and relationships).
        
        Args:
            graph_documents: Output from convert_to_graph_documents
        
        Returns:
            Dictionary with 'nodes' and 'relationships'
        """
        nodes = []
        relationships = []

        for doc in graph_documents:
            for node in doc.nodes:
                nodes.append({"id": node.id, "type": node.type})
            for rel in doc.relationships:
                relationships.append({
                    "source": rel.source.id if rel.source else None,
                    "target": rel.target.id if rel.target else None,
                    "type": rel.type
                })

        return {"nodes": nodes, "relationships": relationships}
