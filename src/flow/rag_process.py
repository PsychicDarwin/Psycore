"""
RAG process module for integrating RAG into the processing flow.
"""
from typing import Dict, List, Tuple, Optional, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from src.flow.processflow import ProcessFlow
from src.model.wrappers import ChatModelWrapper
from src.interaction.rag_knowledge_graph import RAGKnowledgeGraph

class RAGProcess(ProcessFlow):
    """
    A process flow for RAG with knowledge graph.
    """
    
    def __init__(
        self, 
        rag_knowledge_graph: RAGKnowledgeGraph,
        model_wrapper: ChatModelWrapper,
        next: Optional['ProcessFlow'] = None
    ):
        """
        Initialize the RAGProcess.
        
        Args:
            rag_knowledge_graph: RAGKnowledgeGraph instance
            model_wrapper: ChatModelWrapper instance for the LLM
            next: Next process in the chain
        """
        super().__init__(runner=rag_knowledge_graph, next=next)
        self.rag_kg = rag_knowledge_graph
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        
        # Create the QA chain
        self._create_qa_chain()
    
    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        """
        Format chat history for the LLM.
        
        Args:
            chat_history: List of (human, ai) message tuples
            
        Returns:
            Formatted chat history
        """
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    # def _create_qa_chain(self):
    #     """
    #     Create the question answering chain.
    #     """
    #     # Create the condense question prompt
    #     condense_question_prompt = PromptTemplate.from_template(
    #         """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    #         in its original language.
            
    #         Chat History:
    #         {chat_history}
            
    #         Follow Up Input: {question}
            
    #         Standalone question:"""
    #     )
        
    #     # Create the search query component
    #     self._search_query = RunnableBranch(
    #         # If input includes chat_history, we condense it with the follow-up question
    #         (
    #             RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
    #                 run_name="HasChatHistoryCheck"
    #             ),
    #             RunnablePassthrough.assign(
    #                 chat_history=lambda x: self._format_chat_history(x["chat_history"])
    #             )
    #             | condense_question_prompt
    #             | self.model
    #             | StrOutputParser(),
    #         ),
    #         # Else, we have no chat history, so just pass through the question
    #         RunnableLambda(lambda x: x["question"]),
    #     )
        
    #     # Create the QA prompt
    #     qa_prompt = ChatPromptTemplate.from_template(
    #         """Answer the question based only on the following context:
            
    #         {context}
            
    #         Question: {question}
            
    #         Use natural language and be concise.
            
    #         Answer:"""
    #     )
        
    #     # Create the QA chain
    #     self.qa_chain = (
    #         RunnableParallel(
    #             {
    #                 # "context": self._search_query | RunnableLambda(self.rag_kg.hybrid_retrieval),
    #                 "context": self._search_query | RunnableLambda(lambda q: self.rag_kg.hybrid_retrieval(q)["context"]),
    #                 "question": RunnablePassthrough(),
    #             }
    #         )
    #         | qa_prompt
    #         | self.model
    #         | StrOutputParser()
    #     )

    def _create_qa_chain(self):
        """
        Create the question answering chain.
        """
        condense_question_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
            in its original language.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            Standalone question:"""
        )

        self._search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | condense_question_prompt
                | self.model
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )

        qa_prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            
            {context}
            
            Question: {question}
            
            Use natural language and be concise.
            
            Answer:"""
        )

        self.qa_chain = (
            RunnableParallel(
                {
                    "context": self._search_query | RunnableLambda(lambda q: self.rag_kg.hybrid_retrieval(q)["context"]),
                    "question": RunnablePassthrough(),
                }
            )
            | qa_prompt
            | self.model
            | StrOutputParser()
        )

    
    # def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Run the RAG process.
        
    #     Args:
    #         data: Input data containing at least a "question" key
            
    #     Returns:
    #         Dictionary containing the answer
    #     """
    #     # When running a RAG process we expect data in the form
    #     # {
    #     #     "question": str,
    #     #     "chat_history": List[Tuple[str, str]] (optional)
    #     # }
        
    #     if "question" not in data:
    #         raise ValueError("Input data must contain a 'question' key")
        
    #     # Invoke the QA chain
    #     # answer = self.qa_chain.invoke(data)

    #     # Get retrieval context and evidence
    #     retrieval_output = self.rag_kg.hybrid_retrieval(data["question"])
    #     data["context"] = retrieval_output["context"]

    #     # Invoke LLM
    #     answer = self.qa_chain.invoke(data)

    #     output = {
    #         "question": data["question"],
    #         "answer": answer,
    #         "graph_evidence": retrieval_output["graph_evidence"],
    #         "documents": retrieval_output["documents"]
    #     }

        
    #     # Prepare output
    #     # output = {
    #     #     "question": data["question"],
    #     #     "answer": answer,
    #     # }
        
    #     # If chat history was provided, include it in the output
    #     if "chat_history" in data:
    #         output["chat_history"] = data["chat_history"]
        
    #     # Pass to next process if available
    #     if self.next is not None:
    #         return self.next.run(output)
        
    #     return output

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the RAG process.

        Args:
            data: Input data containing at least a "question" key

        Returns:
            Dictionary containing the answer and evidence
        """
        if "question" not in data:
            raise ValueError("Input data must contain a 'question' key")

        # Get retrieval context and evidence
        retrieval_output = self.rag_kg.hybrid_retrieval(data["question"])
        data["context"] = retrieval_output["context"]

        # Invoke LLM
        answer = self.qa_chain.invoke(data)

        output = {
            "question": data["question"],
            "answer": answer,
            "graph_evidence": retrieval_output["graph_evidence"],
            "documents": retrieval_output["documents"],
        }

        if "chat_history" in data:
            output["chat_history"] = data["chat_history"]

        if self.next is not None:
            return self.next.run(output)

        return output

