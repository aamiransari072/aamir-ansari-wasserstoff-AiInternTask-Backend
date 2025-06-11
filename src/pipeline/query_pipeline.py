import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.components.data_retriever import DataRetriever
from src.Agent.google import Gemini
from src.Agent.openai import OpenAI
from src.logging_config import logger
from src.utils.environment import get_env_variable


load_dotenv()

# Default prompt template for answering queries
DEFAULT_PROMPT_TEMPLATE = """
You are an experienced research scholar and academic expert with deep knowledge in various fields. Your role is to provide well-researched, analytical, and evidence-based responses to academic and research-related queries.

Guidelines for your response:
1. Analyze the provided context thoroughly and draw meaningful insights
2. Support your answers with specific evidence from the context
3. Maintain academic rigor while ensuring clarity
4. If the context is insufficient, acknowledge limitations and suggest potential research directions
5. Structure your response with:
   - Key findings/insights
   - Supporting evidence
   - Critical analysis
   - Implications or applications
   - Areas for further research (if applicable)

Context information:
{context}



User Question: {question}

Answer like Chatbot dont use any other text
"""


class QueryPipeline:
    """
    Pipeline for answering user queries using document retrieval and LLM:
    1. Retrieve relevant documents based on user query
    2. Construct context from retrieved documents
    3. Send query and context to LLM for answering
    """
    
    def __init__(self, top_k: int = 10, prompt_template: str = None):
        """
        Initialize the query pipeline with retriever and LLM
        
        Args:
            top_k: Number of documents to retrieve per query
            prompt_template: Custom prompt template (if None, use default)
        """
        logger.info("Initializing QueryPipeline")
        
        # Initialize the data retriever
        self.retriever = DataRetriever(top_k=top_k)
        logger.info(f"Data retriever initialized with top_k={top_k}")
        
        # Initialize the Gemini LLM
        gpt_api_key = get_env_variable("OPENAI_API_KEY")
        if not gpt_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = OpenAI(api_key=gpt_api_key,model='gpt-4.1')
        logger.info("OpenAI LLM initialized")
        
        # Set up the prompt template
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        logger.info("Prompt template initialized")
    
    def format_documents(self, docs: List[Document]) -> str:
        """
        Format a list of documents into a single context string
        
        Args:
            docs: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        if not docs:
            return "No relevant documents found."
        
        # Format each document with its content
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_doc = f"[Document {i+1}]\n\nContent:\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        # Join all formatted documents
        return "\n".join(formatted_docs)
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a user query using retrieval and LLM
        
        Args:
            query: The user's query string
        
        Returns:
            Dictionary with query results including answer and sources
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.retrieve_documents(query=query)
            if not docs:
                logger.warning("No documents retrieved for query")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "success": False
                }
            
            # Format documents into context
            context = self.format_documents(docs)
            logger.info(f"Created context from {len(docs)} documents")
            
            # Prepare the prompt with context and query
            full_prompt = self.prompt.format(context=context, question=query)
            
            # Generate response using Gemini
            response = self.llm.generate(full_prompt)
            answer = response.text
            logger.info("Generated answer using Gemini LLM")
            
            # Prepare source information with S3 URLs
            sources = []
            for doc in docs:
                s3_key = doc.metadata.get("s3_key")
                if s3_key:
                    # Generate S3 URL using the Cloudflare R2 endpoint
                    s3_url = f"https://{os.getenv('S3_BUCKET_NAME')}.r2.cloudflarestorage.com/{s3_key}"
                else:
                    s3_url = None
                    
                source_info = {
                    "document_id": doc.metadata.get("document_id", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "s3_url": s3_url
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def answer_query_with_reranking(self, query: str, top_k_retrieve: int = 10, 
                                    top_k_rerank: int = 5) -> Dict[str, Any]:
        """
        Answer a user query with reranking of retrieved documents
        
        Args:
            query: The user's query string
            top_k_retrieve: Number of documents to initially retrieve
            top_k_rerank: Number of documents to use after reranking
        
        Returns:
            Dictionary with query results including answer and sources
        """
        logger.info(f"Processing query with reranking: {query}")
        
        try:
            # Retrieve and rerank documents
            docs = self.retriever.retrieve_and_rerank(
                query=query, 
                top_k_retrieve=top_k_retrieve, 
                top_k_rerank=top_k_rerank
            )
            
            if not docs:
                logger.warning("No documents retrieved for query")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "success": False
                }
            
            # Format documents into context
            context = self.format_documents(docs)
            logger.info(f"Created context from {len(docs)} reranked documents")
            
            # Prepare the prompt with context and query
            full_prompt = self.prompt.format(context=context, question=query)
            
            # Generate response using Gemini
            response = self.llm.generate(full_prompt)
            answer = response.text
            logger.info("Generated answer using Gemini LLM")
            
            # Prepare source information
            sources = []
            for doc in docs:
                source_info = {
                    "document_id": doc.metadata.get("document_id", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown")
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error answering query with reranking: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "success": False
            } 