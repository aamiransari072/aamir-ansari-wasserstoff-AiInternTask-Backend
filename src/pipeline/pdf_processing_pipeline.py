import os
import uuid
import io
import filetype
from typing import List, Dict, Any, Optional
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
import itertools
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from src.components.document_extraction import DocumentExtractor
from src.components.data_splitter import DataSplitter
from src.components.data_ingestion import DataIngestionService
from src.logging_config import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.env_checker import check_required_env_vars
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import cohere

load_dotenv()

class PDFProcessingPipeline:
    """
    Pipeline for processing PDF documents:
    1. Store PDFs in temp directory
    2. Upload PDFs to S3
    3. Store metadata in MongoDB
    4. Load documents using PyPDFDirectoryLoader
    5. Split documents into chunks
    6. Create embeddings
    7. Store vectors in Pinecone
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize all necessary connections and components
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        logger.info("Initializing PDFProcessingPipeline")
        
        # Check required environment variables
        check_required_env_vars()
        
        # Initialize data ingestion service
        self.data_ingestion = DataIngestionService()
        
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
            pool_threads=30  # Enable parallel processing
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "kiet-docs")
        
        # Get embedding model dimensions
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
       
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            # Create a sample embedding to determine dimensions
            sample_embedding = self.embeddings.embed_query("Sample text for dimension check")
            embedding_dim = len(sample_embedding)
            logger.info(f"Embedding model produces vectors with {embedding_dim} dimensions")
            
            self.pc.create_index(
                self.index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            

        # Connect to Pinecone Index
        self.pinecone_index = self.pc.Index(self.index_name)
        logger.info(f"Pinecone initialized with index: {self.index_name}")
        
        # Initialize document extractor
        # self.document_extractor = DocumentExtractor()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_splitter = DataSplitter(self.text_splitter)
        
        # MongoDB collection name for PDF documents
        self.pdf_collection = "documents"
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Create temp directory for document storage
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Temporary directory created at: {self.temp_dir}")
        
        # Initialize file type checker
        self.mime = filetype

    def validate_file(self, file_content: bytes) -> Optional[str]:
        """
        Validate file type and return its MIME type if supported.
        
        Returns:
            MIME type string if supported; otherwise, None.
        """
        try:
            kind = filetype.guess(file_content)
            if not kind:
                logger.warning("Unable to guess file type.")
                return None
            if kind.mime in ['application/pdf', 'text/plain']:
                return kind.mime
            logger.warning(f"Unsupported MIME type: {kind.mime}")
            return None
        except Exception as e:
            logger.error(f"Error during file validation: {str(e)}")
            return None


    def cleanup_failed_processing(self, document_id: str, s3_key: str, temp_file_path: str = None) -> None:
        """
        Clean up resources if PDF processing fails
        
        Args:
            document_id: The MongoDB document ID
            s3_key: The S3 key of the uploaded file
            temp_file_path: The path to the temporary file, if any
        """
        try:
            # Delete from S3
            self.data_ingestion.s3_client.delete_object(
                Bucket=self.data_ingestion.bucket_name,
                Key=s3_key
            )
            
            # Delete from MongoDB
            self.data_ingestion.db[self.pdf_collection].delete_one(
                {"_id": ObjectId(document_id)}
            )
            
            # Delete temp file if exists
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            logger.info(f"Cleaned up failed processing for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up failed processing: {str(e)}")

    def save_pdf_to_temp(self, file_content: bytes, filename: str) -> str:
        """
        Save PDF content to a temporary file
        
        Args:
            file_content: The binary content of the PDF file
            filename: The name of the file
            
        Returns:
            temp_file_path: Path to the saved temporary file
        """
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        temp_file_path = os.path.join(self.temp_dir, unique_filename)
        
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
            
        logger.info(f"Saved PDF to temporary file: {temp_file_path}")
        return temp_file_path


    def load_documents_from_txt(self, file_path: str) -> List[Document]:
        """
        Load a .txt file and return as a single Document.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Loaded text file with {len(text)} characters.")
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception as e:
            logger.error(f"Failed to load .txt file: {str(e)}")
            return []


    def load_documents_from_pdf(self,pdf_path: str) -> List[Document]:
        """
        Load documents from a PDF file using PyPDF first, then fallback to UnstructuredPDFLoader if needed.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            documents: List of Document objects
        """
        documents = []

        try:
            logger.info(f"Attempting to load PDF with PyPDFLoader: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Check if any document has non-empty content
            if all(len(doc.page_content.strip()) == 0 for doc in documents):
                raise ValueError("PyPDFLoader extracted no meaningful text.")
            
            logger.info(f"PyPDFLoader successfully extracted {len(documents)} documents.")
        
        except Exception as e:
            logger.warning(f"PyPDFLoader failed or found empty content: {str(e)}")
            logger.info(f"Falling back to UnstructuredPDFLoader with OCR for: {pdf_path}")
            
            try:
                loader = UnstructuredPDFLoader(pdf_path, mode="elements")
                documents = loader.load()
                if all(len(doc.page_content.strip()) == 0 for doc in documents):
                    logger.warning("UnstructuredPDFLoader also found no text. File may be empty or unreadable.")
                else:
                    logger.info(f"UnstructuredPDFLoader extracted {len(documents)} documents.")
            except Exception as fallback_error:
                logger.error(f"UnstructuredPDFLoader failed to process PDF: {str(fallback_error)}")
                return []

        # Print debug info for first few documents
        for i, doc in enumerate(documents[:5]):
            logger.debug(f"Doc {i} length: {len(doc.page_content.strip())}")
            logger.debug(f"Doc {i} content preview:\n{repr(doc.page_content.strip()[:200])}")
            print(f"[DEBUG] Doc {i} | Length: {len(doc.page_content.strip())}")
            print(f"[DEBUG] Preview: {repr(doc.page_content.strip()[:200])}")

        return documents

    def chunks(self, iterable, batch_size=200):
        """Helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def process_documents(self, documents: List[Document], document_id: str, filename: str, s3_key: str) -> bool:
        """
        Process documents by splitting and storing in Pinecone using parallel batch upserts
        
        Args:
            documents: List of Document objects
            document_id: MongoDB document ID
            filename: Original filename
            s3_key: S3 key for the uploaded file
            
        Returns:
            success: Whether processing was successful
        """
        try:
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "filename": filename,
                    "s3_key": s3_key,
                    "document_id": document_id,
                })
            
            # Split documents into chunks
            chunks = self.data_splitter.split_data(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            
            # Prepare vectors for batch processing
            vectors_to_upsert = []

            logger.info(f"Preparing to upsert {len(chunks)} chunks for document ID: {document_id}")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_{i}"

                # Generate embeddings
                embedding_vector = self.embeddings.embed_query(chunk.page_content) 
                logger.debug(f"Generated embedding for chunk {i}, ID: {chunk_id}")

                # Prepare metadata
                metadata = {
                    "filename": filename,
                    "s3_key": s3_key,
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.page_content,
                    "page_number": chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source", "")
                }

                # Clean metadata
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool, list, dict)) and key != "values":
                        clean_metadata[key] = value

                # Add to vectors list
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding_vector,
                    "metadata": clean_metadata
                })

            logger.info(f"Prepared {len(vectors_to_upsert)} vectors for upserting to index: {self.index_name}")

            # Process vectors in parallel batches
            with self.pc.Index(self.index_name, pool_threads=30) as index:
                async_results = []
                for i, vectors_chunk in enumerate(self.chunks(vectors_to_upsert, batch_size=200)):
                    logger.info(f"Upserting batch {i+1} with {len(vectors_chunk)} vectors...")
                    result = index.upsert(vectors=vectors_chunk, async_req=True)
                    async_results.append(result)

                try:
                    [async_result.get() for async_result in async_results]
                    logger.info(f"Successfully upserted all vectors for document {document_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error in parallel upsert for document {document_id}: {str(e)}")
                    return False
            
        except Exception as e:
            logger.error(f"Error processing document chunks: {str(e)}")
            return False

    def process_file_stream(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        logger.info(f"Processing file stream: {filename}")
        temp_file_path = None

        try:
            mime_type = self.validate_file(file_content)
            if not mime_type:
                return {"success": False, "error": "Unsupported or invalid file type"}

            # Save file to temp dir
            temp_file_path = self.save_pdf_to_temp(file_content, filename)

            # Upload to S3
            s3_key = self.data_ingestion.upload_file_to_s3(file_content, filename, folder="uploads")

            # Save metadata to MongoDB
            metadata = {
                "filename": filename,
                "s3_key": s3_key,
                "upload_time": ObjectId().generation_time,
                "processed": False,
                "file_type": mime_type,
                "temp_path": temp_file_path
            }
            document_id = self.data_ingestion.save_metadata_to_mongodb(self.pdf_collection, metadata)

            # Load documents based on file type
            if mime_type == 'application/pdf':
                documents = self.load_documents_from_pdf(temp_file_path)
            elif mime_type == 'text/plain':
                documents = self.load_documents_from_txt(temp_file_path)
            else:
                raise ValueError("Unsupported file type passed validation.")

            if not documents:
                logger.warning(f"No documents loaded from file {filename}")
                self.cleanup_failed_processing(document_id, s3_key, temp_file_path)
                return {
                    "success": False,
                    "document_id": document_id,
                    "s3_key": s3_key,
                    "vectorized": False,
                    "error": "Failed to load documents"
                }

            # Process documents (chunking, embedding, Pinecone)
            processing_success = self.process_documents(documents, document_id, filename, s3_key)

            if not processing_success:
                self.cleanup_failed_processing(document_id, s3_key, temp_file_path)
                return {
                    "success": False,
                    "document_id": document_id,
                    "s3_key": s3_key,
                    "vectorized": False,
                    "error": "Failed to process documents"
                }

            # Update DB
            self.data_ingestion.update_document_in_mongodb(
                self.pdf_collection,
                document_id,
                {
                    "processed": True,
                    "document_count": len(documents),
                    "chunk_count": sum(1 for _ in self.data_splitter.split_data(documents))
                }
            )

            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Removed temporary file: {temp_file_path}")

            return {
                "success": True,
                "document_id": document_id,
                "s3_key": s3_key,
                "vectorized": True,
                "document_count": len(documents)
            }

        except Exception as e:
            logger.error(f"Error processing file stream {filename}: {str(e)}")
            if 'document_id' in locals() and 's3_key' in locals():
                self.cleanup_failed_processing(document_id, s3_key, temp_file_path)
            return {"success": False, "error": str(e)}

    def process_multiple_pdfs(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files in parallel
        
        Args:
            files: List of dictionaries with file content and filename
                  Each dict should have 'content' and 'filename' keys
        
        Returns:
            results: List of processing results for each file
        """
        logger.info(f"Processing {len(files)} PDF files")
        
        # Process files in parallel
        futures = []
        for file_info in files:
            future = self.executor.submit(
                self.process_file_stream,
                file_info["content"],
                file_info["filename"]
            )
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        return results

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path: The path to the PDF file
        
        Returns:
            text: The extracted text
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        
        text = self.document_extractor.extract_text_from_file(file_path)
        
        if text:
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
        else:
            logger.warning(f"Failed to extract text from PDF: {file_path}")
            text = ""
        
        return text
        
    def extract_text_from_pdf_stream(self, file_content: bytes) -> str:
        """
        Extract text from a PDF file content without saving to disk
        
        Args:
            file_content: The binary content of the PDF file
        
        Returns:
            text: The extracted text
        """
        logger.info(f"Extracting text from PDF stream")
        
        try:
            # Create a BytesIO object from the file content
            pdf_stream = io.BytesIO(file_content)
            
            # Use PyPDF2 to extract text
            pdf_reader = PdfReader(pdf_stream)
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            if text:
                logger.info(f"Successfully extracted {len(text)} characters from PDF stream")
            else:
                logger.warning("No text extracted from PDF stream")
                text = ""
                
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF stream: {str(e)}")
            return ""

    def store_vectors_in_pinecone(self, vector_id: str, vector_values: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Store precomputed vectors in Pinecone
        
        Args:
            vector_id: Unique ID for the vector
            vector_values: The vector embedding
            metadata: Metadata to store with the vector
        
        Returns:
            success: Whether the operation was successful
        """
        logger.info(f"Storing vector {vector_id} in Pinecone")
        
        try:
            # Upsert vector into Pinecone
            self.pinecone_index.upsert(
                vectors=[
                    {"id": vector_id, "values": vector_values, "metadata": metadata}
                ]
            )
            
            logger.info(f"Successfully stored vector {vector_id} in Pinecone")
            return True
        
        except Exception as e:
            logger.error(f"Error storing vector {vector_id} in Pinecone: {str(e)}")
            return False
            
    # 
    
    def create_embeddings(self, texts) -> List[List[float]]:
        """
        Create embeddings for a single text or a list of texts using Cohere.

        Args:
            texts: A single string or a list of strings (texts) to embed.
            
        Returns:
            embeddings: A list of lists, each containing embedding values for the corresponding text.
        """
        # If a single string is passed, convert it into a list
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Creating embeddings for {len(texts)} texts")

        try:
            # Generate embeddings for the list of texts
            embeddings = self.embeddings.embed_query(texts)
            
            # Check if the response contains embeddings
            

            # Log the dimensions of the embeddings
            logger.info(f"Successfully created embeddings with dimension {len(embeddings[0])} for {len(texts)} texts.")
            
            return embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
