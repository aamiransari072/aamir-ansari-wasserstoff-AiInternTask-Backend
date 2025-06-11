from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ CORS Middleware import
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
import os
from io import BytesIO

from src.pipeline.pdf_processing_pipeline import PDFProcessingPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.logging_config import logger
from src.utils.environment import check_env_variables

# Load environment variables and check for required ones
load_dotenv()
check_env_variables()

app = FastAPI(
    title="Chatbot API",
    description="API for processing PDFs and storing them in vector database"
)

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the PDF processing pipeline
pdf_pipeline = PDFProcessingPipeline()

# Initialize the query pipeline
query_pipeline = QueryPipeline()

class PDFResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    s3_key: Optional[str] = None
    vectorized: Optional[bool] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    success: bool

@app.post("/process-pdfs", response_model=List[PDFResponse])
async def process_pdfs(files: List[UploadFile] = File(...)):
    """
    Process multiple PDF files and return processing results for each file.
    """
    logger.info(f"Received request to process {len(files)} PDFs")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

    try:
        results = []
        for file in files:
            content = await file.read()
            result = pdf_pipeline.process_file_stream(content, file.filename)
            results.append(result)
        return results

    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query using the query pipeline.
    """
    logger.info(f"Received query request: {request.query}")

    try:
        result = query_pipeline.answer_query(request.query)
        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to Vedic Pedia AI API"}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
