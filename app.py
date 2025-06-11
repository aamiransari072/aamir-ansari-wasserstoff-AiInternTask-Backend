import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ CORS Middleware import
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
from datetime import datetime
import hashlib
import os
import hmac
from io import BytesIO
import urllib.parse
from src.pipeline.pdf_processing_pipeline import PDFProcessingPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.logging_config import logger
from src.utils.environment import check_env_variables
from src.utility import get_signature_key
import boto3
from botocore.client import Config


# Load environment variables and check for required ones
load_dotenv()
check_env_variables()

app = FastAPI(
    title="Chatbot API",
    description="API for processing PDFs and storing them in vector database"
)

# AWS/R2 credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
ENDPOINT_URL = os.getenv('ENDPOINT_URL')

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
    sources: List[Dict[str, Any]]  # Will contain document_id, filename, and s3_url
    success: bool

class DownloadRequest(BaseModel):
    filename: str
    key : str
    

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


@app.post("/api/proxy/download")
async def proxy_download(request: DownloadRequest):
    try:
        # Step 1: Create boto3 S3-compatible client
        s3 = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )

        # Step 2: Generate signed URL
        signed_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': request.key},
            ExpiresIn=3600
        )

        # Step 3: Download from signed URL
        response = requests.get(signed_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch file: {response.text}"
            )

        # Step 4: Stream response to client
        def file_stream():
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        return StreamingResponse(
            file_stream(),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{request.filename}"'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to Chatbot Backend API"}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
