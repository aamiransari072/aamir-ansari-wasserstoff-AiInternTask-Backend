# Chatbot Backend API

A powerful backend API for processing PDFs and answering queries using advanced AI technologies. This system combines document processing, vector storage, and large language models to provide intelligent responses to user queries.

## 🎯 Objective

The Chatbot Backend API is designed to:
- Process and store PDF documents efficiently
- Create vector embeddings for semantic search
- Provide intelligent answers to user queries based on document content
- Support document retrieval and source attribution
- Enable secure file storage and access

## 🏗️ Architecture

The system consists of several key components:

### 1. PDF Processing Pipeline
- Handles PDF document ingestion and processing
- Extracts text content from PDFs
- Splits documents into manageable chunks
- Creates vector embeddings using OpenAI's embedding model
- Stores vectors in Pinecone for efficient retrieval
- Manages document metadata in MongoDB
- Stores original files in S3-compatible storage (Cloudflare R2)

### 2. Query Pipeline
- Processes user queries using semantic search
- Retrieves relevant document chunks from Pinecone
- Uses OpenAI's GPT-4 model for generating responses
- Provides source attribution for answers
- Supports document reranking for improved relevance

### 3. API Layer
- FastAPI-based REST API
- CORS-enabled for frontend integration
- Secure file upload and download endpoints
- Health check and monitoring endpoints

## 🔄 Workflow

### PDF Processing Flow

```
User Upload
   │
   ▼
validate_pdf()
   │
   ▼
save_pdf_to_temp()
   │
   ▼
upload_file_to_s3() ─► S3 Bucket
   │
   ▼
save_metadata_to_mongodb() ─► MongoDB
   │
   ▼
load_documents_from_pdf()
   │
   ▼
split_data() ─► Recursive chunks
   │
   ▼
embed_query() ─► Embeddings
   │
   ▼
upsert() ─► Pinecone Vector DB
   │
   ▼
update_document_in_mongodb(processed=True)
   │
   ▼
Clean up temp file
```

### PDF Processing Decision Flow

```
         ┌────────────┐
         │  Upload    │
         └─────┬──────┘
               │
               ▼
      Validate PDF Format
               │
               ▼
  Try PyPDFLoader (fails on scanned PDF)
               │
               ▼
Fallback to UnstructuredPDFLoader (uses OCR)
               │
        ┌──────┴─────┐
        │Text Found? │──No─► Clean up + Error
        └──────┬─────┘
               │Yes
               ▼
     Proceed with chunking, embedding, vector storage
```

### Query Processing Flow

```
User Query
   │
   ▼
Process Query
   │
   ▼
Retrieve Relevant Documents from Pinecone
   │
   ▼
Generate Context from Retrieved Documents
   │
   ▼
Send to OpenAI GPT-4
   │
   ▼
Generate Response
   │
   ▼
Prepare Source Information
   │
   ▼
Return Response with:
   ├─► Answer
   ├─► Source Documents
   └─► S3 URLs for Download
```

### Source Document Access
- Each source document is accessible via a secure S3 URL
- URLs are generated using signed requests for security
- Users can download original documents through the `/api/proxy/download` endpoint
- Source attribution includes:
  - Document ID
  - Original filename
  - S3 URL for download
  - Relevance score

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB
- Pinecone account
- OpenAI API key
- Cloudflare R2 (or S3-compatible storage)
- Virtual environment (recommended)

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name

# MongoDB Configuration
MONGODB_URI=your_mongodb_uri
MONGODB_DB_NAME=your_database_name

# S3/R2 Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
ENDPOINT_URL=your_endpoint_url
```

### Running the Application

1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

### API Endpoints

- `GET /`: Welcome message
- `GET /healthz`: Health check endpoint
- `POST /process-pdfs`: Upload and process PDF documents
- `POST /query`: Submit queries for processing
- `POST /api/proxy/download`: Download processed documents

## 🧪 Testing

The project includes test scripts for both pipelines:

1. Test PDF Processing:
```bash
python src/examples/test_pdf_pipeline.py <path-to-pdf>
```

2. Test Query Pipeline:
```bash
python src/examples/test_query_pipeline.py "your query"
```

## 📝 Author

@aamiransari072

