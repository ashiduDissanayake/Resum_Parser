from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
import PyPDF2
import io
import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
import certifi
import dns.resolver
from datetime import datetime
from pathlib import Path

# Import HybridCVParser instead of ImprovedResumeParser
from hybrid_cv_parser import HybridCVParser

# Load environment variables
load_dotenv()

app = FastAPI(title="Resume Parser Service")

# Initialize MongoDB client
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MONGO_URI environment variable is not set")

# Configure DNS resolver for MongoDB Atlas
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']  # Use Google's DNS

mongo_client = MongoClient(
    mongo_uri,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000
)
db = mongo_client[os.getenv("DATABASE_NAME", "resume_rover_db")]
resumes_collection = db["parsed_resumes"]

class StatusEnum(str, Enum):
    SAVED = "saved"
    INPROGRESS = "inprogress"
    COMPLETED = "completed"

class ResumeData(BaseModel):
    job_id: str
    username: str
    is_verified: bool = False
    ranking_score: Optional[int] = None
    status: StatusEnum = StatusEnum.SAVED

# Extract text from PDF
def extract_text_from_pdf(pdf_file: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def convert_objectid_to_str(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict):
        return {k: convert_objectid_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

def extract_text_from_word(file_content: bytes) -> str:
    try:
        from docx import Document
        document = Document(io.BytesIO(file_content))
        text = []
        for para in document.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing Word file: {str(e)}")


@app.post("/parse")
async def parse_resume_endpoint(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    username: str = Form(...)
):
    try:
        # Create all necessary data internally
        resume_data = {
            "job_id": job_id,
            "username": username,
            "is_verified": False,  # Default value
            "status": "saved",     # Default status
            "ranking_score": None
        }

        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if not file_extension:
            raise HTTPException(status_code=400, detail="File has no extension")
            
        # Process file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(contents)
        elif file_extension in ['.docx', '.doc']:
            text = extract_text_from_word(contents)
        elif file_extension == '.txt':
            text = contents.decode('utf-8')
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}"
            )
        
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable is not set")
        
        # Use HybridCVParser instead of ImprovedResumeParser
        parser = HybridCVParser(api_key=api_key)
        parsed_data = parser.parse(text)  # Note: The method is parse() not parse_resume()
        
        # Create complete record
        resume_record = {
            **resume_data,
            **parsed_data,
            "original_filename": file.filename,
            "upload_date": datetime.now()
        }

        # Save to database
        result = resumes_collection.insert_one(resume_record)
        
        return {
            "success": True,
            "message": "Resume processed successfully",
            "resume_id": str(result.inserted_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)