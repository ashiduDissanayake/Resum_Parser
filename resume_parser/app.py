from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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

@app.post("/parse")
async def parse_resume_endpoint(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        # Get form data
        form_data = await request.form()
        job_id = form_data.get("job_id")
        username = form_data.get("username")
        
        if not job_id or not username:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields"
            )

        # Create all necessary data internally
        resume_data = {
            "job_id": job_id,
            "username": username,
            "is_verified": False,  # Default value
            "status": "saved",     # Default status
            "ranking_score": None
        }

        # Process file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")
        
        text = extract_text_from_pdf(contents)  # Implement other file types
        
        # Create complete record
        resume_record = {
            **resume_data,
            "parsed_data": {"extracted_text": text},
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
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
