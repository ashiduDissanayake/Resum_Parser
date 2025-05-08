from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import language_v1
from google.oauth2 import service_account
import PyPDF2
import io
import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from typing import Dict, Any
from main import ImprovedResumeParser
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

# Initialize the resume parser
resume_parser = ImprovedResumeParser()

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def convert_objectid_to_str(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ObjectId to string in the response data."""
    if isinstance(data, dict):
        return {k: convert_objectid_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

@app.post("/parse")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """
    Parse a resume PDF file and extract structured information.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(contents)
        
        # Parse resume using the ImprovedResumeParser
        parsed_data = resume_parser.parse_resume(text)
        
        # Store in MongoDB
        result = resumes_collection.insert_one(parsed_data)
        
        # Convert ObjectId to string in the response
        response_data = {
            "success": True,
            "message": "Resume parsed successfully",
            "resume_id": str(result.inserted_id),
            "parsed_data": convert_objectid_to_str(parsed_data)
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing resume: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)