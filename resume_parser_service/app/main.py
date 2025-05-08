from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

from .parser import ResumeParser
from .models import ParserResponse

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MongoDB connection
mongo_client = None
resumes_collection = None

try:
    # Get MongoDB URI from environment variable
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test the connection
        mongo_client.server_info()
        db = mongo_client.resume_rover_db
        resumes_collection = db.parsed_resumes
        logger.info("Successfully connected to MongoDB")
    else:
        logger.warning("MONGO_URI not set. MongoDB functionality will be disabled.")
except (ConnectionFailure, ConfigurationError) as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    mongo_client = None
    resumes_collection = None

# Initialize app
app = FastAPI(
    title="ResumeRover Parser Service", 
    description="Resume parsing microservice using SpaCy",
    version="1.0.0",
    debug=os.getenv("DEBUG", "False").lower() == "true"
)

# Add CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parser
resume_parser = ResumeParser()

@app.get("/")
async def root():
    return {"message": "ResumeRover Parser Service is running"}

@app.get("/health")
async def health_check():
    mongo_status = "connected" if mongo_client else "disconnected"
    return {
        "status": "healthy",
        "mongodb": mongo_status,
        "debug": os.getenv("DEBUG", "False").lower() == "true"
    }

@app.post("/parse", response_model=ParserResponse)
async def parse_resume(file: UploadFile = File(...), user_id: str = None):
    """
    Parse a resume document and extract structured information.
    
    - **file**: Resume file (PDF, DOCX, or TXT)
    - **user_id**: Optional user ID to associate with the parsed resume
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix
        if file_extension.lower() not in ['.pdf', '.docx', '.doc', '.txt']:
            return ParserResponse(
                success=False,
                message=f"Unsupported file format: {file_extension}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Parse the resume
        parsed_resume = resume_parser.parse(file_content, file_extension)
        
        # Store in MongoDB if user_id is provided and MongoDB is connected
        if user_id and resumes_collection:
            try:
                resume_data = parsed_resume.model_dump()
                resume_data["user_id"] = user_id
                resume_data["filename"] = file.filename
                
                # Insert into database
                result = resumes_collection.insert_one(resume_data)
                logger.info(f"Saved parsed resume to database with ID: {result.inserted_id}")
            except Exception as db_error:
                logger.error(f"Failed to save to database: {str(db_error)}")
                # Continue with the response even if database save fails
        
        return parsed_resume
        
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        return ParserResponse(
            success=False,
            message=f"Failed to parse resume: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug
    )