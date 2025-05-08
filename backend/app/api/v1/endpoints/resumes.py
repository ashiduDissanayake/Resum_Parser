from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from typing import List
from bson import ObjectId
import requests
from pathlib import Path
import os
from ....schemas.resume import ResumeUploadResponse
from ....schemas.user import User
from ....db.mongodb import mongodb
from ..deps import get_current_active_user

router = APIRouter()

# Configuration for the parser service
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://localhost:8001")

@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload and parse a resume document.
    
    The file will be sent to the resume parser service for processing
    and the results will be stored in the database.
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix
        if file_extension.lower() not in ['.pdf', '.docx', '.doc', '.txt']:
            return ResumeUploadResponse(
                success=False,
                message=f"Unsupported file format: {file_extension}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Create a new multipart request to the parser service
        files = {"file": (file.filename, file_content, file.content_type)}
        params = {"user_id": current_user.username}
        
        # Send the file to the parser service
        response = requests.post(
            f"{PARSER_SERVICE_URL}/parse",
            files=files,
            params=params
        )
        
        # Check the response
        if response.status_code == 200:
            parser_response = response.json()
            print(parser_response)
            if parser_response.get("success"):
                return ResumeUploadResponse(
                    success=True,
                    message="Resume uploaded and parsed successfully",
                    resume_id=parser_response.get("data", {}).get("id")
                )
        
        # Handle error from parser service
        return ResumeUploadResponse(
            success=False,
            message=f"Resume parsing failed: {response.json().get('message', 'Unknown error')}"
        )
        
    except Exception as e:
        return ResumeUploadResponse(
            success=False,
            message=f"Error uploading resume: {str(e)}"
        )

@router.get("/", response_model=List[dict])
async def get_user_resumes(current_user: User = Depends(get_current_active_user)):
    """Get all resumes uploaded by the current user."""
    print(current_user)
    resumes = await mongodb.get_collection("parsed_resumes").find({"user_id": current_user.username}).to_list(length=None)
    
    # Convert ObjectIds to strings
    for resume in resumes:
        resume["id"] = str(resume.pop("_id"))
    
    return resumes

@router.get("/{resume_id}", response_model=dict)
async def get_resume(
    resume_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific resume by ID."""
    try:
        resume = await mongodb.get_collection("parsed_resumes").find_one({"_id": ObjectId(resume_id)})
        
        # Check if resume exists and belongs to the user
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
            
        if resume.get("user_id") != current_user.username and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this resume"
            )
        
        # Convert ObjectId to string
        resume["id"] = str(resume.pop("_id"))
        
        return resume
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving resume: {str(e)}") 