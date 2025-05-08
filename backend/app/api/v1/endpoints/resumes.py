from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Request, Form
from typing import List
from bson import ObjectId
import requests
from pathlib import Path
import os
from ....schemas.resume import ResumeUploadResponse
from ....schemas.user import User
from ....db.mongodb import mongodb
from ..deps import get_current_active_user
from typing import Optional
import json
from io import BytesIO

router = APIRouter()

# Configuration for the parser service
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://localhost:8001")

@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    job_id: str = Form(...),  # Add this parameter
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload and parse a resume document.
    """
    try:
        # No need to extract job_id from form_data anymore
        # since it's directly available as a parameter
        
        if not job_id:
            return ResumeUploadResponse(
                success=False,
                message="Job ID is required."
            )

          # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.pdf', '.docx', '.doc', '.txt']:
            return ResumeUploadResponse(
                success=False,
                message=f"Unsupported file format: {file_extension}"
            )
        
        # Read file content once
        file_content = await file.read()
        
        # Prepare the request to parser service
        files = {'file': (file.filename, BytesIO(file_content))}
        
        # Use job_id directly in your data dictionary
        data = {
            'job_id': job_id,
            'username': current_user.username
        }
        # Send to parser service
        response = requests.post(
            f"{PARSER_SERVICE_URL}/parse",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            return ResumeUploadResponse(
                success=True,
                message="Resume uploaded successfully",
                resume_id=response.json().get("resume_id")
            )
        else:
            return ResumeUploadResponse(
                success=False,
                message=f"Parser error: {response.text}"
            )
            
    except Exception as e:
        return ResumeUploadResponse(
            success=False,
            message=f"Upload failed: {str(e)}"
        )
    
@router.get("/", response_model=List[dict])
async def get_user_resumes(current_user: User = Depends(get_current_active_user)):
    """Get all resumes uploaded by the current user."""
    resumes = await mongodb.get_collection("parsed_resumes").find({"username": current_user.username}).to_list(length=None)
    
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