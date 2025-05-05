from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from bson import ObjectId
from ....schemas.job import JobRole, JobRoleCreate, JobRoleUpdate
from ....schemas.user import User
from ....db.mongodb import mongodb
from ....api.deps import get_current_active_user, get_admin_user

router = APIRouter()

@router.post("/", response_model=JobRole)
async def create_job(
    job: JobRoleCreate,
    current_user: User = Depends(get_admin_user)
):
    job_data = job.dict()
    job_data["created_by"] = current_user.username
    
    result = await mongodb.get_collection("jobs").insert_one(job_data)
    created_job = await mongodb.get_collection("jobs").find_one({"_id": result.inserted_id})
    created_job["id"] = str(created_job.pop("_id"))
    return JobRole(**created_job)

@router.get("/", response_model=List[JobRole])
async def get_jobs(
    department: Optional[str] = None,
    status: str = "active",
    current_user: User = Depends(get_current_active_user)
):
    query = {"status": status}
    if department:
        query["department"] = department
    
    jobs = await mongodb.get_collection("jobs").find(query).to_list(length=None)
    for job in jobs:
        job["id"] = str(job.pop("_id"))
    return [JobRole(**job) for job in jobs]

@router.get("/{job_id}", response_model=JobRole)
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    try:
        job = await mongodb.get_collection("jobs").find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job["id"] = str(job.pop("_id"))
        return JobRole(**job)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job ID: {str(e)}")

@router.put("/{job_id}", response_model=JobRole)
async def update_job(
    job_id: str,
    job: JobRoleUpdate,
    current_user: User = Depends(get_current_active_user)
):
    try:
        existing_job = await mongodb.get_collection("jobs").find_one({"_id": ObjectId(job_id)})
        if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if existing_job["created_by"] != current_user.username and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this job"
            )
        
        job_data = job.dict()
        job_data["created_by"] = existing_job["created_by"]
        
        await mongodb.get_collection("jobs").update_one(
            {"_id": ObjectId(job_id)},
            {"$set": job_data}
        )
        
        updated_job = await mongodb.get_collection("jobs").find_one({"_id": ObjectId(job_id)})
        updated_job["id"] = str(updated_job.pop("_id"))
        return JobRole(**updated_job)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating job: {str(e)}")

@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    try:
        existing_job = await mongodb.get_collection("jobs").find_one({"_id": ObjectId(job_id)})
        if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if existing_job["created_by"] != current_user.username and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this job"
            )
        
        await mongodb.get_collection("jobs").delete_one({"_id": ObjectId(job_id)})
        return {"message": "Job deleted successfully", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error deleting job: {str(e)}") 