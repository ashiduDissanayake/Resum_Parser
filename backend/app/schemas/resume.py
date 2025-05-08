from pydantic import BaseModel
from typing import Optional

class ResumeUploadResponse(BaseModel):
    success: bool
    message: str
    resume_id: Optional[str] = None