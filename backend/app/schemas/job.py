from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Skill(BaseModel):
    name: str
    proficiency: Optional[str] = None  # e.g., "beginner", "intermediate", "expert"

class JobRoleBase(BaseModel):
    title: str
    department: str
    location: str
    employment_type: str  # e.g., "full-time", "part-time", "contract"
    description: str
    required_skills: List[Skill] = []
    required_experience: Optional[int] = None  # in years
    required_education: Optional[str] = None
    salary_range: Optional[str] = None
    status: str = "active"  # "active", "closed", "draft"

class JobRoleCreate(JobRoleBase):
    pass

class JobRoleUpdate(JobRoleBase):
    pass

class JobRole(JobRoleBase):
    id: str
    posting_date: datetime = Field(default_factory=datetime.now)
    closing_date: Optional[datetime] = None
    created_by: str  # username of the creator

    class Config:
        from_attributes = True 