from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class WorkExperience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class Skill(BaseModel):
    name: str
    level: Optional[str] = None

class Project(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    technologies: Optional[List[str]] = None
    url: Optional[str] = None

class Certification(BaseModel):
    name: str
    provider: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None

class ParsedResume(BaseModel):
    contact_info: ContactInfo
    education: List[Education]
    work_experience: List[WorkExperience]
    skills: List[Skill]
    projects: List[Project]
    certifications: List[Certification]
    raw_text: str

class SentimentInfo(BaseModel):
    score: float
    magnitude: float
    keywords: List[str]
    summary: Optional[str] = None

class SectionAnalysis(BaseModel):
    sentiment: SentimentInfo
    key_points: List[str]
    confidence: float

class ResumeAnalysis(BaseModel):
    overall_sentiment: SentimentInfo
    section_analyses: Dict[str, SectionAnalysis]
    strengths: List[str]
    areas_for_improvement: List[str]
    skill_gaps: List[str]
    recommendations: List[str]

class RankingData(BaseModel):
    name: Optional[str] = None
    skills: List[str] = []
    total_experience_years: float = 0.0
    highest_education: Optional[str] = None
    major_field: Optional[str] = None

class ParserResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    parsed_data: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    ranking_data: Optional[Dict[str, Any]] = None