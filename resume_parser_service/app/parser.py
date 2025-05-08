import os
import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
import PyPDF2
import docx
from tempfile import NamedTemporaryFile
import io
from datetime import datetime
import dateutil.parser
import dateutil.relativedelta
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from .models import (
    ContactInfo, Education, WorkExperience, Skill, 
    Project, Certification, ParsedResume, SentimentInfo,
    SectionAnalysis, ResumeAnalysis, ParserResponse
)

class ConfidenceLevel(Enum):
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.2

@dataclass
class ExtractedInfo:
    value: Any
    confidence: float
    source: str

class RankingData:
    """Simple class for resume ranking data"""
    def __init__(self, 
                 name: str = None,
                 skills: List[str] = None,
                 total_experience_years: float = 0.0,
                 highest_education: str = None,
                 major_field: str = None):
        self.name = name
        self.skills = skills or []
        self.total_experience_years = total_experience_years
        self.highest_education = highest_education
        self.major_field = major_field
    
    def to_dict(self):
        return {
            "name": self.name,
            "skills": self.skills,
            "total_experience_years": self.total_experience_years,
            "highest_education": self.highest_education,
            "major_field": self.major_field
        }

def extract_ranking_data(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key ranking data from parsed resume"""
    # Initialize ranking data
    ranking_data = RankingData()
    
    # Extract name
    if parsed_data.get("contact_info") and parsed_data["contact_info"].get("name"):
        ranking_data.name = parsed_data["contact_info"]["name"]
    
    # Extract skills
    if parsed_data.get("skills"):
        ranking_data.skills = [skill["name"] for skill in parsed_data["skills"]]
    
    # Calculate total experience
    total_years = 0.0
    if parsed_data.get("work_experience"):
        for exp in parsed_data["work_experience"]:
            if exp.get("start_date") and exp.get("end_date"):
                try:
                    start = datetime.strptime(exp["start_date"], "%Y-%m-%d")
                    end = datetime.strptime(exp["end_date"], "%Y-%m-%d")
                    delta = dateutil.relativedelta.relativedelta(end, start)
                    # Convert to decimal years
                    years = delta.years + (delta.months / 12) + (delta.days / 365)
                    total_years += years
                except Exception:
                    # If date parsing fails, try to extract years from position or description
                    pass
    
    ranking_data.total_experience_years = round(total_years, 1)
    
    # Find highest education degree
    highest_degree = None
    field_of_study = None
    
    # Education ranking dictionary (higher number = higher level)
    degree_rankings = {
        "high school": 1,
        "associate": 2, 
        "diploma": 2,
        "bachelor": 3, "bsc": 3, "ba": 3, "bs": 3, "b.eng": 3, "b.e": 3, "b.tech": 3,
        "master": 4, "msc": 4, "ma": 4, "mba": 4, "m.eng": 4, "m.e": 4, "m.tech": 4,
        "phd": 5, "doctorate": 5, "doctor": 5
    }
    
    current_highest_rank = 0
    
    if parsed_data.get("education"):
        for edu in parsed_data["education"]:
            degree_text = (edu.get("degree") or "").lower()
            for degree_name, rank in degree_rankings.items():
                if degree_name in degree_text and rank > current_highest_rank:
                    current_highest_rank = rank
                    highest_degree = edu.get("degree")
                    field_of_study = edu.get("field_of_study")
    
    ranking_data.highest_education = highest_degree
    ranking_data.major_field = field_of_study
    
    return ranking_data.to_dict()

class ResumeParser:
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize the resume parser with a SpaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # If model not found, download it
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
            
        # Initialize stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # Add custom stopwords
        self.stopwords.update(['experience', 'work', 'job', 'position', 'role', 'responsibilities'])
        
        # Patterns for contact info extraction
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'(?:\+\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
        self.linkedin_pattern = re.compile(r'(?:linkedin\.com/in/|linkedin:)\s*([\w-]+)')
        self.github_pattern = re.compile(r'(?:github\.com/|github:)\s*([\w-]+)')
        self.url_pattern = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        
        # Section patterns for better detection
        self.section_patterns = {
            "education": r"(?:^|\n)(?:Education|Academic|Qualification|University|College|Degree)(?:\n|$)",
            "experience": r"(?:^|\n)(?:Experience|Employment|Work History|Career|Professional Experience|Work Experience|Internship)(?:\n|$)",
            "skills": r"(?:^|\n)(?:Skills|Technical Skills|Abilities|Expertise|Competencies|Programming Languages|Technical Fields|Tools & Services|Frameworks & Libraries)(?:\n|$)",
            "projects": r"(?:^|\n)(?:Projects|Project Experience|Portfolio|Research Projects|Personal Projects)(?:\n|$)",
            "certifications": r"(?:^|\n)(?:Certifications|Certificates|Credentials|Licenses|Courses|Training)(?:\n|$)",
            "highlights": r"(?:^|\n)(?:Highlights|Achievements|Accomplishments|Awards)(?:\n|$)",
            "extra": r"(?:^|\n)(?:Extra-Curricular|Activities|Interests|Hobbies)(?:\n|$)",
            "references": r"(?:^|\n)(?:References|Referees)(?:\n|$)"
        }
        
        # Common degree types with confidence scores
        self.degree_types = {
            "bachelor": 0.9,
            "master": 0.9,
            "phd": 0.9,
            "doctorate": 0.9,
            "associate": 0.8,
            "diploma": 0.8,
            "bsc": 0.9,
            "msc": 0.9,
            "mba": 0.9,
            "ba": 0.8,
            "bs": 0.8,
            "ma": 0.8,
            "md": 0.8,
            "jd": 0.8,
            "b.eng": 0.9,
            "m.eng": 0.9,
            "b.e": 0.9,
            "m.e": 0.9,
            "b.tech": 0.9,
            "m.tech": 0.9
        }
        
        # Common job titles with confidence scores
        self.job_titles = {
            "engineer": 0.9,
            "developer": 0.9,
            "manager": 0.9,
            "director": 0.9,
            "analyst": 0.9,
            "consultant": 0.9,
            "architect": 0.9,
            "designer": 0.9,
            "scientist": 0.9,
            "researcher": 0.9,
            "specialist": 0.9,
            "lead": 0.9,
            "head": 0.9,
            "chief": 0.9,
            "officer": 0.9,
            "coordinator": 0.9,
            "assistant": 0.9,
            "intern": 0.9,
            "teaching assistant": 0.9,
            "instructor": 0.9,
            "lecturer": 0.9,
            "professor": 0.9
        }
        
        # Technical skills with categories and confidence scores
        self.technical_skills = {
            "programming": {
                "python": 0.9,
                "java": 0.9,
                "javascript": 0.9,
                "typescript": 0.9,
                "c++": 0.9,
                "c#": 0.9,
                "ruby": 0.9,
                "php": 0.9,
                "swift": 0.9,
                "kotlin": 0.9,
                "go": 0.9,
                "rust": 0.9
            },
            "web": {
                "html": 0.9,
                "css": 0.9,
                "react": 0.9,
                "angular": 0.9,
                "vue": 0.9,
                "node.js": 0.9,
                "express": 0.9,
                "django": 0.9,
                "flask": 0.9,
                "fastapi": 0.9,
                "spring": 0.9,
                "laravel": 0.9,
                "reactjs": 0.9,
                "next.js": 0.9
            },
            "database": {
                "sql": 0.9,
                "mysql": 0.9,
                "postgresql": 0.9,
                "mongodb": 0.9,
                "redis": 0.9,
                "cassandra": 0.9,
                "oracle": 0.9,
                "sqlite": 0.9,
                "prisma": 0.9
            },
            "cloud": {
                "aws": 0.9,
                "azure": 0.9,
                "gcp": 0.9,
                "docker": 0.9,
                "kubernetes": 0.9,
                "terraform": 0.9,
                "ansible": 0.9,
                "jenkins": 0.9,
                "gitlab ci": 0.9,
                "sagemaker": 0.9
            },
            "ai_ml": {
                "machine learning": 0.9,
                "deep learning": 0.9,
                "tensorflow": 0.9,
                "pytorch": 0.9,
                "scikit-learn": 0.9,
                "numpy": 0.9,
                "pandas": 0.9,
                "opencv": 0.9,
                "cnn": 0.9,
                "llm": 0.9,
                "computer vision": 0.9,
                "nlp": 0.9,
                "natural language processing": 0.9,
                "rag": 0.9
            },
            "tools": {
                "git": 0.9,
                "jira": 0.9,
                "confluence": 0.9,
                "jenkins": 0.9,
                "github": 0.9,
                "bitbucket": 0.9,
                "vscode": 0.9,
                "intellij": 0.9,
                "eclipse": 0.9,
                "power apps": 0.9,
                "figma": 0.9,
                "canva": 0.9,
                "hugging face": 0.9,
                "crewai": 0.9
            }
        }
    
    def _find_all_sections(self, text: str) -> Dict[str, str]:
        """Find all sections in the resume text."""
        sections = {}
        text_lower = text.lower()
        
        for section_name, pattern in self.section_patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                start_pos = match.end()
                # Find next section
                next_pos = len(text)
                for next_pattern in self.section_patterns.values():
                    next_match = re.search(next_pattern, text[start_pos:], re.IGNORECASE)
                    if next_match:
                        next_pos = min(next_pos, start_pos + next_match.start())
                sections[section_name] = text[start_pos:next_pos].strip()
        
        return sections
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        bullet_points = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                bullet_points.append(line.lstrip('•-*').strip())
        return bullet_points
    
    def _extract_name(self, text: str) -> Tuple[Optional[str], float]:
        """Extract name from the first few lines of the resume."""
        first_lines = text.split('\n')[:5]  # Check first 5 lines
        for line in first_lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip lines with contact information
            if any(keyword in line.lower() for keyword in ['phone:', 'email:', 'address:', 'linkedin:', 'github:']):
                continue
            
            # Process the line with SpaCy
            doc_line = self.nlp(line)
            
            # Check for PERSON entities
            for ent in doc_line.ents:
                if ent.label_ == "PERSON":
                    return ent.text, 0.9
            
            # If no PERSON entity found, use the first line that looks like a name
            if not any(char.isdigit() for char in line) and len(line.split()) >= 2:
                return line, 0.7
        
        return None, 0.0
    
    def _extract_social_links(self, text: str) -> Dict[str, str]:
        """Extract social media and website links."""
        links = {}
        
        # Extract LinkedIn
        linkedin_match = self.linkedin_pattern.search(text)
        if linkedin_match:
            username = linkedin_match.group(1)
            links['linkedin'] = f"https://linkedin.com/in/{username}"
        
        # Extract GitHub
        github_match = self.github_pattern.search(text)
        if github_match:
            username = github_match.group(1)
            links['github'] = f"https://github.com/{username}"
        
        # Extract other URLs
        url_matches = self.url_pattern.finditer(text)
        for match in url_matches:
            url = match.group(0)
            parsed = urlparse(url)
            if parsed.netloc not in ['linkedin.com', 'github.com']:
                links['website'] = url
        
        return links
    
    def extract_contact_info(self, doc) -> ContactInfo:
        """Extract contact information using SpaCy and regex."""
        text = doc.text
        
        # Extract name
        name, name_confidence = self._extract_name(text)
        
        # Extract email
        email_matches = self.email_pattern.findall(text)
        email_info = self._calculate_confidence(
            email_matches[0] if email_matches else None,
            self._validate_email
        )
        
        # Extract phone
        phone_matches = self.phone_pattern.findall(text)
        phone_info = self._calculate_confidence(
            phone_matches[0] if phone_matches else None,
            self._validate_phone
        )
        
        # Extract social links
        social_links = self._extract_social_links(text)
        
        # Extract location
        location = None
        location_confidence = 0.0
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                # Check if it's near contact info
                if email_info.value and abs(ent.start_char - text.find(email_info.value)) < 100:
                    location = ent.text
                    location_confidence = 0.8
                    break
                elif phone_info.value and abs(ent.start_char - text.find(phone_info.value)) < 100:
                    location = ent.text
                    location_confidence = 0.8
                    break
        
        return ContactInfo(
            name=name,
            email=email_info.value,
            phone=phone_info.value,
            location=location,
            linkedin=social_links.get('linkedin'),
            github=social_links.get('github'),
            website=social_links.get('website')
        )
    
    def extract_education(self, doc) -> List[Education]:
        """Extract education information using SpaCy and patterns."""
        text = doc.text
        education_list = []
        
        # Find education section
        sections = self._find_all_sections(text)
        education_section = sections.get("education")
        
        if education_section:
            # Split into potential education entries - look for institution names
            entries = re.split(r'\n(?=[A-Z][^a-z]+(?:\s+[A-Z][^a-z]+)*\s*(?:University|College|School|Institute|Academy))', education_section)
            
            for entry in entries:
                if not entry.strip():
                    continue
                    
                # Process with SpaCy
                doc_entry = self.nlp(entry)
                
                # Extract institution
                institution = None
                institution_confidence = 0.0
                for ent in doc_entry.ents:
                    if ent.label_ == "ORG":
                        institution = ent.text
                        institution_confidence = 0.9
                        break
                
                if institution:
                    # Extract degree
                    degree = None
                    degree_confidence = 0.0
                    for degree_type, confidence in self.degree_types.items():
                        if degree_type.lower() in entry.lower():
                            # Get the full degree text
                            degree_match = re.search(rf'{degree_type}[^,\n]*', entry, re.IGNORECASE)
                            if degree_match:
                                degree = degree_match.group(0)
                                degree_confidence = confidence
                                break
                    
                    # Extract dates
                    dates = self._extract_dates(entry)
                    start_date = None
                    end_date = None
                    date_confidence = 0.0
                    
                    if dates:
                        start_date = dates[0]
                        end_date = dates[1] if len(dates) > 1 else None
                        date_confidence = 0.9
                    
                    # Extract field of study - improved pattern
                    field_of_study = None
                    field_confidence = 0.0
                    field_patterns = [
                        r'in\s+([^,\n]+?)(?:\s*\d{4}|$)',
                        r'major(?:ing)?\s+in\s+([^,\n]+?)(?:\s*\d{4}|$)',
                        r'field\s+of\s+study:\s*([^,\n]+?)(?:\s*\d{4}|$)',
                        r'specializing\s+in\s+([^,\n]+?)(?:\s*\d{4}|$)',
                        r'focus(?:ing)?\s+on\s+([^,\n]+?)(?:\s*\d{4}|$)',
                        r'([^,\n]+?)(?:\s*degree|\s*program)(?:\s*\d{4}|$)'
                    ]
                    
                    for pattern in field_patterns:
                        match = re.search(pattern, entry, re.IGNORECASE)
                        if match:
                            field_of_study = match.group(1).strip()
                            # Remove any dates from field of study
                            field_of_study = re.sub(r'\d{4}\s*[-–]\s*(?:present|current|ongoing|\d{4})', '', field_of_study).strip()
                            field_confidence = 0.8
                            break
                    
                    # Extract GPA if present
                    gpa_match = re.search(r'(?:GPA|CGPA)[:\s]*(\d+\.\d+)(?:/\d+\.\d+)?', entry, re.IGNORECASE)
                    if gpa_match:
                        gpa = gpa_match.group(1)
                        if degree:
                            degree += f" | GPA: {gpa}"
                            degree_confidence = min(degree_confidence + 0.1, 1.0)
                    
                    education_list.append(
                        Education(
                            institution=institution,
                            degree=degree,
                            field_of_study=field_of_study,
                            start_date=start_date,
                            end_date=end_date
                        )
                    )
        
        return education_list
    
    def extract_work_experience(self, doc) -> List[WorkExperience]:
        """Extract work experience using SpaCy and patterns."""
        text = doc.text
        experience_list = []
        
        # Find experience section
        sections = self._find_all_sections(text)
        experience_section = sections.get("experience")
        
        if experience_section:
            # Split into potential experience entries - look for company names or positions
            entries = re.split(r'\n(?=[A-Z][^a-z]+(?:\s+[A-Z][^a-z]+)*\s*(?:Inc\.?|LLC|Ltd\.?|Company|Corp\.?|University|College|School|Institute|Academy)|(?:\n[A-Z][^a-z]+(?:\s+[A-Z][^a-z]+)*\s*(?:Engineer|Developer|Manager|Director|Analyst|Consultant|Architect|Designer|Scientist|Researcher|Specialist|Lead|Head|Chief|Officer|Coordinator|Assistant|Intern|Instructor|Lecturer|Professor)))', experience_section)
            
            for entry in entries:
                if not entry.strip():
                    continue
                    
                # Process with SpaCy
                doc_entry = self.nlp(entry)
                
                # Extract company and position
                company = None
                position = None
                company_confidence = 0.0
                position_confidence = 0.0
                
                # First try to find position using job titles
                for title, confidence in self.job_titles.items():
                    if title.lower() in entry.lower():
                        position_match = re.search(rf'[^,\n]*{title}[^,\n]*', entry, re.IGNORECASE)
                        if position_match:
                            position = position_match.group(0)
                            position_confidence = confidence
                            break
                
                # Then try to find company
                for ent in doc_entry.ents:
                    if ent.label_ == "ORG":
                        # Skip if the organization is part of the position
                        if position and ent.text in position:
                            continue
                        company = ent.text
                        company_confidence = 0.9
                        break
                
                if company or position:
                    # Extract dates
                    dates = self._extract_dates(entry)
                    start_date = None
                    end_date = None
                    date_confidence = 0.0
                    
                    if dates:
                        start_date = dates[0]
                        end_date = dates[1] if len(dates) > 1 else None
                        date_confidence = 0.9
                    
                    # Extract description
                    description = None
                    description_confidence = 0.0
                    
                    # Get bullet points with proper formatting
                    bullet_points = self._extract_bullet_points(entry)
                    if bullet_points:
                        description = "\n• " + "\n• ".join(bullet_points)
                        description_confidence = 0.8
                    
                    # Extract tech stack if present
                    tech_stack_match = re.search(r'(?:tech stack|technologies|tools):\s*([^,\n]+(?:\s*,\s*[^,\n]+)*)', entry, re.IGNORECASE)
                    if tech_stack_match:
                        tech_stack = tech_stack_match.group(1).strip()
                        if description:
                            description += f"\n\nTech Stack: {tech_stack}"
                        else:
                            description = f"Tech Stack: {tech_stack}"
                        description_confidence = min(description_confidence + 0.1, 1.0)
                    
                    experience_list.append(
                        WorkExperience(
                            company=company,
                            position=position,
                            start_date=start_date,
                            end_date=end_date,
                            description=description
                        )
                    )
        
        return experience_list
    
    def extract_projects(self, doc) -> List[Project]:
        """Extract project information."""
        text = doc.text
        projects = []
        
        # Find projects section
        sections = self._find_all_sections(text)
        projects_section = sections.get("projects")
        
        if projects_section:
            # Split by project title pattern
            project_blocks = re.split(r'\n(?=[A-Z][^a-z]+(?:\s+[A-Z][^a-z]+)*\s*(?:Project|System|Application|Platform|Tool|Framework|Library|API|Service|Website|App|Software|Solution))', projects_section)
            
            for block in project_blocks:
                if not block.strip():
                    continue
                
                lines = block.split('\n')
                title_line = lines[0].strip()
                
                # Extract project name and type
                project_name = title_line
                project_type = None
                if '|' in title_line:
                    parts = title_line.split('|')
                    project_name = parts[0].strip()
                    project_type = parts[1].strip()
                
                # Extract dates
                dates = self._extract_dates(block)
                
                # Extract description (bullet points)
                bullet_points = self._extract_bullet_points(block)
                description = "\n• " + "\n• ".join(bullet_points) if bullet_points else None
                
                # Extract technologies
                technologies = []
                for category, skills in self.technical_skills.items():
                    for skill in skills:
                        if skill in block.lower():
                            technologies.append(skill)
                
                # Extract URL if present
                url_match = self.url_pattern.search(block)
                url = url_match.group(0) if url_match else None
                
                # Create project dictionary
                project_dict = {
                    "name": project_name,
                    "type": project_type,
                    "description": description,
                    "start_date": dates[0] if dates else None,
                    "end_date": dates[-1] if len(dates) > 1 else None,
                    "technologies": technologies,
                    "url": url
                }
                
                # Create Project instance using Pydantic model
                projects.append(Project(**project_dict))
        
        return projects
    
    def extract_certifications(self, doc) -> List[Certification]:
        """Extract certification information."""
        text = doc.text
        certifications = []
        
        # Find certifications section
        sections = self._find_all_sections(text)
        certs_section = sections.get("certifications")
        
        if certs_section:
            # Split into individual certifications
            cert_blocks = re.split(r'\n(?=[A-Z][^a-z]+(?:\s+[A-Z][^a-z]+)*\s*(?:Certification|Certificate|License|Course|Training|Program|Professional|Development))', certs_section)
            
            for block in cert_blocks:
                if not block.strip():
                    continue
                
                # Extract certification name
                name = block.split('\n')[0].strip()
                
                # Extract provider
                provider = None
                provider_match = re.search(r'(?:by|from|provider|issued by|offered by):\s*([^,\n]+)', block, re.IGNORECASE)
                if provider_match:
                    provider = provider_match.group(1).strip()
                
                # Extract date
                dates = self._extract_dates(block)
                date = dates[0] if dates else None
                
                # Extract URL if present
                url_match = self.url_pattern.search(block)
                url = url_match.group(0) if url_match else None
                
                # Create certification dictionary
                cert_dict = {
                    "name": name,
                    "provider": provider,
                    "date": date,
                    "url": url
                }
                
                # Create Certification instance using Pydantic model
                certifications.append(Certification(**cert_dict))
        
        return certifications
    
    def extract_skills(self, doc) -> List[Skill]:
        """Extract skills from resume text."""
        text = doc.text.lower()
        skills_list = []
        
        # Find skills section
        sections = self._find_all_sections(text)
        skills_section = sections.get("skills")
        
        if skills_section:
            # Extract skills from each category
            for category, skills in self.technical_skills.items():
                for skill, confidence in skills.items():
                    if skill in skills_section:
                        # Look for skill level if present
                        level = None
                        level_confidence = 0.0
                        level_patterns = [
                            r'(\d+)\s*(?:years?|yrs?)',
                            r'(expert|advanced|intermediate|beginner)',
                            r'(\d+)\s*(?:out of|/)\s*\d+'
                        ]
                        
                        for pattern in level_patterns:
                            match = re.search(pattern, skills_section, re.IGNORECASE)
                            if match:
                                level = match.group(1)
                                level_confidence = 0.8
                                break
                        
                        # Create skill dictionary
                        skill_dict = {
                            "name": skill,
                            "level": level
                        }
                        
                        # Create Skill instance using Pydantic model
                        skills_list.append(Skill(**skill_dict))
        
        return skills_list
    
    def parse(self, file_content, file_extension: str) -> ParserResponse:
        """Main parsing function to extract structured information from a resume."""
        try:
            # Extract text based on file type
            raw_text = self.extract_text(file_content, file_extension)
            
            if not raw_text:
                return ParserResponse(
                    success=False,
                    message="Failed to extract text from the document"
                )
            
            # Process with SpaCy
            doc = self.nlp(raw_text)
            
            # Extract different components
            contact_info = self.extract_contact_info(doc)
            education = self.extract_education(doc)
            work_experience = self.extract_work_experience(doc)
            skills = self.extract_skills(doc)
            projects = self.extract_projects(doc)
            certifications = self.extract_certifications(doc)
            
            # Create parsed resume object
            parsed_resume = ParsedResume(
                contact_info=contact_info,
                education=education,
                work_experience=work_experience,
                skills=skills,
                projects=projects,
                certifications=certifications,
                raw_text=raw_text
            )
            
            # Perform analysis
            analysis = self.analyze_resume(parsed_resume)
            
            # Extract ranking data
            ranking_data = extract_ranking_data(parsed_resume.model_dump())
            
            # Convert to response format
            return ParserResponse(
                success=True,
                parsed_data=parsed_resume.model_dump(),
                analysis=analysis.model_dump(),
                ranking_data=ranking_data
            )
            
        except Exception as e:
            return ParserResponse(
                success=False,
                message=f"Error parsing resume: {str(e)}"
            )

    def extract_text_from_pdf(self, file_content) -> str:
        """Extract text from PDF bytes."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_content) -> str:
        """Extract text from DOCX bytes."""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return ""
    
    def extract_text(self, file_content, file_extension: str) -> str:
        """Extract text based on file type."""
        if file_extension.lower() == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_extension.lower() in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_content)
        elif file_extension.lower() == '.txt':
            try:
                return file_content.decode('utf-8')
            except Exception as e:
                print(f"Error decoding text file: {str(e)}")
                return ""
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _validate_email(self, email: str) -> Tuple[bool, float]:
        """Validate email format and return confidence score."""
        if not email:
            return False, 0.0
        
        # Basic email validation
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return False, 0.0
        
        # Check for common email providers
        common_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'edu']
        confidence = 0.5  # Base confidence
        
        if any(provider in email.lower() for provider in common_providers):
            confidence += 0.3
        
        # Check for educational institution emails
        if '.edu' in email.lower() or '.ac.' in email.lower():
            confidence += 0.2
        
        return True, min(confidence, 1.0)
    
    def _validate_phone(self, phone: str) -> Tuple[bool, float]:
        """Validate phone number format and return confidence score."""
        if not phone:
            return False, 0.0
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Check length (assuming international numbers are 10-15 digits)
        if not (10 <= len(digits) <= 15):
            return False, 0.0
        
        return True, 0.8
    
    def _validate_date(self, date_str: str) -> Tuple[bool, float]:
        """Validate date format and return confidence score."""
        if not date_str:
            return False, 0.0
        
        try:
            dateutil.parser.parse(date_str)
            return True, 0.9
        except:
            return False, 0.0
    
    def _calculate_confidence(self, value: Any, validation_func) -> ExtractedInfo:
        """Calculate confidence score for extracted information."""
        is_valid, confidence = validation_func(value)
        return ExtractedInfo(
            value=value if is_valid else None,
            confidence=confidence,
            source="validation"
        )

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using various patterns."""
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}',
            r'\d{4}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*',
            r'\d{4}\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)',
            r'(?:present|current|ongoing)',
            r'\d{4}\s*–\s*(?:present|current|ongoing)'
        ]
        
        # Find all dates in the text
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Parse the date
                    date_str = match.group(0)
                    if date_str.lower() in ['present', 'current', 'ongoing']:
                        dates.append(datetime.now().strftime("%Y-%m-%d"))
                    else:
                        parsed_date = dateutil.parser.parse(date_str)
                        dates.append(parsed_date.strftime("%Y-%m-%d"))
                except:
                    continue
        
        return sorted(dates)

    def _analyze_sentiment(self, text: str) -> SentimentInfo:
        """Analyze sentiment of text using TextBlob and custom analysis."""
        blob = TextBlob(text)
        
        # Calculate sentiment score
        sentiment_score = blob.sentiment.polarity
        
        # Calculate magnitude (how strong the sentiment is)
        magnitude = abs(sentiment_score)
        
        # Extract keywords
        doc = self.nlp(text)
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                token.text.lower() not in self.stopwords and 
                len(token.text) > 2):
                keywords.append(token.text.lower())
        
        # Generate summary
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:2]) if sentences else ""
        
        return SentimentInfo(
            score=sentiment_score,
            magnitude=magnitude,
            keywords=list(set(keywords)),
            summary=summary
        )

    def _analyze_section(self, text: str) -> SectionAnalysis:
        """Analyze a section of the resume."""
        sentiment = self._analyze_sentiment(text)
        
        # Extract key points
        doc = self.nlp(text)
        key_points = []
        
        # Look for bullet points and important phrases
        for sent in doc.sents:
            if any(marker in sent.text for marker in ['•', '-', '*', '→']):
                key_points.append(sent.text.strip())
            elif len(sent.text.split()) > 5:  # Only consider substantial sentences
                key_points.append(sent.text.strip())
        
        # Calculate confidence based on content quality
        confidence = min(0.5 + (len(key_points) * 0.1), 1.0)
        
        return SectionAnalysis(
            sentiment=sentiment,
            key_points=key_points[:5],  # Limit to top 5 key points
            confidence=confidence
        )

    def analyze_resume(self, parsed_resume: ParsedResume) -> ResumeAnalysis:
        """Perform comprehensive analysis of the parsed resume."""
        # Analyze overall sentiment
        overall_text = parsed_resume.raw_text
        overall_sentiment = self._analyze_sentiment(overall_text)
        
        # Analyze each section
        section_analyses = {}
        
        # Analyze work experience
        if parsed_resume.work_experience:
            exp_text = "\n".join([
                f"{exp.position} at {exp.company}: {exp.description}"
                for exp in parsed_resume.work_experience
            ])
            section_analyses['work_experience'] = self._analyze_section(exp_text)
        
        # Analyze education
        if parsed_resume.education:
            edu_text = "\n".join([
                f"{edu.degree} in {edu.field_of_study} at {edu.institution}"
                for edu in parsed_resume.education
            ])
            section_analyses['education'] = self._analyze_section(edu_text)
        
        # Analyze skills
        if parsed_resume.skills:
            skills_text = ", ".join([skill.name for skill in parsed_resume.skills])
            section_analyses['skills'] = self._analyze_section(skills_text)
        
        # Analyze projects
        if parsed_resume.projects:
            proj_text = "\n".join([
                f"{proj.name}: {proj.description}"
                for proj in parsed_resume.projects
            ])
            section_analyses['projects'] = self._analyze_section(proj_text)
        
        # Identify strengths
        strengths = []
        for section, analysis in section_analyses.items():
            if analysis.sentiment.score > 0.3:
                strengths.extend(analysis.key_points[:2])
        
        # Identify areas for improvement
        areas_for_improvement = []
        for section, analysis in section_analyses.items():
            if analysis.sentiment.score < 0:
                areas_for_improvement.extend(analysis.key_points[:2])
        
        # Identify skill gaps
        skill_gaps = []
        if 'skills' in section_analyses:
            current_skills = set(skill.name.lower() for skill in parsed_resume.skills)
            desired_skills = {
                'python', 'java', 'javascript', 'react', 'node.js', 'aws',
                'docker', 'kubernetes', 'machine learning', 'data analysis'
            }
            skill_gaps = list(desired_skills - current_skills)
        
        # Generate recommendations
        recommendations = []
        if skill_gaps:
            recommendations.append(f"Consider adding these skills: {', '.join(skill_gaps[:3])}")
        if areas_for_improvement:
            recommendations.append("Focus on improving areas with negative sentiment")
        if overall_sentiment.score < 0.3:
            recommendations.append("Consider making the resume more achievement-focused")
        
        # Create ResumeAnalysis using Pydantic model
        return ResumeAnalysis(
            overall_sentiment=overall_sentiment,
            section_analyses=section_analyses,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            skill_gaps=skill_gaps,
            recommendations=recommendations
        )

    def parse_for_ranking(self, file_content, file_extension: str) -> Dict[str, Any]:
        """Parse a resume and return only the data needed for ranking."""
        try:
            # Use the existing parser to get full details
            full_response = self.parse(file_content, file_extension)
            
            if not full_response.success:
                return {
                    "success": False,
                    "message": full_response.message or "Parsing failed"
                }
            
            # Extract the ranking-specific data
            ranking_data = extract_ranking_data(full_response.parsed_data)
            
            return {
                "success": True,
                "ranking_data": ranking_data
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error in parsing for ranking: {str(e)}"
            }