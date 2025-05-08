"""
Advanced CV Parser combining Gemini AI with spaCy NLP and BERT transformers.
"""
import os
import json
import logging
import google.generativeai as genai
import spacy
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridCVParser:
    """
    Advanced CV Parser that combines Gemini AI with spaCy NLP and BERT-based verification
    for extracting structured data from CVs.
    """
    
    def __init__(self, api_key: Optional[str] = None, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the hybrid CV parser with Gemini AI, spaCy, and BERT.
        
        Args:
            api_key: Google API key for Gemini. If None, tries to get from environment.
            spacy_model: Name of the spaCy model to use for NLP tasks.
        """
        # Initialize Gemini API
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Gemini AI initialized successfully")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model {spacy_model}: {e}. Downloading...")
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        # Initialize BERT
        try:
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.bert_model.eval()  # Set the model to evaluation mode
            logger.info("Loaded BERT model successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def parse(self, cv_text: str) -> Dict[str, Any]:
        """
        Parse a CV using our hybrid approach: Gemini AI for initial parsing,
        then spaCy and BERT for verification and enhancement.
        
        Args:
            cv_text: The CV text to parse in any format
            
        Returns:
            Dict containing parsed CV data
        """
        if not cv_text or not isinstance(cv_text, str):
            raise ValueError("CV text must be a non-empty string")
        
        try:
            logger.info("Starting CV parsing process...")
            
            # STEP 1: Initial parsing with Gemini AI
            logger.info("Performing initial parsing with Gemini AI")
            initial_data = self._parse_with_gemini(cv_text)
            
            # STEP 2: Process with spaCy for NER verification
            logger.info("Verifying entities with spaCy NLP")
            spacy_doc = self.nlp(cv_text)
            spacy_verified_data = self._verify_with_spacy(initial_data, spacy_doc)
            
            # STEP 3: Enhance with BERT for higher-level semantic understanding
            logger.info("Enhancing data with BERT transformer models")
            enhanced_data = self._enhance_with_bert(spacy_verified_data, cv_text)
            
            # STEP 4: Post-process and finalize the data
            logger.info("Post-processing and finalizing data")
            final_data = self._post_process_data(enhanced_data)
            
            logger.info("Successfully parsed CV using hybrid approach")
            return final_data
            
        except Exception as e:
            logger.error(f"Error parsing CV: {e}")
            raise RuntimeError(f"Failed to parse CV: {e}")
    
    def _parse_with_gemini(self, cv_text: str) -> Dict[str, Any]:
        """
        Perform initial parsing with Gemini AI.
        
        Args:
            cv_text: The CV text to parse
            
        Returns:
            Initial parsed data from Gemini
        """
        # Create prompt for Gemini
        prompt = f"""
        You are an expert CV/resume parser that extracts structured data from curriculum vitae or resumes.
        
        Extract the following information from the CV text and return it in valid JSON format:
        
        1. name: Full name of the person
        2. email: Email address
        3. phone: Phone number
        4. location: Physical location/address
        5. education: List of education entries, each with:
           - degree: The obtained degree or qualification
           - institution: Name of the institution
           - dates: Date range in format "YYYY - YYYY" or "YYYY - Present"
           - details: Any additional details like GPA, honors, etc.
        6. work_experience: List of work experiences, each with:
           - position: Job title/position
           - company: Company name
           - location: Job location
           - dates: Employment period in format "YYYY - YYYY" or "YYYY - Present" or "Month YYYY - Month YYYY"
           - description: List of responsibilities and achievements
        7. skills: List of skills, categorized if possible into:
           - technical_skills: Technical abilities, programming languages, etc.
           - soft_skills: Communication, teamwork, etc.
           - languages: Language proficiencies
        8. summary: Brief professional summary or objective if present
        
        CRITICAL: Output ONLY valid JSON without any explanation, preamble or other text. Follow this exact format:
        
        {{
          "name": "...",
          "email": "...",
          "phone": "...",
          "location": "...",
          "education": [
            {{
              "degree": "...",
              "institution": "...",
              "dates": "...",
              "details": "..."
            }}
          ],
          "work_experience": [
            {{
              "position": "...",
              "company": "...",
              "location": "...",
              "dates": "...",
              "description": ["...", "...", "..."]
            }}
          ],
          "skills": {{
            "technical_skills": ["...", "...", "..."],
            "soft_skills": ["...", "...", "..."],
            "languages": ["...", "...", "..."]
          }},
          "summary": "..."
        }}
        
        Extract as much relevant information as possible from the CV. If a field is not found, use an empty string or array as appropriate.
        
        Here's the CV text to parse:
        
        {cv_text}
        """
        
        # Call Gemini API
        response = self.gemini_model.generate_content(prompt)
        
        # Extract JSON
        json_data = self._extract_json_from_response(response.text)
        return json_data
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON data from Gemini response."""
        try:
            # Clean up the response to extract just the JSON part
            json_text = response_text.strip()
            
            # If response is wrapped in code blocks, extract just the JSON
            if json_text.startswith("```json"):
                json_text = json_text.split("```json", 1)[1]
                if "```" in json_text:
                    json_text = json_text.split("```", 1)[0]
            elif json_text.startswith("```"):
                json_text = json_text.split("```", 1)[1]
                if "```" in json_text:
                    json_text = json_text.split("```", 1)[0]
            
            # Parse JSON
            data = json.loads(json_text)
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
    
    def _verify_with_spacy(self, data: Dict[str, Any], doc) -> Dict[str, Any]:
        """
        Verify extracted data using spaCy's NER capabilities.
        
        Args:
            data: Initially parsed data
            doc: spaCy document
            
        Returns:
            Verified data with spaCy
        """
        verified_data = data.copy()
        
        # Verify name using PERSON entity
        if not verified_data.get("name"):
            person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if person_entities:
                verified_data["name"] = person_entities[0]
                logger.info(f"spaCy identified name: {person_entities[0]}")
        
        # Verify email using pattern recognition in spaCy
        if not verified_data.get("email"):
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, doc.text)
            if emails:
                verified_data["email"] = emails[0]
                logger.info(f"spaCy pattern matching identified email: {emails[0]}")
        
        # Verify location using GPE entities
        if not verified_data.get("location"):
            locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
            if locations:
                verified_data["location"] = ", ".join(locations[:2])
                logger.info(f"spaCy identified location: {verified_data['location']}")
        
        # Verify organizations for education and work experience
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if orgs:
            # Check if any organizations are missing in education
            for i, edu in enumerate(verified_data.get("education", [])):
                if not edu.get("institution") and i < len(orgs):
                    verified_data["education"][i]["institution"] = orgs[i]
                    logger.info(f"spaCy identified educational institution: {orgs[i]}")
            
            # Check if any organizations are missing in work experience
            for i, work in enumerate(verified_data.get("work_experience", [])):
                if not work.get("company") and i < len(orgs):
                    verified_data["work_experience"][i]["company"] = orgs[i]
                    logger.info(f"spaCy identified company: {orgs[i]}")
        
        # Verify dates
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        if dates:
            # Try to match date patterns for experience and education
            for date in dates:
                if "-" in date or "–" in date:  # Looks like a date range
                    # Find entries without dates
                    for exp in verified_data.get("work_experience", []):
                        if not exp.get("dates"):
                            exp["dates"] = date
                            logger.info(f"spaCy identified work date range: {date}")
                            break
                    
                    for edu in verified_data.get("education", []):
                        if not edu.get("dates"):
                            edu["dates"] = date
                            logger.info(f"spaCy identified education date range: {date}")
                            break
        
        return verified_data
    
    def _get_bert_embeddings(self, text: str) -> np.ndarray:
        """
        Generate BERT embeddings for text.
        
        Args:
            text: Text to encode
            
        Returns:
            NumPy array of embeddings
        """
        # Tokenize and get BERT embeddings
        tokens = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        
        # Get the [CLS] token embedding (represents the entire sequence)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings[0]  # Return as a 1D array
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using BERT.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        emb1 = self._get_bert_embeddings(text1)
        emb2 = self._get_bert_embeddings(text2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity
    
    def _enhance_with_bert(self, data: Dict[str, Any], cv_text: str) -> Dict[str, Any]:
        """
        Enhance parsed data using BERT for semantic understanding.
        
        Args:
            data: Verified data from spaCy
            cv_text: Original CV text
            
        Returns:
            Enhanced data with BERT
        """
        enhanced_data = data.copy()
        
        # Use BERT to classify skills into technical vs soft skills if not already classified
        if "skills" in enhanced_data and isinstance(enhanced_data["skills"], list):
            logger.info("Using BERT to classify skills")
            technical_skills = []
            soft_skills = []
            
            # Example technical and soft skill references for comparison
            tech_reference = "programming languages frameworks libraries technical tools Python Java JavaScript"
            soft_reference = "communication teamwork leadership time management organization interpersonal"
            
            for skill in enhanced_data["skills"]:
                # Compare skill to technical and soft skill references
                tech_similarity = self._compute_similarity(skill, tech_reference)
                soft_similarity = self._compute_similarity(skill, soft_reference)
                
                if tech_similarity > soft_similarity:
                    technical_skills.append(skill)
                    logger.info(f"BERT classified '{skill}' as technical skill")
                else:
                    soft_skills.append(skill)
                    logger.info(f"BERT classified '{skill}' as soft skill")
            
            # Replace flat list with categorized skills
            enhanced_data["skills"] = {
                "technical_skills": technical_skills,
                "soft_skills": soft_skills,
                "languages": []  # Would need more complex logic to identify languages
            }
        
        # Use BERT to extract or verify summary if missing
        if not enhanced_data.get("summary"):
            logger.info("Using BERT to identify professional summary")
            
            # Find paragraphs that might be summaries
            paragraphs = cv_text.split("\n\n")
            summary_reference = "professional summary profile about me overview career objective"
            
            best_score = 0
            best_summary = ""
            
            for para in paragraphs:
                if 20 < len(para) < 500:  # Reasonable summary length
                    similarity = self._compute_similarity(para, summary_reference)
                    if similarity > best_score:
                        best_score = similarity
                        best_summary = para
            
            if best_score > 0.5:  # Only use if reasonably confident
                enhanced_data["summary"] = best_summary
                logger.info(f"BERT identified summary with confidence {best_score:.2f}")
        
        # Use BERT to verify job descriptions
        for i, exp in enumerate(enhanced_data.get("work_experience", [])):
            # Check if descriptions are in first person or bullets
            if isinstance(exp.get("description"), list) and exp["description"]:
                for j, desc in enumerate(exp["description"]):
                    # Verify if it's a likely job responsibility
                    resp_reference = "responsible for managed developed created implemented led designed"
                    similarity = self._compute_similarity(desc, resp_reference)
                    
                    if similarity > 0.5:
                        logger.info(f"BERT verified job description {j+1} for position {i+1}")
                    else:
                        logger.info(f"BERT flagged job description {j+1} for position {i+1} as suspicious")
        
        return enhanced_data
    
    def _post_process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process and finalize the data.
        
        Args:
            data: Enhanced data from BERT
            
        Returns:
            Final processed data
        """
        # Initialize result with metadata
        result = {
            "metadata": {
                "parser_version": "3.0-hybrid",
                "parsing_method": "gemini-spacy-bert",
                "processed_at": self._get_current_timestamp()
            }
        }
        
        # Clean and process contact information
        result["name"] = self._clean_text(data.get("name", ""))
        result["email"] = self._clean_email(data.get("email", ""))
        result["phone"] = self._clean_phone(data.get("phone", ""))
        result["location"] = self._clean_text(data.get("location", ""))
        result["summary"] = self._clean_text(data.get("summary", ""))
        
        # Process education
        result["education"] = []
        for edu in data.get("education", []):
            education_entry = {
                "degree": self._clean_text(edu.get("degree", "")),
                "institution": self._clean_text(edu.get("institution", "")),
                "dates": self._clean_text(edu.get("dates", "")),
                "details": self._clean_text(edu.get("details", ""))
            }
            if education_entry["degree"] or education_entry["institution"]:
                result["education"].append(education_entry)
        
        # Process work experience
        result["work_experience"] = []
        for work in data.get("work_experience", []):
            # Ensure description is always a list
            if isinstance(work.get("description"), str):
                description = [work.get("description")]
            elif isinstance(work.get("description"), list):
                description = work.get("description")
            else:
                description = []
            
            # Clean each description item
            description = [self._clean_text(item) for item in description if item]
            
            work_entry = {
                "position": self._clean_text(work.get("position", "")),
                "company": self._clean_text(work.get("company", "")),
                "location": self._clean_text(work.get("location", "")),
                "dates": self._clean_text(work.get("dates", "")),
                "description": description
            }
            if work_entry["position"] or work_entry["company"]:
                result["work_experience"].append(work_entry)
        
        # Process skills
        skills = data.get("skills", {})
        if isinstance(skills, dict):
            result["skills"] = {
                "technical_skills": [self._clean_text(s) for s in skills.get("technical_skills", []) if s],
                "soft_skills": [self._clean_text(s) for s in skills.get("soft_skills", []) if s],
                "languages": [self._clean_text(s) for s in skills.get("languages", []) if s]
            }
        elif isinstance(skills, list):
            # Handle case where skills is a flat list
            result["skills"] = {
                "technical_skills": [self._clean_text(s) for s in skills if s],
                "soft_skills": [],
                "languages": []
            }
        else:
            result["skills"] = {
                "technical_skills": [],
                "soft_skills": [],
                "languages": []
            }
        
        # Calculate confidence scores
        result["metadata"]["confidence_scores"] = self._calculate_confidence(result)
        
        # Add model usage information
        result["metadata"]["model_usage"] = {
            "spacy_model": self.nlp.meta["name"],
            "bert_model": "bert-base-uncased",
            "gemini_model": "gemini-1.5-pro"
        }
        
        return result
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for each section of the parsed data."""
        confidence = {}
        
        # Contact information confidence
        contact_fields = ["name", "email", "phone", "location"]
        contact_score = sum(1 for field in contact_fields if data.get(field)) / len(contact_fields)
        confidence["contact"] = round(contact_score, 2)
        
        # Education confidence
        education = data.get("education", [])
        education_score = min(1.0, len(education) / 2) if education else 0
        confidence["education"] = round(education_score, 2)
        
        # Work experience confidence
        work = data.get("work_experience", [])
        work_score = min(1.0, len(work) / 3) if work else 0
        confidence["experience"] = round(work_score, 2)
        
        # Skills confidence
        skills = data.get("skills", {})
        all_skills = []
        for skill_type in ["technical_skills", "soft_skills", "languages"]:
            all_skills.extend(skills.get(skill_type, []))
        skills_score = min(1.0, len(all_skills) / 10) if all_skills else 0
        confidence["skills"] = round(skills_score, 2)
        
        # Overall confidence
        weights = {
            "contact": 0.3,
            "education": 0.2,
            "experience": 0.3,
            "skills": 0.2
        }
        overall_score = sum(confidence[key] * weights[key] for key in weights)
        confidence["overall"] = round(overall_score, 2)
        
        return confidence
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text fields."""
        if not text or not isinstance(text, str):
            return ""
        return text.strip()
    
    def _clean_email(self, email: str) -> str:
        """Validate and clean email address."""
        if not email or not isinstance(email, str):
            return ""
        
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email.strip()):
            return email.strip()
        return ""
    
    def _clean_phone(self, phone: str) -> str:
        """Validate and clean phone number."""
        if not phone or not isinstance(phone, str):
            return ""
        
        import re
        cleaned = re.sub(r'[^\d\+\(\)\-\s]', '', phone)
        return cleaned.strip()
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()