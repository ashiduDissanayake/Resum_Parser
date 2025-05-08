#!/usr/bin/env python3
"""
Simplified script for testing the Hybrid CV Parser with an embedded sample CV.
"""
import os
import json
from dotenv import load_dotenv
from hybrid_cv_parser import HybridCVParser
import time

# Load environment variables
load_dotenv()

# Sample CV text embedded directly in the script
SAMPLE_CV = """
Resume
Candidate Profile
Name: Nimal WijesingheEmail: nimal.wijesinghe@gmail.comPhone: +94 76 543 2109Address: 10/2, Station Road, Panadura, Sri LankaGitHub: github.com/nimalwLinkedIn: linkedin.com/in/nimalwijesinghe  
Education



Qualification
Institution
Period
Details



BSc (Hons) in Software Engineering
University of Kelaniya, Sri Lanka
2019 - 2023
CGPA: 3.5/4.0, Project: "Smart Library Management System"


G.C.E. Advanced Level (Technology Stream)
Mahinda College, Galle
2016 - 2018
2 A’s (ICT, Engineering Technology), B (Science for Technology)


Professional Experience



Role
Company
Period
Responsibilities



Junior Software Developer
Nexlify IT Solutions, Colombo, Sri Lanka
June 2023 - Present
- Built web apps using PHP (CodeIgniter) and JavaScript (Vue.js) for 5K+ users- Optimized SQLite queries, improving data retrieval by 25%- Assisted in API integration with third-party services- Contributed to 6+ sprint cycles in Agile teams


Internship - Software Development
IFS R&D International, Colombo, Sri Lanka
January 2022 - June 2022
- Developed ERP module features using C# and .NET- Wrote unit tests with NUnit, achieving 80% coverage- Supported database migrations in SQL Server


Skills

Languages: PHP, JavaScript, C#, SQL  
Frameworks: CodeIgniter, Vue.js, .NET Core  
Tools: Git, Visual Studio, Postman  
Databases: SQLite, SQL Server  
Practices: Agile, Test-Driven Development, API Integration


"""

def main():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key provided. Please set GOOGLE_API_KEY in .env file")
        return
    
    try:
        print("Starting CV parsing test with embedded sample CV")
        
        # Output file path
        output_file = 'parsed_cv_result.json'
        
        print("Initializing NLP models (spaCy + BERT + Gemini)...")
        
        # Initialize parser
        start_time = time.time()
        cv_parser = HybridCVParser(api_key=api_key)
        init_time = time.time() - start_time
        print(f"Models initialized in {init_time:.2f} seconds")
        
        # Parse CV
        print("Parsing CV with hybrid approach...")
        start_time = time.time()
        result = cv_parser.parse(SAMPLE_CV)
        parse_time = time.time() - start_time
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Parsing completed in {parse_time:.2f} seconds")
        print(f"Results saved to {output_file}")
        
        # Print summary
        print("\nParsing summary:")
        print(f"Name: {result.get('name')}")
        print(f"Email: {result.get('email')}")
        print(f"Education entries: {len(result.get('education', []))}")
        print(f"Work experience entries: {len(result.get('work_experience', []))}")
        
        skills = result.get('skills', {})
        print(f"Technical skills: {len(skills.get('technical_skills', []))}")
        print(f"Soft skills: {len(skills.get('soft_skills', []))}")
        print(f"Languages: {len(skills.get('languages', []))}")
        
        confidence = result.get('metadata', {}).get('confidence_scores', {})
        print(f"Overall confidence: {confidence.get('overall', 0):.2f}")
        
        # Show first few skills as example
        if skills.get('technical_skills'):
            print("\nSample technical skills:")
            for skill in skills.get('technical_skills')[:5]:
                print(f"- {skill}")
        
        print("\nNLP models used:")
        model_usage = result.get('metadata', {}).get('model_usage', {})
        print(f"- spaCy: {model_usage.get('spacy_model', 'en_core_web_sm')}")
        print(f"- BERT: {model_usage.get('bert_model', 'unknown')}")
        print(f"- Gemini: {model_usage.get('gemini_model', 'unknown')}")
        
    except Exception as e:
        print(f"Error parsing CV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()