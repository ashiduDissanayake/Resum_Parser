#!/bin/bash

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "GOOGLE_CREDENTIALS_PATH=google-credentials.json
MONGO_URI=mongodb+srv://ashidudissanayake1:chP0CyGcYR89zDeg@cluster0.4dg71cd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
DATABASE_NAME=resume_rover_db" > .env
fi

# Run the server
echo "Starting Resume Parser service on port 8001..."
uvicorn app:app --reload --port 8001 