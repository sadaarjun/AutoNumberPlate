# Instructions to Upload ANPR System to GitHub

To push this project to GitHub, follow these steps:

## Step 1: Download the Project
First, download all the project files from Replit. You can do this by:
1. Click on the three dots in the file browser in Replit
2. Select "Download as zip"
3. Extract the zip file on your local computer

## Step 2: Set Up Git Repository
In your local project directory, initialize a Git repository and commit all files:

```bash
git init
git add .
git commit -m "Initial commit of ANPR system"
```

## Step 3: Connect to GitHub
Connect your local repository to the GitHub repository:

```bash
git remote add origin https://github.com/sadaarjun/AutoNumberPlate.git
git branch -M main
```

## Step 4: Push to GitHub
Push your code to GitHub:

```bash
git push -u origin main
```

If you encounter authentication issues, you may need to use a personal access token or set up SSH authentication.

## Project Structure
The ANPR system contains the following main components:

- **Flask Web Application**: Main web interface for the system
- **Camera Management**: Handles multiple cameras with selection UI
- **ANPR Processing**: License plate recognition logic
- **Database Models**: Vehicle and log tracking
- **API Endpoints**: For system control and data retrieval

## Key Features Implemented
- Society name customization
- Admin password management
- Multi-camera configuration and switching
- Dashboard with real-time updates
- Vehicle and log management
- System settings configuration

The code is organized in a modular structure with separate files for different components.