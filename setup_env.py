#!/usr/bin/env python3
"""
Setup script to create virtual environment and install dependencies
"""
import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, shell=True):
    """Execute command and return True if successful, False otherwise"""
    try:
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        logger.info(f"Command executed successfully: {command}")
        logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return False

def create_virtual_environment():
    """Create virtual environment"""
    logger.info("Creating virtual environment...")
    
    # Check if python command is available
    python_cmd = "python"
    logger.info(f"Checking Python availability with: {python_cmd} --version")
    
    if not run_command(f"{python_cmd} --version"):
        logger.error(f"Python command '{python_cmd}' not available")
        return False
    
    # Check the presence of venv module
    logger.info("Checking venv module...")
    
    # Remove existing virtual environment if it exists
    import shutil
    if os.path.exists("age_gender_env"):
        logger.info("Removing existing virtual environment...")
        try:
            shutil.rmtree("age_gender_env")
            logger.info("Existing virtual environment removed")
        except Exception as e:
            logger.error(f"Failed to remove existing virtual environment: {e}")
            return False
    
    # Create virtual environment with verbose output
    logger.info(f"Creating virtual environment with: {python_cmd} -m venv age_gender_env")
    if run_command(f"{python_cmd} -m venv age_gender_env"):
        # Verify the virtual environment was created correctly
        scripts_dir = "age_gender_env\\Scripts" if os.name == 'nt' else "age_gender_env/bin"
        if not os.path.exists(scripts_dir):
            logger.error(f"Virtual environment created but {scripts_dir} directory not found")
            return False
        
        # On Windows, check if pip.exe exists
        pip_path = os.path.join(scripts_dir, "pip.exe" if os.name == 'nt' else "pip")
        if not os.path.exists(pip_path):
            logger.error(f"Pip not found at {pip_path}, trying to install pip manually...")
            # Try to get pip
            if not run_command(f"{python_cmd} -m ensurepip --default-pip --upgrade"):
                logger.error("Failed to install pip")
                return False
        
        logger.info("Virtual environment created successfully")
        return True
    else:
        logger.error("Failed to create virtual environment")
        return False

def activate_and_install():
    """Activate virtual environment and install requirements"""
    logger.info("Installing requirements...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        logger.error("requirements.txt file not found")
        return False
    
    # Determine activation script path and commands based on OS
    if os.name == 'nt':  # Windows
        scripts_dir = "age_gender_env\\Scripts"
        python_cmd = f"{scripts_dir}\\python.exe"
        pip_cmd = f"{scripts_dir}\\pip.exe"
    else:  # Unix/Linux/MacOS
        scripts_dir = "age_gender_env/bin"
        python_cmd = f"{scripts_dir}/python"
        pip_cmd = f"{scripts_dir}/pip"
    
    # Check if python exists in the virtual environment
    python_path = python_cmd if os.name != 'nt' else python_cmd.replace("/", "\\")
    if not os.path.exists(python_path):
        logger.error(f"Python not found at {python_path}")
        return False
    
    # Install pip if it doesn't exist
    pip_path = pip_cmd if os.name != 'nt' else pip_cmd.replace("/", "\\")
    if not os.path.exists(pip_path):
        logger.warning(f"Pip not found at {pip_path}, installing pip...")
        if not run_command(f"{python_cmd} -m ensurepip --upgrade"):
            logger.error("Failed to install pip")
            return False
    
    # Upgrade pip
    logger.info("Upgrading pip...")
    if not run_command(f"{python_cmd} -m pip install --upgrade pip"):
        logger.error("Failed to upgrade pip")
        return False
    
    # Install requirements
    logger.info("Installing requirements from requirements.txt...")
    if run_command(f"{python_cmd} -m pip install -r requirements.txt"):
        logger.info("Requirements installed successfully")
        return True
    else:
        logger.error("Failed to install requirements from requirements.txt")
        return False

def create_project_structure():
    """Create necessary project directories"""
    logger.info("Creating project structure...")
    
    directories = [
        "data",
        "models",
        "logs",
        "reports",
        "static/uploads",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main setup function"""
    logger.info("Starting project setup...")
    
    try:
        # Create project structure
        create_project_structure()
        
        # Create virtual environment
        if not create_virtual_environment():
            logger.error("Failed to create virtual environment. Please check your Python installation.")
            sys.exit(1)
        
        # Install requirements
        if not activate_and_install():
            logger.error("Failed to install requirements. Please check the requirements.txt file.")
            sys.exit(1)
        
        logger.info("Setup completed successfully!")
        logger.info("To activate the virtual environment:")
        if os.name == 'nt':
            logger.info("  age_gender_env\\Scripts\\activate")
        else:
            logger.info("  source age_gender_env/bin/activate")
        
        logger.info("Then run: python train_model.py")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()