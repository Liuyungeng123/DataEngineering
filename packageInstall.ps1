# Find Python
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    "python3"
} else {
    Write-Host "Python not found. Please install Python and add it to your PATH."
    exit 1
}

# Create virtual environment
& $pythonCmd -m venv dataEngVenv

# Activate virtual environment
.\dataEngVenv\Scripts\Activate.ps1

Write-Host "Virtual Environment set up and activated: dataEngVenv"

# Install packages
& $pythonCmd -m pip install --upgrade pip
& $pythonCmd -m pip install streamlit cryptography langchain langchain_community openai lagent requests beautifulsoup4 sumy torch nltk

# Download NLTK data
& $pythonCmd -c "import nltk; nltk.download('punkt')"
& $pythonCmd -m nltk.downloader all

Write-Host "Packages installed successfully!"
