# Complete Setup Guide - Sign Language Recognizer

This guide covers all steps to set up the project on a fresh Windows laptop.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Java** | 21 LTS | Spring Boot backend |
| **Python** | 3.10 or 3.11 | ML/MediaPipe scripts |
| **Maven** | 3.9+ | Build tool (included via mvnw) |
| **Git** | Latest | Version control |
| **FastAPI** | Latest | Python web framework |
| **PyTorch** | 2.0+ | ML training & inference |
| **MediaPipe** | Latest | Hand/pose detection |
| **OpenCV** | Latest | Video processing |
| **ONNX Runtime** | 1.17.1 (Java) | Model inference |

---

## Installation Steps

### STEP 1: Install Java 21

#### Option A: Using Windows Package Manager (Recommended)
```powershell
winget install Oracle.JDK.21
```

#### Option B: Manual Installation
1. Download from: https://www.oracle.com/java/technologies/downloads/
2. Choose "Windows x64 Installer"
3. Run installer and select "Add to PATH"
4. Restart PowerShell

**Verify Installation:**
```powershell
java -version
# Expected output: java version "21" or similar
```

---

### STEP 2: Install Python 3.11

#### Option A: Using Windows Package Manager (Recommended)
```powershell
winget install Python.Python.3.11
```

#### Option B: Manual Installation
1. Download from: https://www.python.org/downloads/
2. Choose Python 3.11 (Windows installer)
3. **Important:** Check "Add Python to PATH" during installation
4. Restart PowerShell

**Verify Installation:**
```powershell
python --version
# Expected output: Python 3.11.x or similar

pip --version
# Expected output: pip 23.x or similar
```

---

### STEP 3: Clone/Navigate to Project

```powershell
# Navigate to your project directory
cd C:\Users\lavib\Downloads\sign-language-recognizer

# Verify you're in the right location
Get-Location
# Should show: C:\Users\lavib\Downloads\sign-language-recognizer

# List contents to verify
Get-ChildItem
# Should see: pom.xml, mvnw, mvnw.cmd, mediapipe_service, src, plan.md, HELP.md
```

---

### STEP 4: Create Python Virtual Environment

```powershell
# Navigate to project root
cd C:\Users\lavib\Downloads\sign-language-recognizer

# Create virtual environment (named .venv311)
python -m venv .venv311

# Activate virtual environment (Windows PowerShell)
.\.venv311\Scripts\Activate.ps1

# If you get execution policy error, run this once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify activation - prompt should show (.venv311) prefix
# Example: (.venv311) PS C:\Users\lavib\Downloads\sign-language-recognizer>
```

---

### STEP 5: Upgrade pip and Install Python Dependencies

```powershell
# Make sure virtual environment is activated
# (you should see (.venv311) in prompt)

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all Python dependencies
pip install fastapi uvicorn[standard] mediapipe opencv-python numpy torch torchvision python-multipart

# Verify installations
pip list
# Should show: fastapi, uvicorn, mediapipe, opencv-python, numpy, torch, torchvision, python-multipart
```

**Alternative: Using requirements.txt**
```powershell
# First, navigate to mediapipe_service
cd mediapipe_service

# Install from requirements.txt
pip install -r requirements.txt

# Navigate back to project root
cd ..
```

---

### STEP 6: Verify Maven Setup

```powershell
# Deactivate Python virtual environment (optional)
deactivate

# Navigate to project root (if not already there)
cd C:\Users\lavib\Downloads\sign-language-recognizer

# Test Maven wrapper
.\mvnw --version
# Expected output: Apache Maven 3.9.x or similar
```

---

### STEP 7: Build Java Project

```powershell
# Full build with tests
.\mvnw clean install

# OR skip tests for faster build (first time)
.\mvnw clean install -DskipTests

# Wait 2-5 minutes for dependencies to download
```

**Expected Output:**
```
[INFO] BUILD SUCCESS
[INFO] Total time: X.XXs
[INFO] Finished at: YYYY-MM-DD'T'HH:MM:SS...
```

---

### STEP 8: (Optional) Install Git

```powershell
winget install Git.Git
```

Verify:
```powershell
git --version
```

---

## Verification

### Complete Verification Script

Run this PowerShell script to verify everything is installed correctly:

```powershell
# Create a verification script
@"
Write-Host "=== JAVA VERIFICATION ===" -ForegroundColor Green
java -version 2>&1

Write-Host "`n=== PYTHON VERIFICATION ===" -ForegroundColor Green
python --version

Write-Host "`n=== PIP PACKAGES ===" -ForegroundColor Green
pip list | findstr "fastapi|uvicorn|mediapipe|opencv|numpy|torch"

Write-Host "`n=== MAVEN VERIFICATION ===" -ForegroundColor Green
.\mvnw --version

Write-Host "`n=== PROJECT STRUCTURE ===" -ForegroundColor Green
Get-ChildItem -Path . | Select-Object Name

Write-Host "`n=== PYTHON VENV ===" -ForegroundColor Green
if (Test-Path ".\.venv311") {
    Write-Host ".venv311 folder exists: YES" -ForegroundColor Green
} else {
    Write-Host ".venv311 folder exists: NO" -ForegroundColor Red
}

Write-Host "`n=== BUILD VERIFICATION ===" -ForegroundColor Green
if (Test-Path ".\target\classes") {
    Write-Host "Java build successful: YES" -ForegroundColor Green
} else {
    Write-Host "Java build successful: NO (run mvnw clean install)" -ForegroundColor Yellow
}

Write-Host "`n=== ALL CHECKS COMPLETE ===" -ForegroundColor Cyan
"@ | Out-File -FilePath verify_setup.ps1 -Encoding UTF8

# Run the verification script
.\verify_setup.ps1
```

---

## Running the Project

### Start Python MediaPipe Service

```powershell
# Activate virtual environment
.\.venv311\Scripts\Activate.ps1

# Navigate to mediapipe_service
cd mediapipe_service

# Run FastAPI server
uvicorn extractor:app --host 127.0.0.1 --port 8000

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete
```

**Test the service** (in another PowerShell window):
```powershell
# Test endpoint
curl -X GET "http://127.0.0.1:8000/docs"
# Opens interactive API documentation
```

### Start Java Spring Application

```powershell
# In project root (deactivate Python venv if still active)
deactivate

# Run Spring Boot application
.\mvnw spring-boot:run

# You should see:
# ... c.s.s.SignLanguageRecognizerApplication : Started SignLanguageRecognizerApplication in X.XXX seconds
```

Access the application at: `http://localhost:8080`

---

## Quick Reference Commands

### First-Time Setup (Copy-Paste All)
```powershell
# Install Java
winget install Oracle.JDK.21

# Install Python
winget install Python.Python.3.11

# Create venv
cd C:\Users\lavib\Downloads\sign-language-recognizer
python -m venv .venv311

# Activate venv
.\.venv311\Scripts\Activate.ps1

# Install Python packages
pip install --upgrade pip
pip install fastapi uvicorn[standard] mediapipe opencv-python numpy torch torchvision python-multipart

# Verify Maven
.\mvnw --version

# Build Java project
.\mvnw clean install -DskipTests
```

### Daily Development
```powershell
# Terminal 1: Start Python service
cd C:\Users\lavib\Downloads\sign-language-recognizer
.\.venv311\Scripts\Activate.ps1
cd mediapipe_service
uvicorn extractor:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Java service
cd C:\Users\lavib\Downloads\sign-language-recognizer
deactivate
.\mvnw spring-boot:run
```

---

## Troubleshooting

### Issue: "Python not found" or "python is not recognized"
**Solution:**
```powershell
# Restart PowerShell after installation
# OR add Python to PATH manually:
# 1. Search "Environment Variables" in Windows
# 2. Click "Edit the system environment variables"
# 3. Click "Environment Variables" button
# 4. Under "User variables", click "New"
# 5. Variable name: PATH, Variable value: C:\Users\{YourUsername}\AppData\Local\Programs\Python\Python311
# 6. Click OK and restart PowerShell
```

### Issue: "Java not found" or "java is not recognized"
**Solution:**
```powershell
# Restart PowerShell after installation
# Verify installation path:
dir "C:\Program Files\Java"
```

### Issue: Execution Policy Error for `.venv311\Scripts\Activate.ps1`
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Type 'Y' and press Enter
# Then try activating venv again
.\.venv311\Scripts\Activate.ps1
```

### Issue: Maven Build Fails ("BUILD FAILURE")
**Solution:**
```powershell
# Try skipping tests first
.\mvnw clean install -DskipTests

# If still fails, check Java version
java -version
# Should be 21 or higher

# Clear Maven cache and rebuild
.\mvnw clean
.\mvnw install -DskipTests
```

### Issue: Port 8000 or 8080 Already in Use
**Solution:**
```powershell
# Find process using port 8000 (Python)
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID {PID} /F

# For Java port 8080
netstat -ano | findstr :8080
taskkill /PID {PID} /F
```

### Issue: "pip install" is very slow
**Solution:**
```powershell
# Use a faster PyPI mirror
pip install -i https://mirrors.aliyun.com/pypi/simple/ fastapi uvicorn mediapipe opencv-python numpy torch torchvision python-multipart

# Or upgrade pip first
python -m pip install --upgrade pip
```

### Issue: CUDA/GPU not working with PyTorch
**Solution:**
```powershell
# For CPU-only (recommended for testing):
# Already installed correctly

# For NVIDIA GPU support:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## System Specifications Verified On

- **OS:** Windows 10/11 (64-bit)
- **RAM:** 8GB minimum (16GB recommended)
- **Disk Space:** 20GB available
- **GPU:** Optional (NVIDIA CUDA 11.8+ for GPU support)

---

## Next Steps After Installation

1. ✅ Complete setup using this guide
2. Collect training data (record videos of sign language)
3. Process videos through MediaPipe extractor
4. Train the model using `train.py`
5. Test predictions with `quick_predict.py`
6. Export model to ONNX for Java integration

See `plan.md` for detailed development roadmap.

---

## Support

If you encounter issues:
1. Check troubleshooting section above
2. Verify all prerequisites are installed: `verify_setup.ps1`
3. Check application logs in PowerShell console
4. Review `plan.md` for detailed step-by-step guide

