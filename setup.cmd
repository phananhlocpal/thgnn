@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ================================
echo   THGNN ENVIRONMENT SETUP
echo ================================

echo [1/7] Detect Python...
set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo ERROR: Python 3.10+ not found.
    exit /b 1
)

echo Using Python: %PY_CMD%

echo [2/7] Create virtual environment...
if not exist ".venv" (
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv
        exit /b 1
    )
) else (
    echo .venv exists, reusing.
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate venv
    exit /b 1
)

echo [3/7] Upgrade pip tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: pip upgrade failed
    exit /b 1
)

echo [4/7] Fix core dependencies...

REM 🔴 QUAN TRỌNG: Fix NumPy (tránh crash DGL/PyG)
python -m pip install numpy==1.26.4
if errorlevel 1 (
    echo ERROR: Failed to install numpy
    exit /b 1
)

REM Fix missing dependency for DGL
python -m pip install pydantic
if errorlevel 1 (
    echo ERROR: Failed to install pydantic
    exit /b 1
)

REM Set DGL backend
set DGLBACKEND=pytorch

echo [5/7] Install PyTorch (CUDA 12.1)...

python -m pip install --upgrade torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 ^
--index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)

echo [6/7] Install GNN libraries (DGL + PyG)...

REM DGL CUDA
python -m pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/cu121/repo.html
if errorlevel 1 (
    echo ERROR: Failed to install DGL
    exit /b 1
)

REM torch_scatter
python -m pip install torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
if errorlevel 1 (
    echo ERROR: Failed torch_scatter
    exit /b 1
)

REM PyG dependencies (CỰC QUAN TRỌNG)
python -m pip install pyg_lib torch_sparse torch_cluster torch_spline_conv ^
-f https://data.pyg.org/whl/torch-2.3.0+cu121.html
if errorlevel 1 (
    echo ERROR: Failed PyG CUDA deps
    exit /b 1
)

REM PyTorch Geometric
python -m pip install torch_geometric
if errorlevel 1 (
    echo ERROR: Failed torch_geometric
    exit /b 1
)

echo [6.1/7] Install ML/NLP libraries...

python -m pip install torchdata==0.7.1
python -m pip install scikit-learn==1.4.2
python -m pip install transformers==4.41.2

if errorlevel 1 (
    echo ERROR: Failed ML packages
    exit /b 1
)

echo [6.2/7] Install project requirements...

if exist requirements.txt (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: requirements.txt failed
        exit /b 1
    )
)

echo.
echo ================================
echo   SETUP SUCCESS
echo ================================
echo Activate with:
echo    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
echo     .venv\Scripts\activate

exit /b 0