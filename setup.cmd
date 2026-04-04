@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================
echo   THGNN ENVIRONMENT SETUP
echo   GPU: RTX 4060 Ti (Ada, sm_89)
echo ================================

REM ── [1/7] Detect Python ──────────────────────────────────────────────────
echo [1/7] Detect Python...
set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 set "PY_CMD=python"
)
if "%PY_CMD%"=="" (
    echo ERROR: Python 3.10+ not found.
    exit /b 1
)
echo Using: %PY_CMD%

REM ── [2/7] Virtual environment ────────────────────────────────────────────
echo [2/7] Create virtual environment...
if not exist ".venv" (
    %PY_CMD% -m venv .venv
    if errorlevel 1 ( echo ERROR: venv failed & exit /b 1 )
) else (
    echo .venv exists, reusing.
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 ( echo ERROR: activate failed & exit /b 1 )

REM ── [3/7] pip tools ──────────────────────────────────────────────────────
echo [3/7] Upgrade pip tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 ( echo ERROR: pip upgrade failed & exit /b 1 )

REM ── [4/7] Core dependencies ──────────────────────────────────────────────
echo [4/7] Core dependencies...

REM NumPy 1.x — DGL/PyG không tương thích với NumPy 2.x
python -m pip install "numpy==1.26.4"
if errorlevel 1 ( echo ERROR: numpy & exit /b 1 )

python -m pip install pydantic scipy
if errorlevel 1 ( echo ERROR: pydantic/scipy & exit /b 1 )

REM ── [5/7] PyTorch CUDA 12.4 ──────────────────────────────────────────────
echo [5/7] PyTorch 2.4.1 + CUDA 12.4...
REM cu124 tối ưu hơn cu121 cho Ada Lovelace (RTX 40xx)
REM torch 2.4.x bật Flash Attention 2 tự động trên sm_89
python -m pip install --upgrade ^
    torch==2.4.1 ^
    torchvision==0.19.1 ^
    torchaudio==2.4.1 ^
    --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 ( echo ERROR: PyTorch & exit /b 1 )

REM ── [6/7] GNN libraries ──────────────────────────────────────────────────
echo [6/7] GNN libraries...

REM DGL — CUDA 12.x wheel
python -m pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/cu121/repo.html
if errorlevel 1 ( echo ERROR: DGL & exit /b 1 )

REM PyG CUDA extensions — phải match torch+cuda version
python -m pip install torch_scatter torch_sparse torch_cluster ^
    torch_spline_conv pyg_lib ^
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
if errorlevel 1 (
    echo WARNING: PyG CUDA extensions failed, trying CPU fallback...
    python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv
)

python -m pip install torch_geometric
if errorlevel 1 ( echo ERROR: torch_geometric & exit /b 1 )

REM ── [6.1/7] ML/NLP libraries ─────────────────────────────────────────────
echo [6.1/7] ML/NLP libraries...
python -m pip install ^
    scikit-learn==1.4.2 ^
    transformers==4.41.2 ^
    tokenizers ^
    huggingface_hub ^
    librosa ^
    soundfile ^
    pandas ^
    tqdm
if errorlevel 1 ( echo ERROR: ML packages & exit /b 1 )

REM ── [6.2/7] requirements.txt ─────────────────────────────────────────────
echo [6.2/7] requirements.txt...
if exist requirements.txt (
    python -m pip install -r requirements.txt
    if errorlevel 1 ( echo ERROR: requirements.txt & exit /b 1 )
)

echo.
echo ================================
echo   SETUP SUCCESS
echo ================================
echo.
echo Activate venv:
echo   .venv\Scripts\activate
echo.
echo Recommended GPU flags for training (add to train.py):
echo   torch.backends.cudnn.benchmark        = True
echo   torch.backends.cuda.matmul.allow_tf32 = True
echo   torch.backends.cudnn.allow_tf32       = True
echo.
echo Recommended batch sizes for RTX 4060 Ti (16GB):
echo   WavLM extract  : --batch_size 64
echo   HMSGNet train  : batch_size=16 (graph data)
echo.
exit /b 0