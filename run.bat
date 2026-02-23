@echo off
echo ========================================
echo Breast Cancer Prediction System
echo ========================================
echo.

:menu
echo Select an option:
echo 1. Train Model
echo 2. Run Streamlit Web App
echo 3. Run FastAPI Server
echo 4. Compare Models
echo 5. Generate Visualizations
echo 6. Run Tests
echo 7. Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo Training model...
    python main.py
    pause
    goto menu
)

if "%choice%"=="2" (
    echo Starting Streamlit app...
    streamlit run app_streamlit.py
    goto menu
)

if "%choice%"=="3" (
    echo Starting FastAPI server...
    uvicorn api.app:app --reload
    goto menu
)

if "%choice%"=="4" (
    echo Comparing models...
    python compare_models.py
    pause
    goto menu
)

if "%choice%"=="5" (
    echo Generating visualizations...
    python visualize_results.py
    pause
    goto menu
)

if "%choice%"=="6" (
    echo Running tests...
    python -m pytest tests/
    pause
    goto menu
)

if "%choice%"=="7" (
    echo Exiting...
    exit
)

echo Invalid choice. Please try again.
pause
goto menu
