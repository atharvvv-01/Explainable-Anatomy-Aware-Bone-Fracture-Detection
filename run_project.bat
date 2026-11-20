@echo off
REM =====================================================================
REM Master Execution Script: Anatomy-Guided Bone Fracture Detection
REM YOLOv8 Localization + EfficientNetB3 Classifier + Grad-CAM Dashboard
REM =====================================================================

TITLE Anatomy-Guided Bone Fracture Detection System

:MENU
cls
echo =====================================================================
echo    ANATOMY-GUIDED BONE FRACTURE DETECTION SYSTEM (YOLO + EFFNET-B3)
echo =====================================================================
echo  1. Run Dataset Audit, Duplicate Detection and Integrity Report
echo  2. Generate Reproducible 70/15/15 Stratified Splits
echo  3. Train Stage 1 YOLO Anatomical Localization Detector
echo  4. Train Stage 2 EfficientNetB3 Classifier
echo  5. Run Full Experimental Benchmark (All 5 Experiments)
echo  6. Run System Unit Test Suite
echo  7. Launch Streamlit Medical Research Web Application
echo  8. Exit
echo =====================================================================
set /p choice="Enter your selection [1-8]: "

if "%choice%"=="1" goto AUDIT
if "%choice%"=="2" goto SPLITS
if "%choice%"=="3" goto DETECTOR
if "%choice%"=="4" goto CLASSIFIER
if "%choice%"=="5" goto BENCHMARK
if "%choice%"=="6" goto TESTS
if "%choice%"=="7" goto APP
if "%choice%"=="8" goto EXIT

echo Invalid choice. Please try again.
pause
goto MENU

:AUDIT
cls
echo Running Dataset Analysis and Integrity Report...
python -m src.dataset.dataset_analysis
pause
goto MENU

:SPLITS
cls
echo Generating Reproducible Data Splits...
python -m src.dataset.split_dataset
pause
goto MENU

:DETECTOR
cls
echo Training Stage 1 YOLO Anatomical Localization Detector...
python -m src.localization.train_detector
pause
goto MENU

:CLASSIFIER
cls
echo Training Stage 2 EfficientNetB3 Fracture Classifier...
python -m src.classification.train_classifier --experiment_name anatomy_guided_best --use_crops --loss_type class_weighted --epochs_p1 3 --epochs_p2 5
pause
goto MENU

:BENCHMARK
cls
echo Running Full 5-Experiment Benchmark Suite...
python scripts/run_all_experiments.py
pause
goto MENU

:TESTS
cls
echo Running Unit Test Suite...
python -m unittest discover tests
pause
goto MENU

:APP
cls
echo Launching Streamlit Web Application Dashboard...
echo Open your browser to the local URL displayed below.
streamlit run app.py
pause
goto MENU

:EXIT
echo Exiting. Goodbye!
exit /b 0
