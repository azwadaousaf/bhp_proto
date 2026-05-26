# BHP Dynamic Ore Recovery Optimisation Prototype

This is a functional AI prototype for ISYS3443 Assessment 3.

## What it demonstrates
- User input: current mine shift conditions
- AI processing: Random Forest machine learning model predicts ore recovery
- Output: predicted recovery, optimised scenario, risk score, confidence score
- Business action: approve, escalate, or hold current plan

## How to run locally
1. Install Python from https://www.python.org/downloads/
2. Open Command Prompt or PowerShell in this folder
3. Run:
   pip install -r requirements.txt
4. Run:
   streamlit run app.py

## How to use in presentation
Move the sliders to create a current mine shift scenario. Then explain:
Input -> AI model -> digital twin scenario -> recommendation -> business action.
