HEART DISEASE PREDICTOR - STREAMLIT FRONTEND

1. Put these files in the same folder:
   app.py
   requirements.txt
   KNN_hear (1).pkl
   scaler (1).pkl
   columns (1).pkl

2. Open a terminal in this folder and install dependencies:
   python -m pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

4. Streamlit will show a local URL (usually http://localhost:8501). Open it in your browser.

IMPORTANT:
- This app assumes the saved model predicts 0 = negative class and 1 = positive class.
- The saved feature list contains 14 features and does not include Age, so Age is collected for display/context but is not passed to the model.
- This is an educational demo, not a medical diagnosis. Confirm the class labels and feature preprocessing against the notebook used to train the model.
