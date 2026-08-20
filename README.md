# AI Medical Diagnosis Assistant

An interactive machine learning application that predicts possible diseases from user-provided symptoms. The application uses a trained **Bernoulli Naive Bayes** model and provides disease predictions, probability scores, symptom severity analysis, and specialist recommendations through a Streamlit interface.

## Features

* Select symptoms from the trained model's symptom list
* Enter symptoms using comma-separated text
* Binary symptom representation for machine learning prediction
* Top 5 possible disease predictions
* Probability scores for each prediction
* Symptom severity scoring
* Suggested medical specialist based on the predicted disease
* Patient information display
* Interactive Streamlit interface
* New diagnosis functionality
* Educational medical disclaimer

## Technology Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* Joblib
* Bernoulli Naive Bayes

## How It Works

The application follows a simple machine learning prediction workflow:

```text
User
 |
 v
Enter Patient Information
 |
 v
Select or Enter Symptoms
 |
 v
Convert Symptoms to Binary Features
 |
 v
Trained Bernoulli Naive Bayes Model
 |
 v
Generate Disease Probabilities
 |
 v
Sort Predictions
 |
 +---- Top 5 Possible Diseases
 |
 +---- Probability Scores
 |
 +---- Severity Score
 |
 +---- Suggested Specialist
 |
 v
Display Results
```

## 1. Patient Information

The application allows the user to optionally enter basic patient information:

* Name
* Age
* Gender

This information is displayed in the diagnosis report and does not affect the machine learning prediction.

## 2. Symptom Selection

Users can select symptoms directly from the symptom list used by the trained model.

The application also provides a text field where symptoms can be entered in comma-separated format.

Example:

```text
fever, headache, cough
```

The application checks the entered symptoms against the model's known symptom list before using them for prediction.

## 3. Feature Preparation

The machine learning model expects symptoms in binary form.

```text
1 = Symptom present
0 = Symptom absent
```

For example:

```text
fever       = 1
headache    = 1
cough       = 1
fatigue     = 0
vomiting    = 0
```

The application creates a Pandas DataFrame containing the complete symptom feature set and maintains the same symptom ordering expected by the trained model.

## 4. Disease Prediction

The trained Bernoulli Naive Bayes model generates probabilities for the possible disease classes using:

```python
model.predict_proba(input_data)
```

The predictions are then sorted from highest to lowest probability.

The application displays the five highest-probability predictions.

Example:

```text
1. Disease A — 72.50%
2. Disease B — 14.20%
3. Disease C — 7.80%
4. Disease D — 3.40%
5. Disease E — 2.10%
```

## 5. Symptom Severity Scoring

The application can use `Symptom-severity.csv` to calculate an overall symptom severity score.

Each selected symptom can have an associated severity weight.

The application adds the weights of the selected symptoms to produce a severity score.

The result is classified as:

| Severity Score | Level            |
| -------------- | ---------------- |
| 0              | Data unavailable |
| 1–5            | Low              |
| 6–10           | Moderate         |
| Above 10       | High             |

## 6. Specialist Recommendation

The application provides a suggested medical specialist based on keywords found in the predicted disease name.

Examples include:

```text
Heart-related disease      → Cardiologist
Skin-related disease       → Dermatologist
Respiratory disease        → Pulmonologist
Stomach-related disease    → Gastroenterologist
Kidney-related disease     → Nephrologist
Brain-related disease      → Neurologist
Eye-related disease        → Ophthalmologist
```

If no matching specialist is found, the application recommends:

```text
General Physician
```

## Model

The application uses a trained **Bernoulli Naive Bayes** classifier.

Bernoulli Naive Bayes is suitable for binary feature representations, making it appropriate for this symptom-based prediction system where each symptom is represented as either present or absent.

The model receives a binary symptom vector and produces probability estimates for the available disease classes.

## Model Files

The application loads the trained model and symptom list using Joblib:

```text
disease_model_v2.pkl
symptoms_v2.pkl
```

The model file contains the trained machine learning classifier.

The symptom file contains the feature names and their expected ordering.

## Project Structure

```text
AI-Medical-Diagnosis-Assistant/
│
├── app.py
├── disease_model_v2.pkl
├── symptoms_v2.pkl
├── Symptom-severity.csv
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Medical-Diagnosis-Assistant
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Application Workflow

```text
1. Enter optional patient information
2. Select symptoms or enter them as comma-separated values
3. Click Predict Disease
4. Convert symptoms into binary features
5. Generate disease probabilities
6. Display the top 5 predictions
7. Calculate symptom severity
8. Display a suggested specialist
```

## Project Highlights

* Developed an end-to-end machine learning application for symptom-based disease prediction.
* Implemented a Bernoulli Naive Bayes classification model.
* Converted user symptoms into binary machine learning features.
* Implemented probability-based Top-5 disease prediction.
* Added symptom severity scoring using symptom weights.
* Implemented specialist recommendation mapping.
* Built an interactive Streamlit interface for model inference.
* Integrated trained machine learning artifacts using Joblib.

## Disclaimer

This project is intended for **educational and demonstration purposes only**.

The predictions generated by this application should not be considered a medical diagnosis or a substitute for professional medical advice, diagnosis, or treatment. Users should consult a qualified healthcare professional for medical concerns.

## Author

**Narravula Samuel Reddy**

B.Tech in Computer Science and Engineering
Rajiv Gandhi University of Knowledge Technologies (RGUKT), RK Valley
