# AI Medical Diagnosis Assistant

An interactive machine learning application that predicts possible diseases from user-provided symptoms. The application combines a trained **Bernoulli Naive Bayes** model with natural-language symptom extraction and a multi-stage diagnostic workflow to progressively refine predictions.

The application supports **754 disease classes, 377 symptoms, and approximately 246K medical records**.

## Features

* Symptom selection from the trained model's symptom list
* Free-text symptom description
* Natural-language symptom extraction
* Synonym mapping for common medical and everyday expressions
* Three-stage diagnostic workflow
* Preliminary disease predictions
* Intelligent follow-up symptom questions
* Top 5 possible disease predictions
* Prediction confidence scores
* Uncertainty detection for closely matched predictions
* Optional disease descriptions
* Interactive Streamlit interface
* Patient information display
* Restart and new-diagnosis functionality

## Technology Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* Joblib
* Regular Expressions
* Bernoulli Naive Bayes
* HTML/CSS

## How It Works

The application follows a three-stage diagnostic workflow:

```text
User Symptoms
      |
      v
Stage 1: Symptom Entry
      |
      +---- Select symptoms
      |
      +---- Describe symptoms in natural language
      |
      v
NLP Symptom Extraction
      |
      +---- Direct symptom matching
      |
      +---- Synonym mapping
      |
      v
Initial Disease Prediction
      |
      v
Stage 2: Follow-Up Questions
      |
      +---- Identify discriminative symptoms
      |
      +---- Ask targeted Yes / No / Not Sure questions
      |
      v
Stage 3: Final Diagnosis
      |
      +---- Top 5 predictions
      |
      +---- Confidence scores
      |
      +---- Uncertainty analysis
      |
      v
Final Prediction
```

## Stage 1: Symptom Entry

Users can provide symptoms in two ways:

1. Select symptoms from the available symptom list.
2. Describe their symptoms using free-form text.

The application contains 377 symptoms from the trained model.

For example:

```text
I have fever, headache and a runny nose for 3 days
```

The application extracts recognized symptoms from the text and converts them into the canonical symptom names used by the trained model.

## Natural Language Symptom Extraction

The application uses regular-expression-based matching to identify symptoms from user text.

It performs:

* Direct phrase matching
* Longest-first matching to reduce partial matches
* Case-insensitive matching
* Synonym mapping
* Canonical symptom conversion

For example:

```text
"runny nose"       -> nasal congestion
"body ache"        -> muscle pain
"breathlessness"   -> shortness of breath
"stomach ache"     -> abdominal pain
"throwing up"      -> vomiting
"head pain"        -> headache
```

The synonym system only maps terms when the corresponding canonical symptom exists in the model's actual symptom list.

## Stage 2: Intelligent Follow-Up Questions

After the initial symptoms are entered, the application generates an initial prediction and identifies the top five disease candidates.

It then selects additional symptoms that can help differentiate between those possible diseases.

The follow-up selection uses the model's feature log-probabilities to rank symptoms according to how discriminative they are among the leading disease predictions.

Users answer each follow-up question with:

* Yes
* No
* Not sure

Symptoms confirmed with "Yes" are added to the final symptom set.

## Stage 3: Final Diagnosis

The final stage runs the trained model using the complete symptom set.

The application provides:

* Most likely diagnosis
* Top 5 possible diagnoses
* Probability for each prediction
* Confidence level
* Patient summary
* Symptoms used for prediction
* Optional condition description

The prediction probabilities are obtained using the model's `predict_proba()` method and sorted from highest to lowest probability.

## Confidence and Uncertainty

The application provides three confidence levels:

| Confidence    | Interpretation      |
| ------------- | ------------------- |
| 60% or higher | High confidence     |
| 35%–59.99%    | Moderate confidence |
| Below 35%     | Low confidence      |

It also checks whether the top two predictions have probabilities within 10 percentage points of each other. If they are close, the application displays an uncertainty warning and recommends providing additional symptoms or consulting a healthcare professional.

## Model

The application uses a trained **Bernoulli Naive Bayes** model for disease prediction.

The model expects a binary symptom representation where:

```text
1 = Symptom present
0 = Symptom absent
```

The application builds the input vector using the exact symptom ordering expected by the trained model before generating predictions.

## Model Files

The application loads two trained artifacts:

```text
disease_model_v2.pkl
symptoms_v2.pkl
```

The model and symptom list are loaded using Joblib when the application starts.

## Optional Disease Information

The application can also load additional disease information from:

```text
disease_info.csv
```

The file is expected to contain at least:

```text
disease
description
```

If available, the description of the predicted condition can be displayed within the application.

## Project Structure

```text
AI-Medical-Diagnosis-Assistant/
│
├── app.py
├── train.py
├── disease_model_v2.pkl
├── symptoms_v2.pkl
├── disease_info.csv
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
2. Select or describe symptoms
3. Review automatically detected symptoms
4. Continue to follow-up questions
5. Answer targeted symptom questions
6. Generate the final diagnosis
7. Review the top 5 predictions
8. Examine confidence and uncertainty information
9. Start a new diagnosis when required
```

## Project Highlights

* Implemented an end-to-end symptom-based machine learning application.
* Integrated natural-language symptom extraction with a trained ML model.
* Built synonym mapping to handle common variations in symptom descriptions.
* Designed a multi-stage prediction workflow instead of relying on a single prediction step.
* Implemented discriminative follow-up symptom selection.
* Added probability-based confidence and uncertainty analysis.
* Built an interactive Streamlit user interface.

## Disclaimer

This project is intended for **educational and demonstration purposes only**.

The predictions generated by this application should not be considered a medical diagnosis or a substitute for professional medical advice, diagnosis, or treatment. Users should consult a qualified healthcare professional for medical concerns.

## Author

**Narravula Samuel Reddy**

B.Tech in Computer Science and Engineering
Rajiv Gandhi University of Knowledge Technologies (RGUKT), RK Valley
