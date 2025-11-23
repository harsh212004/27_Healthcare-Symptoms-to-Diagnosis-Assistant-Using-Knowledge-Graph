Healthcare Symptom-to-Disease Prediction System

A hybrid AI model that predicts possible diseases based on user-provided symptoms using a combination of a Knowledge Graph and a LightGBM Machine Learning classifier.

Features

Builds a Knowledge Graph (KG) linking symptoms and diseases with weighted edges.

Computes symptom importance based on probability, specificity, and variance.

Generates an expanded training dataset with symptom combinations, bigrams/trigrams, and numeric features.

Trains a LightGBM classifier using TF-IDF + engineered features.

Implements a hybrid prediction engine that fuses KG reasoning with ML probabilities.

Produces a condensed diagnostic report with:

Top predicted diseases

Confidence scores

Matched symptoms

KG–ML contribution split

Dataset

The project uses a curated disease-symptom dataset compiled from multiple reliable health sources such as WHO and public medical repositories.
Each symptom includes a probability score indicating its association with the disease.

File used:
final_disease_symptoms_trimmed.json

How It Works

Load and parse disease–symptom data

Build a weighted Knowledge Graph

Generate symptom importance scores

Create a synthetic training dataset

Extract TF-IDF text features + numeric statistical features

Train LightGBM

Run hybrid KG+ML prediction

Output user-friendly diagnostic summaries

Installation
pip install numpy pandas networkx matplotlib scikit-learn scipy lightgbm

Run the Code

Place the JSON dataset in the project directory and run:

python main.py


You can also call the prediction function:

hybrid_predict(['fever','cough','fatigue'])


Or get a formatted diagnostic report:

diagnose_patient_condensed(['fever','nausea','headache'])

Outputs

Knowledge Graph visualization (lightgbm_knowledge_graph.png)

Console-friendly diagnostic reports

Performance metrics (Top-1, Top-3 accuracy, F1 score)

Technologies Used

NetworkX – Knowledge Graph

LightGBM – ML classifier

scikit-learn – TF-IDF + evaluation

Pandas / NumPy – data handling

Matplotlib – visualization

Future Improvements

Add patient metadata (age, duration, severity)

Better symptom normalization & synonym mapping

Deploy as web or mobile API

Add treatment recommendations
