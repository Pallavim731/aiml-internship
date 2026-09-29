\# W11D1 - MLflow Experiment Tracking



\## Objective



Track and compare multiple machine learning experiments using MLflow and register the best-performing model.



\## Technologies Used



\- Python

\- Scikit-learn

\- MLflow

\- Pandas

\- NumPy

\- Joblib



\## Dataset



The Iris dataset from Scikit-learn was used.



\## Machine Learning Model



Random Forest Classifier was used for classification.



\## Experiments



Five experiments were performed with different Random Forest hyperparameters.



| Experiment | n\_estimators | max\_depth | min\_samples\_split |

|------------|--------------|-----------|-------------------|

| 1 | 50 | 3 | 2 |

| 2 | 100 | 5 | 2 |

| 3 | 150 | 8 | 2 |

| 4 | 200 | 10 | 4 |

| 5 | 250 | None | 2 |



For every experiment, MLflow tracked:



\- Parameters

\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Trained model



\## Model Selection



The experiments were compared programmatically using MLflow.



The run with the highest accuracy was selected as the best run.



\## Model Registry



The selected model was registered in MLflow as:



`W11D1\_Iris\_Random\_Forest`



\## Model Serving



The registered model was served using the MLflow model serving API.



Example:



```bash

mlflow models serve -m "models:/W11D1\_Iris\_Random\_Forest/1" -p 5001 --host 127.0.0.1 --env-manager local

