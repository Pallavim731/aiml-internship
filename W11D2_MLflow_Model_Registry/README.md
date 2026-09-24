\# W11D2 - MLflow Model Registry \& Versioning



\## Objective



Track multiple machine learning experiments using MLflow, compare their results, register the best model, and serve it using the MLflow Model Registry.



\## Model Used



\- Dataset: Iris dataset

\- Algorithm: Random Forest Classifier

\- Framework: Scikit-learn

\- Experiment Tracking: MLflow



\## Experiments



Five Random Forest experiments were run with different hyperparameters.



The experiments tracked:



\- Number of trees (`n\_estimators`)

\- Maximum tree depth (`max\_depth`)

\- Accuracy

\- Model artifacts



\## Model Selection



The five MLflow runs were compared using accuracy.



The run with the highest accuracy was selected as the best model.



\## Model Registry



The best model was registered as:



`W11D2\_Iris\_Random\_Forest`



The registered model was then loaded using its model version.



\## Model Serving



The registered model was served using:



```bash

mlflow models serve -m "models:/W11D2\_Iris\_Random\_Forest/1" -p 5002 --host 127.0.0.1 --env-manager local

