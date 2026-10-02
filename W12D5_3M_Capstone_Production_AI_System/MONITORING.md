\# W12D5 Production AI System - Monitoring Strategy



\## 1. What to Monitor



The production AI system should be monitored continuously to make sure the API is available, fast, reliable, and producing useful predictions.



\### API Health

\- API availability

\- Health endpoint status

\- Number of failed requests

\- HTTP error rates



\### Performance

\- Response time

\- Average latency

\- Request throughput

\- CPU and memory usage

\- Docker container status



\### Model Performance

\- Prediction distribution

\- Input data distribution

\- Changes in prediction patterns

\- Model accuracy when labelled data becomes available



\### Data Quality

\- Missing input values

\- Invalid input values

\- Unexpected data ranges

\- Changes in input data distribution



\## 2. Alerts



Alerts should be configured for important production problems.



Examples:



\- API health check fails repeatedly.

\- HTTP 5xx error rate becomes high.

\- Response latency stays above the expected threshold.

\- CPU or memory usage remains high.

\- Docker container stops unexpectedly.

\- Large changes are detected in input data distribution.

\- Model prediction patterns change significantly.



\## 3. Retraining Triggers



The model should be considered for retraining when:



\- Model accuracy decreases below the agreed threshold.

\- Significant data drift is detected.

\- The production data distribution changes substantially.

\- Prediction quality decreases consistently.

\- New labelled training data becomes available.



Retraining should include evaluation of the new model before deployment.



\## 4. Monitoring Workflow



The production monitoring process is:



1\. Collect API, system, data, and model metrics.

2\. Compare metrics with predefined thresholds.

3\. Generate alerts when thresholds are exceeded.

4\. Investigate the cause of the problem.

5\. Retrain the model when model or data performance requires it.

6\. Evaluate the new model.

7\. Deploy the new model only after validation.



\## 5. MLOps Tools



The pipeline can use:



\- Docker for containerization.

\- GitHub Actions for CI/CD.

\- MLflow for experiment tracking and model management.

\- Ragas for evaluating RAG-based AI systems.

\- Python and FastAPI for the ML API.

