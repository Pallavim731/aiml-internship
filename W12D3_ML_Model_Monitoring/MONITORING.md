# W12D3: Monitoring ML Models in Production

## 1. Monitoring Strategy

ML models running in production should be monitored continuously to
ensure that the API and model are working correctly.

## 2. What to Track

### API Performance

- Number of API requests
- Response time / latency
- HTTP error rate
- CPU usage
- Memory usage
- Container health

### Model Performance

- Prediction accuracy
- Precision
- Recall
- Model confidence
- Number of successful and failed predictions

### Data Quality

- Missing values
- Invalid input values
- Unexpected data types
- Changes in input data distribution

### Data Drift

Production input data should be compared with the training data.
Significant changes in the input distribution can indicate data drift.

## 3. Alerts

The following conditions should generate alerts:

- API error rate greater than 5%
- Response latency greater than 2 seconds
- Container becomes unhealthy
- CPU usage remains above 80%
- Memory usage remains above 80%
- Large increase in invalid requests
- Significant data drift
- Model accuracy falls below the required threshold

## 4. Retraining Triggers

The model should be considered for retraining when:

1. Model accuracy falls below the required threshold.
2. Significant data drift is detected.
3. New production data is substantially different from training data.
4. Business requirements change.
5. A newly trained model performs better during evaluation.

The new model should be evaluated before replacing the production model.

## 5. Monitoring Tools

The project uses:

- Docker - containerization
- GitHub Actions - CI/CD automation
- MLflow - experiment and model tracking
- FastAPI / Python - ML API
- Ragas - RAG evaluation
- Git - version control

## 6. Production Monitoring Flow

```text
User Request
     |
     v
ML API Container
     |
     +----> API Metrics
     |
     +----> Model Predictions
     |
     +----> Data Quality
     |
     +----> Data Drift
     |
     v
Monitoring and Alerts
     |
     v
Retrain if Required
     |
     v
Evaluate New Model
     |
     v
Deploy Updated Model