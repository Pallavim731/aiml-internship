\# W12D4: End-to-End MLOps Pipeline Review



\## Monitoring Strategy



The production ML API should be monitored continuously to ensure that

the API, model, and data are working correctly.



\## Metrics to Track



\### API Metrics



\- Number of requests

\- Response latency

\- HTTP error rate

\- CPU usage

\- Memory usage

\- Container health



\### Model Metrics



\- Prediction accuracy

\- Precision

\- Recall

\- Model confidence

\- Successful and failed predictions



\### Data Quality



\- Missing values

\- Invalid input values

\- Unexpected data types

\- Input distribution changes



\### Data Drift



Production input data should be compared with the training data.

Significant changes may indicate data drift.



\## Alerts



Alerts should be generated when:



\- API error rate is above 5%

\- Response latency is above 2 seconds

\- Container becomes unhealthy

\- CPU usage remains above 80%

\- Memory usage remains above 80%

\- Invalid requests increase significantly

\- Significant data drift is detected

\- Model accuracy falls below the required threshold



\## Retraining Triggers



The model should be considered for retraining when:



1\. Model accuracy drops below the required threshold.

2\. Significant data drift is detected.

3\. Production data changes substantially.

4\. Business requirements change.

5\. A newly trained model performs better during evaluation.



The new model must be evaluated before replacing the production model.



\## MLOps Pipeline



```text

Code

&#x20;|

&#x20;v

GitHub

&#x20;|

&#x20;v

GitHub Actions

&#x20;|

&#x20;+--> Lint

&#x20;|

&#x20;+--> Test

&#x20;|

&#x20;+--> Docker Build

&#x20;|

&#x20;+--> Push Image

&#x20;|

&#x20;v

Production

&#x20;|

&#x20;v

Monitoring

&#x20;|

&#x20;+--> Alerts

&#x20;|

&#x20;+--> Drift Detection

&#x20;|

&#x20;v

Retraining

&#x20;|

&#x20;v

Model Evaluation

&#x20;|

&#x20;v

Deployment

