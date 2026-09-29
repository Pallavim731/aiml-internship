\# W12D1 — MLOps Overview: CI/CD for ML Systems



\## Project Overview



This project demonstrates a basic MLOps workflow for a machine learning API.



The ML API uses FastAPI and a Random Forest classifier trained on the Iris dataset.



The application is:



\- Containerized using Docker

\- Tested using pytest

\- Linted using Ruff

\- Built automatically using GitHub Actions

\- Prepared for CI/CD-based deployment



\## API Endpoints



\### Health Check



```text

GET /health

