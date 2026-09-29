\# W12D2 — Containerising ML Apps with Docker



\## Overview



This project demonstrates how to containerise a machine learning API using Docker and automate its CI/CD process with GitHub Actions.



The application uses a Random Forest classifier trained on the Iris dataset and exposes the model through a FastAPI REST API.



\## Technologies Used



\- Python 3.12

\- FastAPI

\- Scikit-learn

\- NumPy

\- Pytest

\- Ruff

\- Docker

\- GitHub Actions

\- GitHub Container Registry

\- MLOps



\## Project Structure



```text

W12D2\_Containerising\_ML\_Apps/

│

├── app/

│   └── api.py

│

├── tests/

│   └── test\_api.py

│

├── .github/

│   └── workflows/

│       └── ci.yml

│

├── Dockerfile

├── .dockerignore

├── pytest.ini

├── requirements.txt

└── README.md

