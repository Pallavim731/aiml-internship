name: W12D1 ML API CI/CD

on:
  push:
    branches:
      - main
      - "feat/**"
  pull_request:
    branches:
      - main

permissions:
  contents: read
  packages: write

jobs:
  ci:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r W12D1_MLOps_CI_CD/requirements.txt
          pip install ruff

      - name: Lint
        working-directory: W12D1_MLOps_CI_CD
        run: |
          ruff check app tests

      - name: Run tests
        working-directory: W12D1_MLOps_CI_CD
        run: |
          pytest -q

      - name: Set Docker image name
        run: |
          echo "IMAGE_NAME=ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/w12d1-ml-api" >> "$GITHUB_ENV"

      - name: Build Docker image
        working-directory: W12D1_MLOps_CI_CD
        run: |
          docker build -t $IMAGE_NAME:${{ github.sha }} -t $IMAGE_NAME:latest .

      - name: Log in to GitHub Container Registry
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Push Docker image
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: |
          docker push $IMAGE_NAME:${{ github.sha }}
          docker push $IMAGE_NAME:latest