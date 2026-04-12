# Credit Risk Prediction - MLOps Production Project

This project is an end-to-end machine learning system that predicts whether a customer is likely to default on their credit card payments. It includes data ingestion, validation, transformation, model training, evaluation, deployment, and a web interface for real-time predictions.

---

## Project Overview

This project demonstrates a full MLOps pipeline, including:

- Data ingestion from MongoDB  
- Data validation and drift detection using Evidently  
- Data transformation and feature engineering  
- Model training using XGBoost  
- Model evaluation and selection  
- Model deployment using AWS (ECR, EC2, S3)  
- CI/CD pipeline using GitHub Actions  
- Web interface using FastAPI for real-time predictions  

---

## Technology Stack

- Python  
- FastAPI  
- Scikit-learn  
- XGBoost  
- MongoDB  
- Docker  
- AWS (ECR, EC2, S3)  
- GitHub Actions  
- Evidently AI  

---

## Project Structure

```
credit-risk-mlops/
│
├── .github/                  # CI/CD workflows (GitHub Actions)
├── artifact/                 # Generated pipeline artifacts
├── config/                   # YAML configuration files
├── logs/                     # Log files
├── notebook/                 # Jupyter notebooks (EDA and experimentation)
│
├── credit_risk/              # Core application package
│   ├── cloud_storage/        # AWS S3 interaction
│   ├── components/           # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/        # Configuration management
│   ├── constants/            # Project constants
│   ├── data_access/          # MongoDB access logic
│   ├── entity/               # Config and artifact entities
│   ├── exception/            # Custom exception handling
│   ├── logger/               # Logging module
│   ├── pipeline/             # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── utils/                # Utility functions
│   └── __init__.py
│
├── static/                   # CSS and static files
├── templates/                # HTML templates (FastAPI frontend)
│   └── creditrisk.html
│
├── app.py                    # FastAPI application entry point
├── demo.py                   # Pipeline execution script
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── template.py               # Project structure generator
├── README.md                 # Project documentation
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/josierra21/credit-risk-mlops.git
cd credit-risk-mlops
```

---

### 2. Create a virtual environment

```bash
conda create -n credit python=3.8 -y
conda activate credit
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Set environment variables

#### Windows (PowerShell)

```powershell
setx MONGODB_URL "your_mongodb_connection_string"
setx AWS_ACCESS_KEY_ID "your_access_key"
setx AWS_SECRET_ACCESS_KEY "your_secret_key"
```

#### macOS/Linux

```bash
export MONGODB_URL="your_mongodb_connection_string"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

---

## Running the Project

### Run the training pipeline

```bash
python demo.py
```

---

### Run the FastAPI application

```bash
python app.py
```

Open a browser and navigate to:

```
http://localhost:8080
```

---

## Model Usage

The web interface allows users to input:

- Customer demographic information  
- Credit limit and financial details  
- Payment history for the previous six months  
- Bill statements and payment amounts  

The model returns one of the following predictions:

- Default risk detected  
- No default risk detected  

---

## CI/CD Deployment with AWS

### 1. Create an IAM User

Assign the following permissions:

- AmazonEC2FullAccess  
- AmazonEC2ContainerRegistryFullAccess  
- AmazonS3FullAccess  

---

### 2. Create an Amazon ECR Repository

Create an ECR repository to store the Docker image for this project.

Repository URI:

```
911229171314.dkr.ecr.us-east-1.amazonaws.com/creditrepo
```

---

### 3. Launch an EC2 Instance (Ubuntu)

Create an EC2 instance that will be used to host the application and act as the self-hosted GitHub Actions runner.

Recommended configuration:
- Operating System: Ubuntu (latest LTS)
- Instance Type: t2.micro or t3.micro
- Storage: Default (8 GB is sufficient)

Ensure the following:
- Security group allows inbound traffic on port 8080 (application access)
- SSH access (port 22) is enabled

---

### 4. Install Docker on the EC2 Instance

Run the following commands on the EC2 instance:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker ubuntu
newgrp docker
```

---

### 5. Configure EC2 as a Self-Hosted Runner

In GitHub:

Settings → Actions → Runners → New self-hosted runner

Select Linux and run the provided commands on the EC2 instance.

This allows GitHub Actions to deploy directly to your EC2 instance.

---

### 6. Configure GitHub Secrets

Go to:

Settings → Secrets → Actions

Add the following:

- AWS_ACCESS_KEY_ID  
- AWS_SECRET_ACCESS_KEY  
- AWS_DEFAULT_REGION  
- ECR_REPO  

---

## Deployment Workflow

1. Build Docker image  
2. Push image to Amazon ECR  
3. Pull image on EC2  
4. Run container on EC2  
5. Serve the FastAPI application  

---

## Features

- Credit limit  
- Age 
- Sex 
- Education level  
- Marital status  
- Payment history (six months)  
- Bill statements (six months)  
- Previous payments (six months)  

---

## Key Highlights

- End-to-end machine learning pipeline  
- Real-time prediction interface  
- Data drift detection  
- Cloud-based deployment  
- Automated CI/CD pipeline  

---

## Git Commands

```bash
git add .
git commit -m "Update"
git push origin main
```

---

## Useful Links

- Anaconda: https://www.anaconda.com/  
- Visual Studio Code: https://code.visualstudio.com/  
- Git: https://git-scm.com/  
- MongoDB: https://account.mongodb.com/  
- Evidently AI: https://www.evidentlyai.com/ 
- Data link: https://www.kaggle.com/datasets/ifeanyichukwunwobodo/credit-card-default

---

## Author

Joanna Sierra-Mendoza 