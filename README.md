# end-to-end-machine-predictive-maintenance-classification

# Overview 
An end to end machine learning project on predicting machine failures using predictive maintenance techniques.

# Project Structure

├── artifacts/               # Model artifacts and saved data
├── config/                  # Configuration files
├── deployment.yaml          # Kubernetes deployment file
├── Dockerfile               # Docker setup for containerization
├── main.py                  # Main script to run the pipeline
├── params.yaml              # Model hyperparameters
├── projectstructure.py      # Project initialization script
├── requirements.txt         # Required Python packages
├── research/                # Notebooks and exploratory analysis
├── schema.yaml              # Data schema definition
├── service.yaml             # Kubernetes service configuration
├── setup.py                 # Python package setup
├── src/                     # Source code including model training & inference
├── templates/               # Streamlit UI templates
└── README.md                # Project documentation


# Technologies Used

Programming Language: Python

Machine Learning: Scikit-learn, Pandas, NumPy

Deployment: Docker, Kubernetes (Minikube)

Web Framework: Streamlit

Version Control: Git, GitHub

# Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/saitarun47/end-to-end-machine-predictive-maintenance-classification.git
cd end-to-end-machine-predictive-maintenance-classification

2️⃣ Create a Virtual Environment

conda create -n predictive-maintenance python=3.8 -y
conda activate predictive-maintenance

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Build and Run the Docker Container

docker build -t machinepredictive2 .
docker run -p 8501:8501 machinepredictive2

5️⃣ Deploy on Kubernetes (Minikube)

minikube start
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
minikube service streamlit-service --url

# Model Performance

The classification model was evaluated using:

Accuracy: 90%+

Precision & Recall: Balanced for minimizing false positives and false negatives

ROC-AUC Score: High predictive capability
