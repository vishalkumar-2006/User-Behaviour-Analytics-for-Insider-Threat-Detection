# Insider Threat Detection System using Machine Learning

## Overview

This project focuses on detecting **malicious insider activities** within an organization using machine learning techniques. Insider threats occur when employees misuse their authorized access to compromise security, steal sensitive information, or damage systems.

The system analyzes user behavior logs such as logon activity, email usage, device access, file operations, and web browsing patterns to identify suspicious activities.

The dataset used in this project is the **CERT Insider Threat Dataset r4.2**, which simulates realistic enterprise activity logs.

---

## Objectives

* Detect anomalous user behavior indicating insider threats
* Build machine learning models for classification of risky users
* Engineer behavioral features from raw activity logs
* Handle class imbalance using synthetic data generation
* Deploy a prediction interface using Streamlit

---

## Dataset

This project uses the **CERT Insider Threat Dataset (r4.2)**.

Due to its large size (~7.5GB), the dataset is hosted on **Kaggle**.

### Download Dataset

Download the dataset from the following link:
https://www.kaggle.com/datasets/vishalkumarsivakumar/insider-threat-detection

After downloading, extract the dataset and place it in the project directory like this:

```
Dataset/
    device.csv
    email.csv
    file.csv
    http.csv
    logon.csv
    insiders.csv
    psychometric.csv
```

---

## Project Structure

```
insider-threat-detection/
│
├── rf_insider_model.pkl
├── rf_features.pkl          
├── rf_scaler.pkl         
├── rf_threshold.pkl           
├── app.py                    # Streamlit web application
│
├── Notebook/
│    UserBehaviour.ipynb
│
├── sample_Datasets/
│   r4.1-3.csv                # Test datasets
│   r4.2-1-BDV0168.csv
│   r4.2-1-BIH0745.csv
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Models Considered

Several machine learning models were initially explored for insider threat detection, including:

* Random Forest Classifier
* XGBoost Classifier
* Support Vector Machine (SVM)
* Multi-Layer Perceptron (MLP) Neural Network

During experimentation, **Random Forest showed strong performance and stability on the behavioral features extracted from the CERT dataset**. Due to its robustness, interpretability, and good performance on imbalanced data, **Random Forest was selected as the primary model for the final implementation**.

While other models such as **XGBoost, SVM, and MLP were studied during the research phase**, they were **not included in the final implementation code** of this repository. The current system focuses on a Random Forest–based approach for predicting insider risk.

---

## Feature Engineering

Several behavioral features were extracted from user activity logs, including:

* Total logon count
* After-hours logon activity
* Weekend activity
* Unique PC usage
* Email communication patterns
* File access frequency
* Web browsing behavior
* USB device usage

These features help identify abnormal user behavior.

---

## Handling Class Imbalance

Insider threat events are rare compared to normal activities.
To address this issue, **synthetic minority samples** were generated using:

* SMOTE (Synthetic Minority Oversampling Technique)

This improves the model's ability to detect rare malicious behaviors.

---

## Installation

Clone the repository:

```
git clone https://github.com/vishalkumar-2006/User-Behaviour-Analytics-for-Insider-Threat-Detection.git
```

Install required dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Preprocess Data and Train Models
```
python UserBehaviour.ipynb
```

### Step 2: Launch Web Application

```
streamlit run app.py
```

---

## Web Application

The project includes a **Streamlit-based web application** that allows users to:

* Upload activity log CSV files
* Generate behavioral features automatically
* Predict insider risk probabilities
* Identify high-risk users

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* RandomForest
* Streamlit
* Joblib

---

## Applications

This system can be used in:

* Enterprise security monitoring
* Insider threat detection systems
* Behavioral analytics platforms
* Cybersecurity research

---

## Author

Vishal Kumar S

---

## License

This project is intended for academic and research purposes.
