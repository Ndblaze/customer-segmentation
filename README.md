# Customer Segmentation with KMeans and Deep Learning

## Project Overview
This project leverages **KMeans clustering** and a fully connected **deep learning model** to classify customers into predefined segments based on 17 financial and behavioral features. The objective is to empower businesses, such as **InsuredHub**, to gain valuable insights into customer behavior, enabling them to optimize their services and offerings.

By combining clustering for initial segmentation and deep learning for precise classification, this solution provides actionable insights that enhance decision-making in customer management and targeted service delivery.


## Dataset Description
The dataset comprises **17 features** that describe financial and behavioral attributes of customers, such as:

- **BALANCE**: Customer account balance.
- **PURCHASES**: Total purchases made by the customer.
- **CREDIT_LIMIT**: The credit card limit assigned to the customer.
- **ONEOFF_PURCHASES**: Purchases made in a single transaction.
- **INSTALLMENTS_PURCHASES**: Purchases made in installments.
- **CASH_ADVANCE**: Cash advances taken by the customer.
- **PURCHASES_FREQUENCY**: Frequency of purchases.
- **PRC_FULL_PAYMENT**: Proportion of months the customer fully paid their balance.
- **TENURE**: Number of months the account has been active.
- etc.
  
### Target Column
The target column, **Cluster**, categorizes customers into four predefined groups:

1. **High Spender**: Customers with high purchases and payments.
2. **Cash Advance Seeker**: Customers who rely heavily on cash advances.
3. **Minimal User**: Customers with minimal activity across all features.
4. **Installment User**: Customers who prefer installment purchases.


## Setup Instructions

### Prerequisites
- **Python 3.7 or higher**: Ensure Python is installed on your system.
- **Recommended**: Use a virtual environment for better dependency management.

### Step 1: Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/Ndblaze/customer-segmentation
cd models
```

### Step 2: Install Dependencies
Install the required Python libraries using the terminal:
```bash
pip install -r requirements.txt
```

### Step 3: Directory Structure
Ensure the following directory structure is maintained:


> **Note:** Ensure that the `models/` directory contains the trained models (`kmeans_model.pkl`, `deep_learning_model.h5`, and `scaler.pkl`) before running the application. Place the `Customer Data.csv` in the root directory for training and testing purposes.

### Step 4: Running the Flask Application
Start the application locally:
```bash
python app.py
```
The application will run at http://127.0.0.1:5000/.


Note: The trained model is saved with the files that you clone from GitHub, so when you run the application you can still make prediction using the saved model. If you want to retrain the model or make some changes to the hyperparameters, you can look at the Projectcode.ipynb file, open it and make all the changes you need, and run the file to build another model for you. After the model is built and saved, go back and repeat step 4











