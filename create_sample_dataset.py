#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para criar um dataset de exemplo do Telco Customer Churn
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_dataset():
    """Cria um dataset de exemplo do Telco Customer Churn"""
    print("Criando dataset de exemplo Telco Customer Churn...")
    
    # Parâmetros do dataset
    n_customers = 1000
    
    # Gerar dados aleatórios
    np.random.seed(42)
    
    # Dados demográficos
    gender = np.random.choice(['Male', 'Female'], n_customers)
    senior_citizen = np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    partner = np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5])
    dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7])
    
    # Dados de serviços
    phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.4, 0.4, 0.2])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.3, 0.4, 0.3])
    
    # Serviços online
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2])
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2])
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2])
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.4, 0.4, 0.2])
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.4, 0.4, 0.2])
    
    # Contrato e pagamento
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.5, 0.3, 0.2])
    paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_customers)
    
    # Valores monetários
    monthly_charges = np.random.normal(65, 30, n_customers)
    monthly_charges = np.clip(monthly_charges, 20, 120)
    
    total_charges = monthly_charges * np.random.uniform(1, 72, n_customers)
    
    # Tenure (tempo como cliente)
    tenure = np.random.exponential(30, n_customers)
    tenure = np.clip(tenure, 1, 72).astype(int)
    
    # Gerar churn baseado em features
    churn_prob = np.zeros(n_customers)
    
    # Fatores que aumentam churn
    churn_prob += (contract == 'Month-to-month') * 0.3
    churn_prob += (internet_service == 'Fiber optic') * 0.2
    churn_prob += (payment_method == 'Electronic check') * 0.1
    churn_prob += (tenure < 12) * 0.2
    churn_prob += (monthly_charges > 80) * 0.1
    churn_prob += (senior_citizen == 1) * 0.1
    
    # Fatores que diminuem churn
    churn_prob -= (contract == 'Two year') * 0.3
    churn_prob -= (payment_method == 'Credit card (automatic)') * 0.1
    churn_prob -= (tenure > 24) * 0.2
    
    # Normalizar probabilidades
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Gerar churn
    churn = np.random.binomial(1, churn_prob)
    churn = np.where(churn == 1, 'Yes', 'No')
    
    # Criar DataFrame
    data = {
        'customerID': [f'CUST{i:04d}' for i in range(1, n_customers + 1)],
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges.round(2),
        'TotalCharges': total_charges.round(2),
        'Churn': churn
    }
    
    df = pd.DataFrame(data)
    
    # Criar diretório data se não existir
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Salvar dataset
    file_path = "data/telco_customer_churn.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Dataset criado com sucesso!")
    print(f"Arquivo: {file_path}")
    print(f"Shape: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nDistribuição do Churn:")
    print(df['Churn'].value_counts())
    print(f"\nPrimeiras 5 linhas:")
    print(df.head())
    
    return True

if __name__ == "__main__":
    create_sample_dataset()
