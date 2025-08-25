"""
Script para baixar o dataset Telco Customer Churn
"""
import os
import pandas as pd
import requests
from io import StringIO


def download_dataset():
    """Baixa o dataset Telco Customer Churn"""
    print("Baixando dataset Telco Customer Churn...")
    
    # URL do dataset (versão pública - fonte alternativa)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # URL alternativa se a primeira falhar
    url_alt = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    try:
        # Cria diretório data se não existir
        os.makedirs("data", exist_ok=True)
        
        # Faz o download
        response = requests.get(url)
        response.raise_for_status()
        
        # Salva o arquivo
        file_path = "data/telco_customer_churn.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Verifica se o arquivo foi salvo corretamente
        df = pd.read_csv(file_path)
        print(f"Dataset baixado com sucesso!")
        print(f"Arquivo: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Colunas: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"Erro ao baixar dataset: {str(e)}")
        print("Por favor, baixe manualmente o dataset do Kaggle:")
        print("https://www.kaggle.com/blastchar/telco-customer-churn")
        print("E coloque-o em: data/telco_customer_churn.csv")
        return False


if __name__ == "__main__":
    download_dataset()

