# -*- coding: utf-8 -*-
"""
Script para baixar o dataset Telco Customer Churn
Compativel com Python 2.7
"""
import os
import pandas as pd
import requests
from io import StringIO


def download_dataset():
    """Baixa o dataset Telco Customer Churn"""
    print("Baixando dataset Telco Customer Churn...")
    
    # URL do dataset (versao publica)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    try:
        # Cria diretorio data se nao existir
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Faz o download
        response = requests.get(url)
        response.raise_for_status()
        
        # Salva o arquivo
        file_path = "data/telco_customer_churn.csv"
        with open(file_path, 'w') as f:
            f.write(response.text.encode('utf-8'))
        
        # Verifica se o arquivo foi salvo corretamente
        df = pd.read_csv(file_path)
        print("Dataset baixado com sucesso!")
        print("Arquivo: {0}".format(file_path))
        print("Shape: {0}".format(df.shape))
        print("Colunas: {0}".format(list(df.columns)))
        
        return True
        
    except Exception as e:
        print("Erro ao baixar dataset: {0}".format(str(e)))
        print("Por favor, baixe manualmente o dataset do Kaggle:")
        print("https://www.kaggle.com/blastchar/telco-customer-churn")
        print("E coloque-o em: data/telco_customer_churn.csv")
        return False


if __name__ == "__main__":
    download_dataset()

