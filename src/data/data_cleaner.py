"""
Implementação do DataCleaner para limpeza de dados
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .interfaces import IDataCleaner


class TelcoDataCleaner(IDataCleaner):
    """Limpeza de dados específica para dataset Telco Customer Churn"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleaning_report = {}
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpeza e tratamento aos dados"""
        self.logger.info("Iniciando limpeza de dados")
        cleaned_data = data.copy()
        
        # Registra informações iniciais
        initial_shape = cleaned_data.shape
        initial_missing = cleaned_data.isnull().sum().sum()
        
        # 1. Tratamento de valores ausentes
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # 2. Tratamento de valores inconsistentes
        cleaned_data = self._handle_inconsistent_values(cleaned_data)
        
        # 3. Conversão de tipos de dados
        cleaned_data = self._convert_data_types(cleaned_data)
        
        # 4. Remoção de duplicatas
        cleaned_data = self._remove_duplicates(cleaned_data)
        
        # Registra informações finais
        final_shape = cleaned_data.shape
        final_missing = cleaned_data.isnull().sum().sum()
        
        # Atualiza relatório
        self.cleaning_report = {
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'initial_missing': initial_missing,
            'final_missing': final_missing,
            'rows_removed': initial_shape[0] - final_shape[0],
            'columns_removed': initial_shape[1] - final_shape[1]
        }
        
        self.logger.info(f"Limpeza concluída. Shape final: {final_shape}")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Trata valores ausentes"""
        # Para colunas numéricas, preenche com mediana
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_columns:
            if col in data.columns:
                if data[col].isnull().any():
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    self.logger.info(f"Preenchidos valores ausentes em {col} com mediana: {median_val}")
        
        # Para colunas categóricas, preenche com moda
        categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_columns:
            if col in data.columns:
                if data[col].isnull().any():
                    mode_val = data[col].mode()[0]
                    data[col].fillna(mode_val, inplace=True)
                    self.logger.info(f"Preenchidos valores ausentes em {col} com moda: {mode_val}")
        
        return data
    
    def _handle_inconsistent_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Trata valores inconsistentes"""
        # Remove espaços em branco de colunas categóricas
        categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod', 'Churn']
        
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()
        
        # Trata valores 'No internet service' e 'No phone service'
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_columns:
            if col in data.columns:
                data[col] = data[col].replace(['No internet service', 'No phone service'], 'No')
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Converte tipos de dados"""
        # Converte TotalCharges para numérico
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            # Preenche valores que não puderam ser convertidos com 0
            data['TotalCharges'].fillna(0, inplace=True)
        
        # Converte SeniorCitizen para int se necessário
        if 'SeniorCitizen' in data.columns:
            data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove linhas duplicadas"""
        initial_rows = len(data)
        data.drop_duplicates(inplace=True)
        final_rows = len(data)
        
        if initial_rows != final_rows:
            self.logger.info(f"Removidas {initial_rows - final_rows} linhas duplicadas")
        
        return data
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Retorna relatório detalhado da limpeza"""
        return self.cleaning_report

