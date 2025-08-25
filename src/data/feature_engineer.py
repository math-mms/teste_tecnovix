"""
Implementação do FeatureEngineer para engenharia de features
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .interfaces import IFeatureEngineer


class TelcoFeatureEngineer(IFeatureEngineer):
    """Engenharia de features específica para dataset Telco Customer Churn"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_info = {}
        self.categorical_columns = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica engenharia de features aos dados"""
        self.logger.info("Iniciando engenharia de features")
        engineered_data = data.copy()
        
        # 1. Criação de features derivadas
        engineered_data = self._create_derived_features(engineered_data)
        
        # 2. Encoding de variáveis categóricas
        engineered_data = self._encode_categorical_features(engineered_data)
        
        # 3. Normalização de features numéricas
        engineered_data = self._normalize_numeric_features(engineered_data)
        
        # 4. Seleção de features finais
        engineered_data = self._select_final_features(engineered_data)
        
        # Registra informações das features
        self.feature_info = {
            'total_features': len(engineered_data.columns),
            'categorical_features': len([col for col in engineered_data.columns if col in self.categorical_columns]),
            'numeric_features': len([col for col in engineered_data.columns if col in self.numeric_columns]),
            'derived_features': len([col for col in engineered_data.columns if col.startswith('feature_')]),
            'feature_names': list(engineered_data.columns)
        }
        
        self.logger.info(f"Engenharia de features concluída. Features finais: {len(engineered_data.columns)}")
        return engineered_data
    
    def _create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria features derivadas"""
        # Feature: Total de serviços contratados
        service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies']
        
        data['feature_total_services'] = 0
        for col in service_columns:
            if col in data.columns:
                data['feature_total_services'] += (data[col] != 'No').astype(int)
        
        # Feature: Cliente com família (Partner + Dependents)
        data['feature_has_family'] = ((data['Partner'] == 'Yes') | 
        (data['Dependents'] == 'Yes')).astype(int)
        
        # Feature: Cliente sênior
        if 'SeniorCitizen' in data.columns:
            data['feature_is_senior'] = data['SeniorCitizen']
        
        # Feature: Contrato longo (Month-to-month = 0, outros = 1)
        data['feature_long_contract'] = (data['Contract'] != 'Month-to-month').astype(int)
        
        # Feature: Múltiplas linhas
        data['feature_multiple_lines'] = (data['MultipleLines'] == 'Yes').astype(int)
        
        # Feature: Carga mensal por tempo de serviço
        data['feature_monthly_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
        
        # Feature: Carga total por tempo de serviço
        data['feature_total_per_tenure'] = data['TotalCharges'] / (data['tenure'] + 1)
        
        # Feature: Diferença entre carga total e mensal
        data['feature_charges_difference'] = data['TotalCharges'] - data['MonthlyCharges']
        
        self.logger.info("Features derivadas criadas com sucesso")
        return data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica encoding em variáveis categóricas"""
        for col in self.categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                self.logger.info(f"Encoding aplicado em: {col}")
        
        return data
    
    def _normalize_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normaliza features numéricas"""
        numeric_features = []
        
        # Adiciona features numéricas originais
        for col in self.numeric_columns:
            if col in data.columns:
                numeric_features.append(col)
        
        # Adiciona features derivadas numéricas
        derived_numeric = [col for col in data.columns if col.startswith('feature_') and 
        data[col].dtype in ['int64', 'float64']]
        numeric_features.extend(derived_numeric)
        
        if numeric_features:
            # Aplica normalização
            data[numeric_features] = self.scaler.fit_transform(data[numeric_features])
            self.logger.info(f"Normalização aplicada em {len(numeric_features)} features")
        
        return data
    
    def _select_final_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Seleciona features finais para o modelo"""
        # Remove colunas originais categóricas (mantém apenas as encoded)
        columns_to_drop = []
        for col in self.categorical_columns:
            if col in data.columns:
                columns_to_drop.append(col)
        
        # Remove customerID (não é feature preditiva)
        if 'customerID' in data.columns:
            columns_to_drop.append('customerID')
        
        # Remove target variable (será separada depois)
        if 'Churn' in data.columns:
            columns_to_drop.append('Churn')
        
        # Remove colunas com muitas categorias (mantém apenas encoded)
        high_cardinality = ['PaymentMethod']
        for col in high_cardinality:
            if col in columns_to_drop:
                columns_to_drop.remove(col)
        
        data = data.drop(columns=columns_to_drop, errors='ignore')
        
        # Garante que apenas features numéricas sejam mantidas
        numeric_columns = []
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_columns.append(col)
            else:
                self.logger.warning(f"Removendo coluna não numérica: {col} (tipo: {data[col].dtype})")
        
        data = data[numeric_columns]
        
        self.logger.info(f"Seleção de features concluída. Features finais: {len(data.columns)}")
        return data
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Retorna informações sobre as features criadas"""
        return self.feature_info
    
    def get_feature_importance_columns(self) -> List[str]:
        """Retorna lista de colunas para análise de importância de features"""
        return [col for col in self.feature_info.get('feature_names', []) 
                if not col.endswith('_encoded')]

