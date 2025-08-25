"""
Implementação do DataLoader para o dataset Telco Customer Churn
"""
import pandas as pd
import logging
from typing import Dict, Any
from .interfaces import IDataLoader


class TelcoDataLoader(IDataLoader):
    """Carregador de dados específico para dataset Telco Customer Churn"""
    
    def __init__(self, file_path: str = "data/telco_customer_churn.csv"):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        
        # Regras de validação (Open/Closed Principle)
        self.validation_rules = {
            'required_columns': ['customerID', 'Churn', 'tenure', 'MonthlyCharges'],
            'numeric_columns': ['tenure', 'MonthlyCharges', 'TotalCharges'],
            'categorical_columns': ['gender', 'Partner', 'Contract', 'InternetService'],
            'churn_values': ['Yes', 'No']
        }
    
    def load_data(self) -> pd.DataFrame:
        """Carrega dados do arquivo CSV"""
        try:
            self.logger.info(f"Carregando dados de: {self.file_path}")
            data = pd.read_csv(self.file_path)
            self.logger.info(f"Dados carregados com sucesso. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            self.logger.error(f"Arquivo não encontrado: {self.file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Valida estrutura e qualidade dos dados"""
        try:
            # Verifica colunas obrigatórias
            missing_required = set(self.validation_rules['required_columns']) - set(data.columns)
            if missing_required:
                self.logger.error(f"Colunas obrigatórias ausentes: {missing_required}")
                return False
            
            # Verifica valores de churn
            if 'Churn' in data.columns:
                invalid_churn = set(data['Churn'].unique()) - set(self.validation_rules['churn_values'])
                if invalid_churn:
                    self.logger.warning(f"Valores de churn inválidos encontrados: {invalid_churn}")
            
            # Verifica tipos de dados numéricos
            for col in self.validation_rules['numeric_columns']:
                if col in data.columns:
                    try:
                        pd.to_numeric(data[col], errors='raise')
                    except:
                        self.logger.warning(f"Coluna {col} não é numérica")
            
            self.logger.info("Validação de dados concluída com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação: {str(e)}")
            return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os dados carregados"""
        try:
            data = self.load_data()
            return {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'churn_distribution': data['Churn'].value_counts().to_dict() if 'Churn' in data.columns else {}
            }
        except Exception as e:
            self.logger.error(f"Erro ao obter informações dos dados: {str(e)}")
            return {}

