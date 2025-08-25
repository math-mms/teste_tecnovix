"""
Interfaces para o layer de dados seguindo princípios SOLID
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class IDataLoader(ABC):
    """Interface para carregamento de dados (Single Responsibility)"""
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Carrega dados do dataset"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Valida estrutura dos dados"""
        pass


class IDataCleaner(ABC):
    """Interface para limpeza de dados (Single Responsibility)"""
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpa e trata os dados"""
        pass
    
    @abstractmethod
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Retorna relatório de limpeza"""
        pass


class IFeatureEngineer(ABC):
    """Interface para engenharia de features (Single Responsibility)"""
    
    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica engenharia de features"""
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """Retorna informações sobre as features criadas"""
        pass

