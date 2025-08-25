"""
Interfaces para o layer de modelos seguindo princípios SOLID
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class IModel(ABC):
    """Interface para modelos de Machine Learning (Single Responsibility)"""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Treina o modelo"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições de probabilidade"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        pass


class IModelTrainer(ABC):
    """Interface para treinamento de modelos (Single Responsibility)"""
    
    @abstractmethod
    def train_model(self, model: IModel, X_train: pd.DataFrame, y_train: pd.Series) -> IModel:
        """Treina um modelo específico"""
        pass
    
    @abstractmethod
    def train_multiple_models(self, models: Dict[str, IModel], 
                            X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, IModel]:
        """Treina múltiplos modelos"""
        pass
    
    @abstractmethod
    def get_training_info(self) -> Dict[str, Any]:
        """Retorna informações do treinamento"""
        pass


class IModelEvaluator(ABC):
    """Interface para avaliação de modelos (Single Responsibility)"""
    
    @abstractmethod
    def evaluate_model(self, model: IModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Avalia performance de um modelo"""
        pass
    
    @abstractmethod
    def evaluate_multiple_models(self, models: Dict[str, IModel], 
        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Avalia performance de múltiplos modelos"""
        pass
    
    @abstractmethod
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Retorna relatório completo de avaliação"""
        pass

