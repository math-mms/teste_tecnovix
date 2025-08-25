"""
Modelo base abstrato para implementações de ML
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from abc import ABC
from .interfaces import IModel
from abc import abstractmethod


class BaseModel(IModel, ABC):
    """Classe base abstrata para todos os modelos (Liskov Substitution Principle)"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        self.model_params = kwargs
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Treina o modelo (implementação base)"""
        self.logger.info(f"Iniciando treinamento do modelo: {self.model_name}")
        
        # Validação de entrada
        if X_train.empty or y_train.empty:
            raise ValueError("Dados de treinamento não podem estar vazios")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train e y_train devem ter o mesmo número de amostras")
        
        # Armazena nomes das features
        self.feature_names = list(X_train.columns)
        
        # Chama implementação específica
        self._train_implementation(X_train, y_train)
        
        self.is_trained = True
        self.logger.info(f"Treinamento concluído para: {self.model_name}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições (implementação base)"""
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
        
        if self.model is None:
            raise ValueError("Modelo não foi inicializado corretamente")
        
        return self._predict_implementation(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições de probabilidade (implementação base)"""
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
        
        if self.model is None:
            raise ValueError("Modelo não foi inicializado corretamente")
        
        return self._predict_proba_implementation(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features (implementação base)"""
        if not self.is_trained:
            return {}
        
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = float(self.model.feature_importances_[i])
        
        return importance_dict
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'model_params': self.model_params,
            'feature_names': self.feature_names
        }
    
    @abstractmethod
    def _train_implementation(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Implementação específica do treinamento"""
        pass
    
    @abstractmethod
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição"""
        pass
    
    @abstractmethod
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição de probabilidade"""
        pass

