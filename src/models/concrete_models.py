"""
Implementações concretas de modelos de Machine Learning
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Implementação de Regressão Logística"""
    
    def __init__(self, **kwargs):
        super().__init__("LogisticRegression", **kwargs)
        self.model = LogisticRegression(random_state=42, **kwargs)
    
    def _train_implementation(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Implementação específica do treinamento"""
        self.model.fit(X_train, y_train)
    
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição"""
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição de probabilidade"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features (coeficientes para regressão logística)"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = float(abs(self.model.coef_[0][i]))
        
        return importance_dict


class RandomForestModel(BaseModel):
    """Implementação de Random Forest"""
    
    def __init__(self, **kwargs):
        # Parâmetros padrão otimizados para o problema
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        super().__init__("RandomForest", **default_params)
        self.model = RandomForestClassifier(**default_params)
    
    def _train_implementation(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Implementação específica do treinamento"""
        self.model.fit(X_train, y_train)
    
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição"""
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição de probabilidade"""
        return self.model.predict_proba(X)


class GradientBoostingModel(BaseModel):
    """Implementação de Gradient Boosting"""
    
    def __init__(self, **kwargs):
        # Parâmetros padrão otimizados para o problema
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("GradientBoosting", **default_params)
        self.model = GradientBoostingClassifier(**default_params)
    
    def _train_implementation(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Implementação específica do treinamento"""
        self.model.fit(X_train, y_train)
    
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição"""
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementação específica da predição de probabilidade"""
        return self.model.predict_proba(X)

