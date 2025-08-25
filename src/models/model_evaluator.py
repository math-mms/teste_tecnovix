"""
Implementação do ModelEvaluator para avaliação de modelos
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from .interfaces import IModelEvaluator, IModel


class ModelEvaluator(IModelEvaluator):
    """Avaliador de modelos seguindo princípios SOLID"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        self.best_model = None
    
    def evaluate_model(self, model: IModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Avalia performance de um modelo"""
        self.logger.info(f"Avaliando modelo: {model.model_name}")
        
        try:
            # Faz predições
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
            
            # Calcula métricas
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Armazena resultados
            self.evaluation_results[model.model_name] = metrics
            
            self.logger.info(f"Avaliação concluída para {model.model_name}")
            self.logger.info(f"F1-Score: {metrics['f1_score']:.4f}, AUC-ROC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação de {model.model_name}: {str(e)}")
            raise
    
    def evaluate_multiple_models(self, models: Dict[str, IModel], 
        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Avalia performance de múltiplos modelos"""
        self.logger.info(f"Avaliando {len(models)} modelos")
        
        all_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(model, X_test, y_test)
                all_results[model_name] = results
                
            except Exception as e:
                self.logger.error(f"Falha na avaliação de {model_name}: {str(e)}")
                continue
        
        # Identifica o melhor modelo
        self._identify_best_model(all_results)
        
        self.logger.info(f"Avaliação concluída. {len(all_results)}/{len(models)} modelos avaliados")
        
        return all_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calcula todas as métricas de avaliação"""
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # AUC-ROC (se houver probabilidades)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            roc_auc = 0.0
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Métricas específicas por classe
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist()
        }
    
    def _identify_best_model(self, results: Dict[str, Dict[str, float]]) -> None:
        """Identifica o melhor modelo baseado em F1-Score"""
        best_model = None
        best_f1 = -1
        
        for model_name, metrics in results.items():
            if 'f1_score' in metrics and metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model = model_name
        
        self.best_model = best_model
        
        if best_model:
            self.logger.info(f"Melhor modelo identificado: {best_model} (F1-Score: {best_f1:.4f})")
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Retorna relatório completo de avaliação"""
        return {
            'evaluation_results': self.evaluation_results,
            'best_model': self.best_model,
            'model_count': len(self.evaluation_results)
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Retorna comparação dos modelos em formato DataFrame"""
        if not self.evaluation_results:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['roc_auc']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_feature_importance_comparison(self, models: Dict[str, IModel]) -> pd.DataFrame:
        """Retorna comparação de importância de features entre modelos"""
        importance_data = []
        
        for model_name, model in models.items():
            if model.is_trained:
                feature_importance = model.get_feature_importance()
                for feature, importance in feature_importance.items():
                    importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': importance
                    })
        
        return pd.DataFrame(importance_data)

