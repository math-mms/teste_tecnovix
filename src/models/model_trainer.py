"""
Implementação do ModelTrainer para treinamento de modelos
"""
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from .interfaces import IModelTrainer, IModel


class ModelTrainer(IModelTrainer):
    """Treinador de modelos seguindo princípios SOLID"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.training_info = {}
    
    def train_model(self, model: IModel, X_train: pd.DataFrame, y_train: pd.Series) -> IModel:
        """Treina um modelo específico"""
        self.logger.info(f"Iniciando treinamento do modelo: {model.model_name}")
        
        start_time = time.time()
        
        try:
            # Treina o modelo
            model.train(X_train, y_train)
            
            # Calcula tempo de treinamento
            training_time = time.time() - start_time
            
            # Registra informações do treinamento
            self.training_info[model.model_name] = {
                'training_time': training_time,
                'training_samples': len(X_train),
                'features_count': len(X_train.columns),
                'success': True
            }
            
            self.logger.info(f"Treinamento concluído para {model.model_name} em {training_time:.2f}s")
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Erro no treinamento de {model.model_name}: {str(e)}")
            
            self.training_info[model.model_name] = {
                'training_time': training_time,
                'error': str(e),
                'success': False
            }
            raise
        
        return model
    
    def train_multiple_models(self, models: Dict[str, IModel], 
                            X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, IModel]:
        """Treina múltiplos modelos"""
        self.logger.info(f"Iniciando treinamento de {len(models)} modelos")
        
        trained_models = {}
        
        for model_name, model in models.items():
            try:
                trained_model = self.train_model(model, X_train, y_train)
                trained_models[model_name] = trained_model
                
            except Exception as e:
                self.logger.error(f"Falha no treinamento de {model_name}: {str(e)}")
                # Continua com os próximos modelos
                continue
        
        self.logger.info(f"Treinamento concluído. {len(trained_models)}/{len(models)} modelos treinados com sucesso")
        
        return trained_models
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Divide os dados em treino e teste"""
        self.logger.info("Dividindo dados em treino e teste")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Mantém proporção das classes
        )
        
        self.logger.info(f"Dados divididos: Treino={len(X_train)}, Teste={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_training_info(self) -> Dict[str, Any]:
        """Retorna informações do treinamento"""
        return self.training_info
    
    def get_best_model(self, models: Dict[str, IModel], 
        evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """Retorna o nome do melhor modelo baseado em F1-Score"""
        best_model = None
        best_f1 = -1
        
        for model_name, metrics in evaluation_results.items():
            if 'f1_score' in metrics and metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model = model_name
        
        return best_model

