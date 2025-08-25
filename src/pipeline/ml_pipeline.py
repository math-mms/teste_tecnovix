"""
Pipeline principal de Machine Learning seguindo princípios SOLID
"""
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, List
from src.data.interfaces import IDataLoader, IDataCleaner, IFeatureEngineer
from src.models.interfaces import IModel, IModelTrainer, IModelEvaluator
from src.models.concrete_models import LogisticRegressionModel, RandomForestModel, GradientBoostingModel


class MLPipeline:
    """Pipeline principal que orquestra todo o processo de ML (Dependency Inversion Principle)"""
    
    def __init__(self, 
                 data_loader: IDataLoader,
                 data_cleaner: IDataCleaner,
                 feature_engineer: IFeatureEngineer,
                 model_trainer: IModelTrainer,
                 model_evaluator: IModelEvaluator,
                 config: Dict[str, Any] = None):
        
        # Injeção de dependências (Dependency Inversion Principle)
        self.data_loader = data_loader
        self.data_cleaner = data_cleaner
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        
        # Configuração
        self.config = config or {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Estado do pipeline
        self.raw_data = None
        self.cleaned_data = None
        self.engineered_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        self.pipeline_results = {}
    
    def run(self) -> Dict[str, Any]:
        """Executa o pipeline completo"""
        self.logger.info("Iniciando execução do pipeline de ML")
        start_time = time.time()
        
        try:
            # Fase 1: Carregamento de dados
            self._load_data()
            
            # Fase 2: Limpeza de dados
            self._clean_data()
            
            # Fase 3: Engenharia de features
            self._engineer_features()
            
            # Fase 4: Preparação dos dados
            self._prepare_data()
            
            # Fase 5: Criação dos modelos
            self._create_models()
            
            # Fase 6: Treinamento dos modelos
            self._train_models()
            
            # Fase 7: Avaliação dos modelos
            self._evaluate_models()
            
            # Fase 8: Geração de resultados
            self._generate_results()
            
            # Calcula tempo total
            total_time = time.time() - start_time
            self.pipeline_results['total_execution_time'] = total_time
            
            self.logger.info(f"Pipeline concluído com sucesso em {total_time:.2f}s")
            
            return self.pipeline_results
            
        except Exception as e:
            self.logger.error(f"Erro na execução do pipeline: {str(e)}")
            raise
    
    def _load_data(self) -> None:
        """Fase 1: Carregamento de dados"""
        self.logger.info("Fase 1: Carregando dados")
        
        self.raw_data = self.data_loader.load_data()
        
        # Valida dados
        if not self.data_loader.validate_data(self.raw_data):
            raise ValueError("Dados não passaram na validação")
        
        self.logger.info(f"Dados carregados: {self.raw_data.shape}")
    
    def _clean_data(self) -> None:
        """Fase 2: Limpeza de dados"""
        self.logger.info("Fase 2: Limpando dados")
        
        self.cleaned_data = self.data_cleaner.clean_data(self.raw_data)
        
        # Registra relatório de limpeza
        cleaning_report = self.data_cleaner.get_cleaning_report()
        self.pipeline_results['cleaning_report'] = cleaning_report
        
        self.logger.info(f"Dados limpos: {self.cleaned_data.shape}")
    
    def _engineer_features(self) -> None:
        """Fase 3: Engenharia de features"""
        self.logger.info("Fase 3: Aplicando engenharia de features")
        
        self.engineered_data = self.feature_engineer.engineer_features(self.cleaned_data)
        
        # Registra informações das features
        feature_info = self.feature_engineer.get_feature_info()
        self.pipeline_results['feature_info'] = feature_info
        
        self.logger.info(f"Features criadas: {len(self.engineered_data.columns)}")
    
    def _prepare_data(self) -> None:
        """Fase 4: Preparação dos dados para treinamento"""
        self.logger.info("Fase 4: Preparando dados para treinamento")
        
        # Separa features e target
        if 'Churn' not in self.cleaned_data.columns:
            raise ValueError("Coluna 'Churn' não encontrada nos dados")
        
        # Converte target para numérico
        y = (self.cleaned_data['Churn'] == 'Yes').astype(int)
        X = self.engineered_data
        
        # Divide dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = self.model_trainer.split_data(X, y)
        
        self.logger.info(f"Dados preparados: Treino={self.X_train.shape}, Teste={self.X_test.shape}")
    
    def _create_models(self) -> None:
        """Fase 5: Criação dos modelos"""
        self.logger.info("Fase 5: Criando modelos")
        
        # Cria diferentes modelos (Strategy Pattern)
        self.models = {
            'LogisticRegression': LogisticRegressionModel(),
            'RandomForest': RandomForestModel(),
            'GradientBoosting': GradientBoostingModel()
        }
        
        self.logger.info(f"Modelos criados: {list(self.models.keys())}")
    
    def _train_models(self) -> None:
        """Fase 6: Treinamento dos modelos"""
        self.logger.info("Fase 6: Treinando modelos")
        
        self.trained_models = self.model_trainer.train_multiple_models(
            self.models, self.X_train, self.y_train
        )
        
        # Registra informações de treinamento
        training_info = self.model_trainer.get_training_info()
        self.pipeline_results['training_info'] = training_info
        
        self.logger.info(f"Modelos treinados: {len(self.trained_models)}")
    
    def _evaluate_models(self) -> None:
        """Fase 7: Avaliação dos modelos"""
        self.logger.info("Fase 7: Avaliando modelos")
        
        self.evaluation_results = self.model_evaluator.evaluate_multiple_models(
            self.trained_models, self.X_test, self.y_test
        )
        
        # Registra relatório de avaliação
        evaluation_report = self.model_evaluator.get_evaluation_report()
        self.pipeline_results['evaluation_report'] = evaluation_report
        
        self.logger.info(f"Modelos avaliados: {len(self.evaluation_results)}")
    
    def _generate_results(self) -> None:
        """Fase 8: Geração de resultados finais"""
        self.logger.info("Fase 8: Gerando resultados finais")
        
        # Melhor modelo
        best_model_name = self.model_evaluator.best_model
        best_model = self.trained_models.get(best_model_name)
        
        # Comparação de modelos
        model_comparison = self.model_evaluator.get_model_comparison()
        
        # Importância de features do melhor modelo
        feature_importance = {}
        if best_model:
            feature_importance = best_model.get_feature_importance()
        
        # Resultados finais
        self.pipeline_results.update({
            'best_model': best_model_name,
            'best_model_metrics': self.evaluation_results.get(best_model_name, {}),
            'model_comparison': model_comparison.to_dict('records'),
            'feature_importance': feature_importance,
            'success': True
        })
        
        self.logger.info(f"Melhor modelo: {best_model_name}")
    
    def get_results(self) -> Dict[str, Any]:
        """Retorna resultados do pipeline"""
        return self.pipeline_results
    
    def get_best_model(self) -> IModel:
        """Retorna o melhor modelo treinado"""
        best_model_name = self.pipeline_results.get('best_model')
        if best_model_name:
            return self.trained_models.get(best_model_name)
        return None
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Faz predições usando o melhor modelo"""
        best_model = self.get_best_model()
        if not best_model:
            raise ValueError("Nenhum modelo treinado disponível")
        
        # Aplica o mesmo processamento dos dados
        cleaned_data = self.data_cleaner.clean_data(data)
        engineered_data = self.feature_engineer.engineer_features(cleaned_data)
        
        return best_model.predict(engineered_data)

