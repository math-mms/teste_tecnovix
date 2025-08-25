"""
Sistema de configuração centralizada
"""
import logging
from typing import Dict, Any


def setup_logging(level: str = "INFO") -> None:
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_pipeline.log')
        ]
    )


def get_default_config() -> Dict[str, Any]:
    """Retorna configuração padrão do pipeline"""
    return {
        'data': {
            'file_path': 'data/telco_customer_churn.csv',
            'test_size': 0.2,
            'random_state': 42
        },
        'models': {
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        },
        'logging': {
            'level': 'INFO'
        }
    }

