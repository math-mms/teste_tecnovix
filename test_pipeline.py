"""
Script de teste para verificar se o pipeline está funcionando
"""
import sys
import os
import pandas as pd
import numpy as np

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Testa se todas as importações estão funcionando"""
    print("Testando importações...")
    
    try:
        from src.utils.config import setup_logging, get_default_config
        from src.data.data_loader import TelcoDataLoader
        from src.data.data_cleaner import TelcoDataCleaner
        from src.data.feature_engineer import TelcoFeatureEngineer
        from src.models.model_trainer import ModelTrainer
        from src.models.model_evaluator import ModelEvaluator
        from src.pipeline.ml_pipeline import MLPipeline
        print("✅ Todas as importações funcionaram!")
        return True
    except Exception as e:
        print(f"❌ Erro nas importações: {str(e)}")
        return False


def test_data_loading():
    """Testa o carregamento de dados"""
    print("\nTestando carregamento de dados...")
    
    try:
        from src.data.data_loader import TelcoDataLoader
        
        # Verifica se o arquivo existe
        file_path = "data/telco_customer_churn.csv"
        if not os.path.exists(file_path):
            print(f"❌ Arquivo não encontrado: {file_path}")
            print("Execute: python download_dataset.py")
            return False
        
        # Testa carregamento
        loader = TelcoDataLoader(file_path)
        data = loader.load_data()
        
        print(f"✅ Dados carregados com sucesso!")
        print(f"   Shape: {data.shape}")
        print(f"   Colunas: {len(data.columns)}")
        
        # Testa validação
        is_valid = loader.validate_data(data)
        print(f"   Validação: {'✅ Passou' if is_valid else '❌ Falhou'}")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Erro no carregamento: {str(e)}")
        return False


def test_data_cleaning():
    """Testa a limpeza de dados"""
    print("\nTestando limpeza de dados...")
    
    try:
        from src.data.data_loader import TelcoDataLoader
        from src.data.data_cleaner import TelcoDataCleaner
        
        # Carrega dados
        loader = TelcoDataLoader()
        data = loader.load_data()
        
        # Testa limpeza
        cleaner = TelcoDataCleaner()
        cleaned_data = cleaner.clean_data(data)
        
        print(f"✅ Limpeza concluída!")
        print(f"   Shape original: {data.shape}")
        print(f"   Shape limpo: {cleaned_data.shape}")
        
        # Verifica se há valores ausentes
        missing_count = cleaned_data.isnull().sum().sum()
        print(f"   Valores ausentes finais: {missing_count}")
        
        return missing_count == 0
        
    except Exception as e:
        print(f"❌ Erro na limpeza: {str(e)}")
        return False


def test_feature_engineering():
    """Testa a engenharia de features"""
    print("\nTestando engenharia de features...")
    
    try:
        from src.data.data_loader import TelcoDataLoader
        from src.data.data_cleaner import TelcoDataCleaner
        from src.data.feature_engineer import TelcoFeatureEngineer
        
        # Carrega e limpa dados
        loader = TelcoDataLoader()
        cleaner = TelcoDataCleaner()
        data = loader.load_data()
        cleaned_data = cleaner.clean_data(data)
        
        # Testa engenharia de features
        engineer = TelcoFeatureEngineer()
        engineered_data = engineer.engineer_features(cleaned_data)
        
        print(f"✅ Engenharia de features concluída!")
        print(f"   Features originais: {len(cleaned_data.columns)}")
        print(f"   Features finais: {len(engineered_data.columns)}")
        
        # Verifica se há features derivadas
        derived_features = [col for col in engineered_data.columns if col.startswith('feature_')]
        print(f"   Features derivadas: {len(derived_features)}")
        
        return len(derived_features) > 0
        
    except Exception as e:
        print(f"❌ Erro na engenharia de features: {str(e)}")
        return False


def test_model_creation():
    """Testa a criação de modelos"""
    print("\nTestando criação de modelos...")
    
    try:
        from src.models.concrete_models import LogisticRegressionModel, RandomForestModel, GradientBoostingModel
        
        # Cria modelos
        models = {
            'LogisticRegression': LogisticRegressionModel(),
            'RandomForest': RandomForestModel(),
            'GradientBoosting': GradientBoostingModel()
        }
        
        print(f"✅ Modelos criados com sucesso!")
        for name, model in models.items():
            print(f"   {name}: {model.model_name}")
        
        return len(models) == 3
        
    except Exception as e:
        print(f"❌ Erro na criação de modelos: {str(e)}")
        return False


def test_pipeline_integration():
    """Testa a integração do pipeline"""
    print("\nTestando integração do pipeline...")
    
    try:
        from src.utils.config import setup_logging, get_default_config
        from src.data.data_loader import TelcoDataLoader
        from src.data.data_cleaner import TelcoDataCleaner
        from src.data.feature_engineer import TelcoFeatureEngineer
        from src.models.model_trainer import ModelTrainer
        from src.models.model_evaluator import ModelEvaluator
        from src.pipeline.ml_pipeline import MLPipeline
        
        # Configuração
        config = get_default_config()
        setup_logging("WARNING")  # Reduz logs para teste
        
        # Cria componentes
        data_loader = TelcoDataLoader()
        data_cleaner = TelcoDataCleaner()
        feature_engineer = TelcoFeatureEngineer()
        model_trainer = ModelTrainer(test_size=0.3)  # Teste mais rápido
        model_evaluator = ModelEvaluator()
        
        # Cria pipeline
        pipeline = MLPipeline(
            data_loader=data_loader,
            data_cleaner=data_cleaner,
            feature_engineer=feature_engineer,
            model_trainer=model_trainer,
            model_evaluator=model_evaluator,
            config=config
        )
        
        print(f"✅ Pipeline criado com sucesso!")
        print(f"   Componentes: {len([data_loader, data_cleaner, feature_engineer, model_trainer, model_evaluator])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {str(e)}")
        return False


def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("TESTE DO PIPELINE DE MACHINE LEARNING")
    print("=" * 60)
    
    tests = [
        ("Importações", test_imports),
        ("Carregamento de Dados", test_data_loading),
        ("Limpeza de Dados", test_data_cleaning),
        ("Engenharia de Features", test_feature_engineering),
        ("Criação de Modelos", test_model_creation),
        ("Integração do Pipeline", test_pipeline_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado: {str(e)}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! O pipeline está pronto para uso.")
        print("Execute: python main.py")
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
    
    return passed == total


if __name__ == "__main__":
    main()

