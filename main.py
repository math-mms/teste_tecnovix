"""
Script principal para execução do pipeline de Machine Learning
Análise de Churn de Clientes - Dataset Telco Customer Churn
"""
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import setup_logging, get_default_config
from src.data.data_loader import TelcoDataLoader
from src.data.data_cleaner import TelcoDataCleaner
from src.data.feature_engineer import TelcoFeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.pipeline.ml_pipeline import MLPipeline


def main():
    """Função principal que executa o pipeline completo"""
    print("=" * 80)
    print("PIPELINE DE MACHINE LEARNING - ANÁLISE DE CHURN DE CLIENTES")
    print("=" * 80)
    print(f"Início da execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Configuração inicial
        print("1. Configurando ambiente...")
        config = get_default_config()
        setup_logging(config['logging']['level'])
        
        # 2. Criação dos componentes
        print("2. Criando componentes do pipeline...")
        data_loader = TelcoDataLoader(config['data']['file_path'])
        data_cleaner = TelcoDataCleaner()
        feature_engineer = TelcoFeatureEngineer()
        model_trainer = ModelTrainer(
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        model_evaluator = ModelEvaluator()
        
        # 3. Criação do pipeline
        print("3. Inicializando pipeline...")
        pipeline = MLPipeline(
            data_loader=data_loader,
            data_cleaner=data_cleaner,
            feature_engineer=feature_engineer,
            model_trainer=model_trainer,
            model_evaluator=model_evaluator,
            config=config
        )
        
        # 4. Execução do pipeline
        print("4. Executando pipeline...")
        print("-" * 50)
        results = pipeline.run()
        print("-" * 50)
        
        # 5. Exibição dos resultados
        print("5. Resultados obtidos:")
        print(f"   Tempo total de execução: {results['total_execution_time']:.2f}s")
        print(f"   Melhor modelo: {results['best_model']}")
        
        if results['best_model_metrics']:
            metrics = results['best_model_metrics']
            print(f"   Acurácia: {metrics['accuracy']:.4f}")
            print(f"   Precisão: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   AUC-ROC: {metrics['roc_auc']:.4f}")
        
        # 6. Salvamento dos resultados
        print("6. Salvando resultados...")
        save_results(results)
        
        # 7. Geração do relatório
        print("7. Gerando relatório...")
        generate_report(results)
        
        print()
        print("=" * 80)
        print("PIPELINE EXECUTADO COM SUCESSO!")
        print(f"Fim da execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"ERRO na execução do pipeline: {str(e)}")
        print("Verifique o arquivo de log para mais detalhes.")
        raise


def save_results(results: dict) -> None:
    """Salva os resultados em arquivo JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    
    # Função para converter tipos não serializáveis
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Remove objetos não serializáveis e converte tipos
    serializable_results = {}
    for key, value in results.items():
        if key in ['cleaning_report', 'feature_info', 'training_info', 
                  'evaluation_report', 'best_model_metrics', 'model_comparison',
                  'feature_importance', 'success', 'total_execution_time',
                  'best_model']:
            serializable_results[key] = convert_to_serializable(value)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"   Resultados salvos em: {filename}")


def generate_report(results: dict) -> None:
    """Gera relatório em formato Markdown"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"REPORT_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Análise de Churn de Clientes\n\n")
        f.write(f"**Data de execução:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Resumo executivo
        f.write("## Resumo Executivo\n\n")
        f.write(f"- **Tempo total de execução:** {results['total_execution_time']:.2f} segundos\n")
        f.write(f"- **Melhor modelo:** {results['best_model']}\n")
        
        if results['best_model_metrics']:
            metrics = results['best_model_metrics']
            f.write(f"- **Acurácia:** {metrics['accuracy']:.4f}\n")
            f.write(f"- **F1-Score:** {metrics['f1_score']:.4f}\n")
            f.write(f"- **AUC-ROC:** {metrics['roc_auc']:.4f}\n")
        
        f.write("\n")
        
        # Informações dos dados
        if 'cleaning_report' in results:
            f.write("## Informações dos Dados\n\n")
            cleaning = results['cleaning_report']
            f.write(f"- **Shape inicial:** {cleaning.get('initial_shape', 'N/A')}\n")
            f.write(f"- **Shape final:** {cleaning.get('final_shape', 'N/A')}\n")
            f.write(f"- **Linhas removidas:** {cleaning.get('rows_removed', 0)}\n")
            f.write(f"- **Valores ausentes iniciais:** {cleaning.get('initial_missing', 0)}\n")
            f.write(f"- **Valores ausentes finais:** {cleaning.get('final_missing', 0)}\n")
            f.write("\n")
        
        # Informações das features
        if 'feature_info' in results:
            f.write("## Engenharia de Features\n\n")
            feature_info = results['feature_info']
            f.write(f"- **Total de features:** {feature_info.get('total_features', 0)}\n")
            f.write(f"- **Features derivadas:** {feature_info.get('derived_features', 0)}\n")
            f.write("\n")
        
        # Comparação de modelos
        if 'model_comparison' in results:
            f.write("## Comparação de Modelos\n\n")
            f.write("| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |\n")
            f.write("|--------|----------|----------|--------|----------|----------|\n")
            
            for model in results['model_comparison']:
                f.write(f"| {model['Model']} | {model['Accuracy']:.4f} | {model['Precision']:.4f} | "
                       f"{model['Recall']:.4f} | {model['F1-Score']:.4f} | {model['AUC-ROC']:.4f} |\n")
            f.write("\n")
        
        # Importância das features
        if 'feature_importance' in results and results['feature_importance']:
            f.write("## Top 10 Features Mais Importantes\n\n")
            f.write("| Feature | Importância |\n")
            f.write("|---------|-------------|\n")
            
            # Ordena features por importância
            sorted_features = sorted(
                results['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            for feature, importance in sorted_features:
                f.write(f"| {feature} | {importance:.4f} |\n")
            f.write("\n")
        
        # Insights e recomendações
        f.write("## Insights e Recomendações\n\n")
        f.write("### Principais Fatores que Influenciam o Churn:\n")
        if 'feature_importance' in results and results['feature_importance']:
            top_features = sorted(
                results['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            for i, (feature, importance) in enumerate(top_features, 1):
                f.write(f"{i}. **{feature}** - Importância: {importance:.4f}\n")
        
        f.write("\n### Estratégias de Retenção:\n")
        f.write("1. **Foco em contratos longos:** Clientes com contratos month-to-month têm maior propensão ao churn\n")
        f.write("2. **Monitoramento de cargas:** Clientes com cargas mensais altas são mais propensos ao churn\n")
        f.write("3. **Serviços adicionais:** Oferecer serviços complementares pode reduzir o churn\n")
        f.write("4. **Suporte técnico:** Melhorar o suporte técnico pode impactar positivamente a retenção\n")
        f.write("5. **Programas de fidelidade:** Implementar programas para clientes seniores\n")
    
    print(f"   Relatório gerado em: {filename}")


if __name__ == "__main__":
    main()

