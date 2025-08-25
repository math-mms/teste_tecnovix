# -*- coding: utf-8 -*-
"""
Script de demonstracao completa do pipeline de Machine Learning
Compativel com Python 2.7
"""
import sys
import os
import time

# Adiciona o diretorio src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_demo():
    """Executa demonstracao completa do pipeline"""
    print("DEMONSTRACAO DO PIPELINE DE MACHINE LEARNING")
    print("=" * 60)
    
    # 1. Verificar se o dataset existe
    print("\n1. Verificando dataset...")
    if not os.path.exists("data/telco_customer_churn.csv"):
        print("Dataset nao encontrado. Baixando...")
        try:
            from download_dataset import download_dataset
            if not download_dataset():
                print("Falha ao baixar dataset. Execute manualmente: python download_dataset.py")
                return False
        except Exception as e:
            print("Erro: {0}".format(str(e)))
            return False
    else:
        print("Dataset encontrado!")
    
    # 2. Executar testes
    print("\n2. Executando testes...")
    try:
        from test_pipeline import main as run_tests
        if not run_tests():
            print("Testes falharam. Verifique os erros acima.")
            return False
    except Exception as e:
        print("Erro nos testes: {0}".format(str(e)))
        return False
    
    # 3. Executar pipeline completo
    print("\n3. Executando pipeline completo...")
    start_time = time.time()
    
    try:
        from main import main as run_pipeline
        results = run_pipeline()
        
        total_time = time.time() - start_time
        print("\nTempo total da demonstracao: {0:.2f}s".format(total_time))
        
        # 4. Mostrar resultados finais
        print("\n4. Resultados Finais:")
        print("   Melhor modelo: {0}".format(results.get('best_model', 'N/A')))
        
        if results.get('best_model_metrics'):
            metrics = results['best_model_metrics']
            print("   Acuracia: {0:.4f}".format(metrics.get('accuracy', 0)))
            print("   F1-Score: {0:.4f}".format(metrics.get('f1_score', 0)))
            print("   AUC-ROC: {0:.4f}".format(metrics.get('roc_auc', 0)))
        
        print("\nArquivos gerados:")
        print("   - results_*.json (resultados detalhados)")
        print("   - REPORT_*.md (relatorio executivo)")
        print("   - ml_pipeline.log (logs de execucao)")
        
        print("\nDEMONSTRACAO CONCLUIDA COM SUCESSO!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("Erro na execucao do pipeline: {0}".format(str(e)))
        return False


if __name__ == "__main__":
    success = run_demo()
    if not success:
        print("\nDemonstracao falhou. Verifique os erros acima.")
        sys.exit(1)
    else:
        print("\nDemonstracao executada com sucesso!")

