"""
Script de demonstração completa do pipeline de Machine Learning
"""
import sys
import os
import time

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_demo():
    """Executa demonstração completa do pipeline"""
    print("🚀 DEMONSTRAÇÃO DO PIPELINE DE MACHINE LEARNING")
    print("=" * 60)
    
    # 1. Verificar se o dataset existe
    print("\n1️⃣ Verificando dataset...")
    if not os.path.exists("data/telco_customer_churn.csv"):
        print("📥 Dataset não encontrado. Baixando...")
        try:
            from download_dataset import download_dataset
            if not download_dataset():
                print("❌ Falha ao baixar dataset. Execute manualmente: python download_dataset.py")
                return False
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            return False
    else:
        print("✅ Dataset encontrado!")
    
    # 2. Executar testes
    print("\n2️⃣ Executando testes...")
    try:
        from test_pipeline import main as run_tests
        if not run_tests():
            print("❌ Testes falharam. Verifique os erros acima.")
            return False
    except Exception as e:
        print(f"❌ Erro nos testes: {str(e)}")
        return False
    
    # 3. Executar pipeline completo
    print("\n3️⃣ Executando pipeline completo...")
    start_time = time.time()
    
    try:
        from main import main as run_pipeline
        results = run_pipeline()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Tempo total da demonstração: {total_time:.2f}s")
        
        # 4. Mostrar resultados finais
        print("\n4️⃣ Resultados Finais:")
        print(f"   🏆 Melhor modelo: {results.get('best_model', 'N/A')}")
        
        if results.get('best_model_metrics'):
            metrics = results['best_model_metrics']
            print(f"   📊 Acurácia: {metrics.get('accuracy', 0):.4f}")
            print(f"   📊 F1-Score: {metrics.get('f1_score', 0):.4f}")
            print(f"   📊 AUC-ROC: {metrics.get('roc_auc', 0):.4f}")
        
        print(f"\n📁 Arquivos gerados:")
        print("   - results_*.json (resultados detalhados)")
        print("   - REPORT_*.md (relatório executivo)")
        print("   - ml_pipeline.log (logs de execução)")
        
        print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na execução do pipeline: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_demo()
    if not success:
        print("\n❌ Demonstração falhou. Verifique os erros acima.")
        sys.exit(1)
    else:
        print("\n✅ Demonstração executada com sucesso!")

