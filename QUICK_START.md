# 🚀 Início Rápido - Pipeline de ML

## Execução em 3 Passos

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Demonstração Completa
```bash
python run_demo.py
```

### 3. Verificar Resultados
- 📊 **Relatório**: `REPORT_*.md`
- 📈 **Resultados**: `results_*.json`
- 📝 **Logs**: `ml_pipeline.log`

## 🎯 O que o Pipeline Faz

1. **📥 Carrega** dataset Telco Customer Churn
2. **🧹 Limpa** dados (valores ausentes, inconsistências)
3. **🔧 Cria** features derivadas (engenharia de features)
4. **🤖 Treina** 3 modelos (LogisticRegression, RandomForest, GradientBoosting)
5. **📊 Avalia** performance com métricas completas
6. **📋 Gera** relatório com insights e recomendações

## 🏗️ Arquitetura SOLID

- **S** - Single Responsibility: Cada classe tem uma função específica
- **O** - Open/Closed: Extensível sem modificar código existente
- **L** - Liskov Substitution: Implementações intercambiáveis
- **I** - Interface Segregation: Interfaces específicas
- **D** - Dependency Inversion: Dependências de abstrações

## 📈 Resultados Esperados

- **Acurácia**: > 80%
- **F1-Score**: > 0.75
- **AUC-ROC**: > 0.80
- **Tempo**: < 3 minutos

## 🔧 Troubleshooting

### Erro: "Dataset não encontrado"
```bash
python download_dataset.py
```

### Erro: "Módulo não encontrado"
```bash
pip install -r requirements.txt
```

### Erro: "Testes falharam"
Verifique se todas as dependências estão instaladas corretamente.

## 📞 Suporte

Para dúvidas ou problemas, consulte:
- `README.md` - Documentação completa
- `test_pipeline.py` - Testes detalhados
- `ml_pipeline.log` - Logs de execução

---

**Desenvolvido com princípios SOLID** 🎯

