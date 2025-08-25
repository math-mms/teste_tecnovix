# Inicio Rapido - Pipeline de ML (Python 2.7)

## ATENCAO: Python 2.7 e muito antigo!
**Recomendamos fortemente atualizar para Python 3.8+**

## Execucao em 3 Passos (Python 2.7)

### 1. Instalar Dependencias
```bash
pip install -r requirements_python2.txt
```

### 2. Executar Demonstracao Completa
```bash
python run_demo_python2.py
```

### 3. Verificar Resultados
- **Relatorio**: `REPORT_*.md`
- **Resultados**: `results_*.json`
- **Logs**: `ml_pipeline.log`

## O que o Pipeline Faz

1. **Carrega** dataset Telco Customer Churn
2. **Limpa** dados (valores ausentes, inconsistencias)
3. **Cria** features derivadas (engenharia de features)
4. **Treina** 3 modelos (LogisticRegression, RandomForest, GradientBoosting)
5. **Avalia** performance com metricas completas
6. **Gera** relatorio com insights e recomendacoes

## Arquitetura SOLID

- **S** - Single Responsibility: Cada classe tem uma funcao especifica
- **O** - Open/Closed: Extensivel sem modificar codigo existente
- **L** - Liskov Substitution: Implementacoes intercambiaveis
- **I** - Interface Segregation: Interfaces especificas
- **D** - Dependency Inversion: Dependencias de abstracoes

## Resultados Esperados

- **Acuracia**: > 80%
- **F1-Score**: > 0.75
- **AUC-ROC**: > 0.80
- **Tempo**: < 3 minutos

## Troubleshooting

### Erro: "Dataset nao encontrado"
```bash
python download_dataset_python2.py
```

### Erro: "Modulo nao encontrado"
```bash
pip install -r requirements_python2.txt
```

### Erro: "Testes falharam"
Verifique se todas as dependencias estao instaladas corretamente.

## IMPORTANTE: Atualizar Python

Para melhor performance e compatibilidade:

1. **Baixe Python 3.8+**: https://www.python.org/downloads/
2. **Instale as dependencias**: `pip install -r requirements.txt`
3. **Execute**: `python run_demo.py`

---

**Desenvolvido com principios SOLID**

