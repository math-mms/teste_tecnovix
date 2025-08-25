# Pipeline de Machine Learning - Análise de Churn de Clientes

## 📋 Descrição

Este projeto implementa um pipeline completo de Machine Learning para previsão de churn de clientes, utilizando o dataset **Telco Customer Churn** do Kaggle. A solução foi desenvolvida seguindo os **princípios SOLID** de design de software, garantindo código modular, extensível e de fácil manutenção.

## 🎯 Objetivos

1. **Preparação de Dados**: Demonstração de habilidades em limpeza, transformação e engenharia de features
2. **Modelagem Preditiva**: Construção e avaliação de modelos de Machine Learning para previsão de churn
3. **Análise e Comunicação**: Interpretação de resultados e apresentação de insights de forma clara

## 🏗️ Arquitetura SOLID

### Princípios Aplicados:

- **S - Single Responsibility**: Cada classe tem uma única responsabilidade
- **O - Open/Closed**: Sistema aberto para extensão, fechado para modificação
- **L - Liskov Substitution**: Implementações podem ser substituídas sem quebrar o sistema
- **I - Interface Segregation**: Interfaces específicas para cada funcionalidade
- **D - Dependency Inversion**: Dependências de abstrações, não de implementações

### Estrutura do Projeto:

```
teste_tecnovix/
├── src/
│   ├── data/                    # Layer de dados
│   │   ├── interfaces.py        # Interfaces SOLID
│   │   ├── data_loader.py       # Carregamento de dados
│   │   ├── data_cleaner.py      # Limpeza de dados
│   │   └── feature_engineer.py  # Engenharia de features
│   ├── models/                  # Layer de modelos
│   │   ├── interfaces.py        # Interfaces SOLID
│   │   ├── base_model.py        # Modelo base abstrato
│   │   ├── concrete_models.py   # Implementações concretas
│   │   ├── model_trainer.py     # Treinamento de modelos
│   │   └── model_evaluator.py   # Avaliação de modelos
│   ├── pipeline/                # Pipeline principal
│   │   └── ml_pipeline.py       # Orquestração do pipeline
│   └── utils/                   # Utilitários
│       └── config.py            # Configuração centralizada
├── data/                        # Dataset
├── requirements.txt             # Dependências
├── main.py                      # Script principal
├── notebook.ipynb              # Notebook Jupyter
└── README.md                   # Documentação
```

## 🚀 Instalação e Execução

### 1. Pré-requisitos

- Python 3.8+
- Dataset Telco Customer Churn (disponível no Kaggle)

### 2. Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd teste_tecnovix

# Instale as dependências
pip install -r requirements.txt
```

### 3. Preparação dos Dados

Baixe o dataset do Kaggle e coloque-o em `data/telco_customer_churn.csv`:

```bash
# Crie o diretório de dados
mkdir data

# Baixe o dataset (substitua pelo caminho correto)
# wget https://www.kaggle.com/blastchar/telco-customer-churn/download -O data/telco_customer_churn.csv
```

### 4. Execução

#### Opção 1: Demonstração Completa (Recomendado)
```bash
python run_demo.py
```

#### Opção 2: Execução Manual
```bash
# 1. Baixar dataset
python download_dataset.py

# 2. Executar testes
python test_pipeline.py

# 3. Executar pipeline
python main.py
```

#### Opção 3: Via Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```

## 📊 Funcionalidades

### 1. Preparação de Dados (45 min)

- **Carregamento**: Leitura e validação do dataset
- **Limpeza**: Tratamento de valores ausentes e inconsistências
- **Engenharia de Features**: Criação de features derivadas e encoding

### 2. Modelagem Preditiva (90 min)

- **Algoritmos Implementados**:
  - Regressão Logística
  - Random Forest
  - Gradient Boosting
- **Avaliação**: Métricas de acurácia, precisão, recall, F1-Score e AUC-ROC

### 3. Análise e Comunicação (45 min)

- **Comparação de Modelos**: Análise de performance entre algoritmos
- **Importância de Features**: Identificação dos fatores mais relevantes
- **Relatório Executivo**: Geração automática de insights e recomendações

## 🔧 Componentes Principais

### Data Layer

```python
# Interfaces SOLID
class IDataLoader(ABC)
class IDataCleaner(ABC)
class IFeatureEngineer(ABC)

# Implementações
class TelcoDataLoader(IDataLoader)
class TelcoDataCleaner(IDataCleaner)
class TelcoFeatureEngineer(IFeatureEngineer)
```

### Model Layer

```python
# Interfaces SOLID
class IModel(ABC)
class IModelTrainer(ABC)
class IModelEvaluator(ABC)

# Implementações
class BaseModel(IModel, ABC)
class LogisticRegressionModel(BaseModel)
class RandomForestModel(BaseModel)
class GradientBoostingModel(BaseModel)
```

### Pipeline Layer

```python
class MLPipeline:
    """Orquestra todo o processo seguindo Dependency Inversion Principle"""
    
    def __init__(self, 
                 data_loader: IDataLoader,
                 data_cleaner: IDataCleaner,
                 feature_engineer: IFeatureEngineer,
                 model_trainer: IModelTrainer,
                 model_evaluator: IModelEvaluator)
```

## 📈 Resultados Esperados

### Métricas de Performance:
- **Acurácia**: > 80%
- **F1-Score**: > 0.75
- **AUC-ROC**: > 0.80

### Features Mais Importantes:
1. Tipo de contrato (month-to-month vs longo prazo)
2. Carga mensal
3. Tempo de serviço (tenure)
4. Serviços adicionais
5. Suporte técnico

### Insights de Negócio:
- Clientes com contratos month-to-month têm maior propensão ao churn
- Cargas mensais altas estão correlacionadas com churn
- Serviços adicionais reduzem a probabilidade de churn
- Suporte técnico é crucial para retenção

## 🎯 Estratégias de Retenção

1. **Foco em Contratos Longos**: Incentivar migração de contratos month-to-month
2. **Monitoramento de Cargas**: Acompanhar clientes com cargas mensais elevadas
3. **Serviços Adicionais**: Oferecer pacotes complementares
4. **Suporte Técnico**: Melhorar qualidade do atendimento
5. **Programas de Fidelidade**: Implementar benefícios para clientes seniores

## 📝 Entregáveis

- **Código**: Script Python (`main.py`) e Notebook (`notebook.ipynb`)
- **Relatório**: Arquivo Markdown com análise completa
- **Resultados**: Arquivo JSON com métricas e insights
- **Logs**: Arquivo de log detalhado da execução

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Machine Learning
- **NumPy**: Operações numéricas
- **Matplotlib/Seaborn**: Visualizações
- **ABC**: Implementação de interfaces SOLID

## 📋 Critérios de Avaliação

- ✅ **Corretude**: Solução funcional e completa
- ✅ **Qualidade do Código**: Princípios SOLID aplicados
- ✅ **Análise**: Insights profundos e relevantes
- ✅ **Comunicação**: Relatório claro e conciso
- ✅ **Eficiência**: Execução dentro do tempo limite (3 horas)

## 🤝 Contribuição

Este projeto foi desenvolvido como demonstração de habilidades em:
- Arquitetura de software SOLID
- Machine Learning
- Análise de dados
- Comunicação técnica

## 📄 Licença

Este projeto é de uso educacional e demonstrativo.

---

**Desenvolvido com princípios SOLID e foco em qualidade de código** 🚀
