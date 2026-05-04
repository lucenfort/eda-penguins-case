# EDA & Penguins Classification

<p align="center">
  <img src="./assets/banner.svg" alt="Project Banner" width="100%" />
</p>

<p align="left">
	<img src="https://img.shields.io/badge/Python-3.x-FFD700?style=for-the-badge&logo=python&logoColor=111111&labelColor=0B0B0B" alt="Python" />
	<img src="https://img.shields.io/badge/Seaborn-00FFF7?style=for-the-badge&logo=pandas&logoColor=111111&labelColor=0B0B0B" alt="Seaborn" />
	<img src="https://img.shields.io/badge/Scikit_Learn-9F00FF?style=for-the-badge&logo=scikit-learn&logoColor=111111&labelColor=0B0B0B" alt="Scikit-Learn" />
	<img src="https://img.shields.io/badge/Status-Finalizado-FF00FF?style=for-the-badge&logoColor=111111&labelColor=0B0B0B" alt="Status" />
</p>

Repositório técnico focado em Análise Exploratória de Dados (EDA) e classificação supervisionada multiclasse utilizando o dataset Palmer Penguins. O projeto demonstra um fluxo completo desde a limpeza de dados até a comparação de 8 modelos de Machine Learning.

## [>] SYS.NAVEGAÇÃO

[Objetivo](#-objetivo) • [Estrutura](#-estrutura-essencial) • [Execução](#-execução) • [Fluxo](#-fluxo-resumo-eda-e-ml) • [Resultados](#-resultados-principais)

---

## [~] OBJETIVO_SISTEMA

1. **Caracterização**: Avaliar qualidade e distribuição dos dados biométricos.
2. **Discriminação**: Analisar a separabilidade das espécies via medidas físicas.
3. **Modelagem**: Treinar e comparar 8 algoritmos de classificação (Ensembles, SVM, MLP).

## [=] ESTRUTURA_ESSENCIAL

```
eda-penguins-case/
├── assets/               # HUDs e Banner Cyberpunk
├── dataset/              # Dados originais e processados (Parquet)
├── docs/                 # Relatórios automáticos de EDA e ML
├── models/               # Modelos serializados (.pkl)
├── notebooks/            # Exploração interativa (Jupyter)
├── outputs/              # Gráficos e visualizações geradas
├── src/                  # Módulos de processamento e treinamento
├── main.py               # Orquestrador do pipeline completo
├── requirements.txt      # Dependências do sistema
└── README.md             # Documentação técnica
```

---

## [*] INSTALAÇÃO_E_EXECUÇÃO

### 1. Preparar Ambiente
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Executar Pipeline
```bash
python main.py
```

---

## [&] FLUXO_RESUMO_EDA_E_ML

```mermaid
flowchart TD
    A[Dataset Palmer Penguins CSV] --> B[EDA]
    B --> B1[Qualidade dos dados]
    B --> B2[Distribuições por ilha e espécie]
    B --> B3[Relações biométricas]
    B --> C[Dataset limpo: Parquet]
    C --> D[ML Pipeline]
    D --> D1[Split Estratificado]
    D --> D2[Treinamento 8 Modelos]
    D --> D3[Avaliação e Tuning]
    D --> E[Relatório Final e Modelos]
```

---

## [#] RESULTADOS_PRINCIPAIS

- **Qualidade**: Baixa taxa de ausências, tratadas via limpeza seletiva.
- **Biometria**: Separação clara entre espécies (Gentoo vs Adelie vs Chinstrap) através de massa corporal e comprimento de nadadeira.
- **Classificação**: Modelos atingiram acurácia de teste entre **98.51% e 100%**, com alta robustez em validação cruzada.

---


