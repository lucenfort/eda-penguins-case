# EDA Pinguins — Análise Exploratória para ONG de Resgate


## Visão Geral

Projeto de análise exploratória para suportar operações da ONG especializada em acolhimento temporário de pinguins migratórios. O pipeline identifica padrões críticos em origem geográfica, composição de espécies, características corporais e dimorfismo sexual.

### Perguntas Respondidas

| # | Pergunta | Resposta |
|---|----------|---------|
| 1 | Dados faltantes? | 3.20% (11 casos - Respondida) |
| 2 | Maioria de pinguins? | Biscoe (48.84%) |
| 3 | Espécie principal? | Adelie (44.19%) |
| 4 | Medidas × espécie? | SIM, forte correlação |
| 5 | Medidas × sexo? | SIM, +11-20% dimorfismo |

---

## Arquitetura da Solução

```
Ingestão (DuckDB)
    ↓
Transformação (Polars)
    ↓
Análises (Agregações, Estatísticas)
    ↓
Visualização (Seaborn + Matplotlib - 300 DPI)
    ↓
Exportação (Parquet)
```

---

## Estrutura do Projeto

```
eda-penguins-case/
├── dataset/
│   ├── penguins.csv                    # Dados originais
│   └── penguins_clean.parquet          # Base processada
├── src/
│   ├── data_loader.py                  # Ingestão DuckDB
│   ├── analysis.py                     # Análises
│   ├── visualization.py                # Gráficos
│   └── penguins_analysis.py            # Orquestrador
├── outputs/graficos/                   # 9 gráficos PNG (300 DPI)
├── docs/
│   └── analise_resultados.md           # Relatório completo
├── main.py                             # Execução
└── requirements.txt                    # Dependências
```

---

## Como Executar

```bash
# 1. Criar ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar
python main.py                  # Gera 9 gráficos e parquet
jupyter notebook eda_penguins.ipynb
```

---

## Tecnologias

- **DuckDB:** Ingestão SQL nativa
- **Polars:** Transformação (3-10x mais rápido, sem Pandas)
- **Seaborn + Matplotlib:** Visualizações (300 DPI)
- **Jupyter:** Notebooks interativos

---

## Documentação

- **Relatório Completo:** [docs/analise_resultados.md](docs/analise_resultados.md) - Análises detalhadas, gráficos, recomendações operacionais
- **Notebook:** [eda_penguins.ipynb](eda_penguins.ipynb) - Versão interativa
- **Dados:** [dataset/penguins.csv](dataset/penguins.csv) | [dataset/penguins_clean.parquet](dataset/penguins_clean.parquet)

