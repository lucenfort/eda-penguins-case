# Resultados do Pipeline de Classificação de Espécies de Pinguins

## 📊 Resumo Executivo

Este documento apresenta os resultados completos do treinamento e avaliação de **8 modelos de aprendizado de máquina** para classificação de espécies de pinguins baseado em características físicas.

**Dataset:** Palmer Penguins (344 registros originais → 333 após limpeza)  
**Classes:** Adelie, Gentoo, Chinstrap  
**Features:** 10 (4 numéricas + 6 derivadas de one-hot encoding)  
**Split:** 80% treino (266 registros) | 20% teste (67 registros)

---

## 📈 Comparativo de Modelos

```
              Modelo Accuracy Train Acc Overfitting F1 (macro) F1 (weighted) CV Mean CV Std
       Random Forest   1.0000    1.0000      0.0000     1.0000        1.0000  0.9943 0.0070
             XGBoost   1.0000    1.0000      0.0000     1.0000        1.0000  0.9971 0.0057
           SVM (RBF)   0.9851    0.9972      0.0121     0.9827        0.9852  0.9971 0.0057
 K-Nearest Neighbors   1.0000    0.9943     -0.0057     1.0000        1.0000  0.9915 0.0070
 Logistic Regression   1.0000    1.0000      0.0000     1.0000        1.0000  1.0000 0.0000
Neural Network (MLP)   0.9851    1.0000      0.0149     0.9827        0.9852  0.9943 0.0070
Soft Voting Ensemble   1.0000    1.0000      0.0000     1.0000        1.0000  0.9971 0.0057
   Stacking Ensemble   1.0000    1.0000      0.0000     1.0000        1.0000  1.0000 0.0000
```

### Análise da Tabela:
- **Accuracy:** métrica principal, porcentagem de predições corretas.
- **Train Acc vs Test Acc:** indicador direto de overfitting.
- **F1-Score:** equilíbrio entre precision e recall.
- **CV Mean/Std:** estabilidade em validação cruzada 5-fold.

---

## 📊 Métricas Detalhadas do Melhor Modelo

### Confusion Matrix
```
Classes: adelie, chinstrap, gentoo

[[29  0  0]
 [ 0 14  0]
 [ 0  0 24]]
```

### Performance por Classe

#### adelie
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **Support:** 29 registros

#### chinstrap
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **Support:** 14 registros

#### gentoo
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **Support:** 24 registros


---

## 🔄 Modelos Avaliados

### 1. Logistic Regression
- **Tipo:** Classificador linear
- **Vantagens:** Interpretável, rápido, boa calibração de probabilidades
- **Desvantagens:** Pode underfitting em dados não-lineares

### 2. K-Nearest Neighbors (k=5)
- **Tipo:** Algoritmo baseado em instâncias
- **Vantagens:** Simples, sem treinamento necessário
- **Desvantagens:** Lento em predição, sensível ao scaling

### 3. Random Forest
- **Tipo:** Ensemble de árvores de decisão
- **Vantagens:** Robusto, feature importance, reduz overfitting
- **Desvantagens:** Menos interpretável que regressão linear

### 4. XGBoost
- **Tipo:** Gradient boosting
- **Vantagens:** Estado-da-arte, altíssima performance
- **Desvantagens:** Mais complexo de tunar

### 5. Support Vector Machine (RBF kernel)
- **Tipo:** Classificador de margem máxima
- **Vantagens:** Poderoso em multiclasse, bom em altas dimensões
- **Desvantagens:** Sensível ao scaling, mais lento

### 6. Neural Network (MLP)
- **Tipo:** Rede neural com backpropagation
- **Vantagens:** Altamente flexível, captura padrões complexos
- **Desvantagens:** Requer mais tuning, risco de overfitting

### 7. Soft Voting Ensemble
- **Tipo:** Ensemble por votação probabilística
- **Vantagens:** Combina pontos fortes de modelos heterogêneos
- **Desvantagens:** Pode herdar vieses dos modelos base

### 8. Stacking Ensemble
- **Tipo:** Ensemble em camadas (meta-learning)
- **Vantagens:** Captura complementaridade entre modelos base
- **Desvantagens:** Mais complexo e com maior custo de treino

---

## 📈 Visualizações Geradas

- `ml_10_comparacao_accuracy.png` - Comparação de acurácia: treino vs teste
- `ml_11_comparacao_f1score.png` - Comparação de F1-Score (macro vs weighted)
- `ml_12_cross_validation.png` - Validação cruzada 5-fold
- `ml_13_overfitting_analysis.png` - Análise de overfitting (gap treino-teste)
- `ml_14_confusion_matrices.png` - Matrizes de confusão para todos os 8 modelos
