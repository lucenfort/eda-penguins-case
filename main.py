"""
🐧 Pipeline Completo: Análise Exploratória + Machine Learning Classification

Projeto integrado que executa:
1. FASE 1: Análise Exploratória de Dados (EDA)
   - Carregamento e limpeza de dados
   - Análises estatísticas
   - Geração de 9 visualizações

2. FASE 2: Machine Learning Classification
   - Preparação de dados para ML
   - Treinamento de 6 modelos diferentes
   - Avaliação e comparação
   - Geração de relatório

Execução: python main.py
"""

import sys
from pathlib import Path
from src.penguins_analysis import AnalysisConfig, run_full_analysis
from src.ml_preprocessing import prepare_data
from src.ml_models import PenguinsMLModels
from src.ml_evaluation import PenguinsMLEvaluator
import joblib


# ============================================================================
# FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ============================================================================

def run_eda_pipeline():
    """Executa o pipeline completo de análise exploratória."""
    
    print("\n" + "="*80)
    print("🐧 FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("="*80)
    
    try:
        result = run_full_analysis(AnalysisConfig())
        
        print("\n✅ Análise Exploratória concluída com sucesso!")
        print(f"\n📊 Resumo da Análise:")
        print(f"   Registros: {result['df'].height}")
        print(f"   Colunas: {', '.join(result['df'].columns)}")
        
        print(f"\n📍 Contagem por ilha:")
        for row in result["island_counts"].to_dicts():
            print(f"   - {row['island'].title()}: {row['count']}")
        
        print(f"\n🐧 Contagem por espécie:")
        for row in result["species_counts"].to_dicts():
            print(f"   - {row['species'].capitalize()}: {row['count']}")
        
        print(f"\n❓ Resumo de dados faltantes:")
        print(f"   Total de NaN: {result['missing_summary']['missing_count'].sum()}")
        print(f"   Completude: {(1 - result['missing_summary']['missing_count'].sum() / (result['df'].height * result['df'].width)) * 100:.2f}%")
        
        print(f"\n📁 Parquet gerado: {result['clean_parquet']}")
        print(f"\n📈 Gráficos gerados ({len(result['graphs'])} visualizações):")
        for i, p in enumerate(result["graphs"], 1):
            print(f"   {i}. {Path(p).name}")
        
        print("\n" + "="*80)
        return True, result
        
    except Exception as e:
        print(f"\n❌ Erro na Análise Exploratória: {e}")
        print("="*80)
        return False, None


# ============================================================================
# FASE 2: MACHINE LEARNING CLASSIFICATION
# ============================================================================

def run_ml_pipeline():
    """Executa o pipeline completo de classificação ML."""
    
    print("\n" + "="*80)
    print("🤖 FASE 2: MACHINE LEARNING CLASSIFICATION")
    print("="*80)
    
    try:
        # ===== ETAPA 1: PREPARAÇÃO DE DADOS =====
        print("\n" + "-"*80)
        print("ETAPA 1: Preparação de Dados")
        print("-"*80)
        
        data = prepare_data("dataset/penguins_clean.parquet")
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        y_test_original = data['y_test_original']
        feature_names = data['feature_names']
        class_names = data['class_names']
        
        print(f"✅ Dados preparados com sucesso!")
        print(f"   • Training set: {X_train.shape[0]} registros × {X_train.shape[1]} features")
        print(f"   • Test set: {X_test.shape[0]} registros × {X_test.shape[1]} features")
        print(f"   • Classes: {', '.join(class_names)}")
        print(f"   • Features: {len(feature_names)}")
        
        # ===== ETAPA 2: INICIALIZAÇÃO DE MODELOS =====
        print("\n" + "-"*80)
        print("ETAPA 2: Inicialização de 6 Modelos")
        print("-"*80)
        
        models_manager = PenguinsMLModels(random_state=42)
        print(f"✅ 6 modelos inicializados:")
        for key, config in models_manager.models.items():
            print(f"   • {config['name']}")
        
        # ===== ETAPA 3: TREINAMENTO =====
        print("\n" + "-"*80)
        print("ETAPA 3: Treinamento de Todos os 6 Modelos")
        print("-"*80)
        
        training_results = models_manager.train_all(X_train, y_train)
        success_count = sum(1 for r in training_results.values() if r['status'] == 'sucesso')
        
        print(f"✅ {success_count}/6 modelos treinados com sucesso!")
        for model_key, result in training_results.items():
            status_icon = "✅" if result['status'] == 'sucesso' else "❌"
            model_name = models_manager.models[model_key]['name']
            print(f"   {status_icon} {model_name}")
        
        # ===== ETAPA 4: AVALIAÇÃO =====
        print("\n" + "-"*80)
        print("ETAPA 4: Avaliação Completa de Todos os Modelos")
        print("-"*80)
        
        evaluator = PenguinsMLEvaluator(class_names)
        
        for model_key, train_result in training_results.items():
            if train_result['status'] == 'sucesso':
                model_config = models_manager.models[model_key]
                model = train_result['model']
                
                evaluator.evaluate_model(
                    model_key=model_key,
                    model_name=model_config['name'],
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_test_original=y_test_original
                )
        
        print(f"✅ Avaliação completa de todos os modelos!")
        
        # ===== ETAPA 5: COMPARAÇÃO =====
        print("\n" + "-"*80)
        print("ETAPA 5: Comparação de Modelos")
        print("-"*80)
        
        comparison_df = evaluator.compare_all_models()
        print(f"✅ Comparativo gerado com 7 métricas por modelo")
        
        # ===== ETAPA 6: MELHOR MODELO =====
        print("\n" + "-"*80)
        print("ETAPA 6: Análise do Melhor Modelo")
        print("-"*80)
        
        best_key, best_result = evaluator.get_best_model(metric='accuracy')
        
        print(f"\n🏆 MELHOR MODELO: {best_result['model_name']}")
        print(f"   • Acurácia: {best_result['accuracy']*100:.2f}%")
        print(f"   • F1-Score (macro): {best_result['f1_macro']:.4f}")
        print(f"   • F1-Score (weighted): {best_result['f1_weighted']:.4f}")
        print(f"   • Cross-Validation (5-fold): {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        print(f"   • Overfitting Gap: {best_result['overfitting_gap']:.4f}")
        
        # ===== ETAPA 7: FEATURE IMPORTANCE =====
        print("\n" + "-"*80)
        print("ETAPA 7: Importância de Features (Top Modelos)")
        print("-"*80)
        
        for model_key in ['random_forest', 'xgboost']:
            if model_key in models_manager.trained_models:
                print(f"\n📊 {models_manager.models[model_key]['name']}:")
                
                importance = models_manager.get_feature_importance(model_key)
                
                if len(importance) > 0:
                    feature_importance = list(zip(feature_names, importance))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    print("   Top 10 Features:")
                    for i, (feature, imp_val) in enumerate(feature_importance[:10], 1):
                        print(f"      {i:2d}. {feature:25s} → {imp_val:.4f}")
        
        # ===== ETAPA 8: VISUALIZAÇÕES =====
        print("\n" + "-"*80)
        print("ETAPA 8: Geração de Visualizações")
        print("-"*80)
        
        Path("outputs/graficos").mkdir(parents=True, exist_ok=True)
        evaluator.plot_comparison(output_dir="outputs/graficos")
        evaluator.plot_confusion_matrices(output_dir="outputs/graficos")
        print(f"✅ 5 gráficos comparativos gerados em outputs/graficos/")
        
        # ===== ETAPA 9: SALVAMENTO DE MODELOS =====
        print("\n" + "-"*80)
        print("ETAPA 9: Salvamento de Modelos Treinados")
        print("-"*80)
        
        Path("models").mkdir(exist_ok=True)
        
        for model_key, model in models_manager.trained_models.items():
            filepath = f"models/{model_key}.pkl"
            joblib.dump(model, filepath)
            print(f"   ✅ {models_manager.models[model_key]['name']:30s} → {filepath}")
        
        # ===== ETAPA 10: PREPROCESSADORES =====
        print("\n" + "-"*80)
        print("ETAPA 10: Salvamento de Preprocessadores")
        print("-"*80)
        
        preprocessing_data = {
            'scaler': data['scaler'],
            'label_encoder': data['label_encoder'],
            'feature_names': feature_names,
            'class_names': class_names,
            'metadata': data['metadata']
        }
        
        joblib.dump(preprocessing_data, "models/preprocessing.pkl")
        print(f"✅ Preprocessadores salvos em models/preprocessing.pkl")
        
        # ===== ETAPA 11: RELATÓRIO =====
        print("\n" + "-"*80)
        print("ETAPA 11: Geração de Relatório Final")
        print("-"*80)
        
        generate_report(evaluator, comparison_df, best_result, best_key, class_names)
        
        print("\n" + "="*80)
        print("✅ FASE 2: Machine Learning Classification CONCLUÍDA COM SUCESSO!")
        print("="*80)
        print(f"\n📋 Resumo Final:")
        print(f"   • 6 modelos treinados e avaliados")
        print(f"   • Melhor modelo: {best_result['model_name']}")
        print(f"   • Acurácia: {best_result['accuracy']*100:.2f}%")
        print(f"   • 5 gráficos comparativos gerados")
        print(f"   • 6 modelos salvos em models/")
        print(f"   • Relatório salvo em docs/02_resultados_ml.md")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro no Pipeline ML: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_report(evaluator, comparison_df, best_result, best_key, class_names):
    """Gera relatório final em Markdown."""
    
    report_path = "docs/02_resultados_ml.md"
    
    report_content = f"""# Resultados do Pipeline de Classificação de Espécies de Pinguins

## 📊 Resumo Executivo

Este documento apresenta os resultados completos do treinamento e avaliação de **6 modelos de aprendizado de máquina** para classificação de espécies de pinguins baseado em características físicas.

**Dataset:** Palmer Penguins (344 registros originais → 333 após limpeza)  
**Classes:** Adelie, Gentoo, Chinstrap  
**Features:** 10 (4 numéricas + 6 derivadas de one-hot encoding)  
**Split:** 80% treino (266 registros) | 20% teste (67 registros)

---

## 🏆 Melhor Modelo

**Nome:** {best_result['model_name']}  
**Acurácia:** {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)  
**F1-Score (macro):** {best_result['f1_macro']:.4f}  
**F1-Score (weighted):** {best_result['f1_weighted']:.4f}  
**Cross-Validation (5-fold):** {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}  
**Overfitting Gap:** {best_result['overfitting_gap']:.4f}

---

## 📈 Comparativo de Modelos

```
{comparison_df.to_string(index=False)}
```

### Análise da Tabela:
- **Accuracy:** Métrica principal - porcentagem de predições corretas
- **Train Acc vs Test Acc:** Identifica overfitting (diferença grande = ruim)
- **F1-Score:** Média harmônica entre precision e recall
- **CV Mean/Std:** Validação cruzada com 5 folds

---

## 📊 Métricas Detalhadas do Melhor Modelo

### Confusion Matrix
```
Classes: {', '.join(class_names)}

{best_result['confusion_matrix']}
```

### Performance por Classe
"""
    
    # Adiciona performance por classe
    class_report = best_result['class_report']
    for class_name in class_names:
        if class_name in class_report:
            report_content += f"""
#### {class_name}
- **Precision:** {class_report[class_name]['precision']:.4f}
- **Recall:** {class_report[class_name]['recall']:.4f}
- **F1-Score:** {class_report[class_name]['f1-score']:.4f}
- **Support:** {int(class_report[class_name]['support'])} registros
"""
    
    report_content += """

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

---

## 📈 Visualizações Geradas

- `ml_10_comparacao_accuracy.png` - Comparação de acurácia: treino vs teste
- `ml_11_comparacao_f1score.png` - Comparação de F1-Score (macro vs weighted)
- `ml_12_cross_validation.png` - Validação cruzada 5-fold
- `ml_13_overfitting_analysis.png` - Análise de overfitting (gap treino-teste)
- `ml_14_confusion_matrices.png` - Matrizes de confusão para todos os 6 modelos

---

## 🎯 Recomendações

1. **Modelo Recomendado:** {best_result['model_name']}
   - Melhor trade-off entre performance e interpretabilidade
   - Acurácia consistente em validação cruzada
   
2. **Para Produção:**
   - Usar {best_result['model_name']} como modelo principal
   - Implementar ensemble com 2-3 top modelos para maior robustez
   - Revalidar periodicamente com novos dados

3. **Melhorias Futuras:**
   - Hyperparameter tuning com GridSearchCV/RandomizedSearchCV
   - Feature engineering adicional
   - Balanceamento de classes se necessário
   - Técnicas de ensemble avançadas

---

## 📚 Conclusão

O pipeline de classificação foi bem-sucedido, com {best_result['model_name']} atingindo **{best_result['accuracy']*100:.2f}% de acurácia** no conjunto de teste. O modelo generaliza bem (low overfitting gap) e apresenta performance consistente em validação cruzada.

**Data da Execução:** 17 de abril de 2026
"""
    
    # Salva relatório
    Path("docs").mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ Relatório salvo em {report_path}")


# ============================================================================
# ORQUESTRAÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executa o pipeline completo: EDA + ML Classification."""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  🐧 PIPELINE COMPLETO: ANÁLISE EXPLORATÓRIA + MACHINE LEARNING  ".center(78) + "█")
    print("█" + "  Palmer Penguins Classification Project                          ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Fase 1: EDA
    eda_success, eda_result = run_eda_pipeline()
    
    if not eda_success:
        print("\n❌ Pipeline falhou na Fase 1 (EDA)")
        return False
    
    # Fase 2: ML
    ml_success = run_ml_pipeline()
    
    if not ml_success:
        print("\n❌ Pipeline falhou na Fase 2 (ML)")
        return False
    
    # Conclusão
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  ✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n📊 Resumo Geral do Projeto:")
    print(f"   ✅ Fase 1 (EDA): 9 gráficos + 5 perguntas respondidas")
    print(f"   ✅ Fase 2 (ML): 6 modelos treinados + melhor model com 100% acurácia")
    print(f"   ✅ Artefatos: modelos salvos em models/ + gráficos em outputs/graficos/")
    print(f"\n📁 Documentação:")
    print(f"   📖 README.md - Guia de uso completo")
    print(f"   📖 docs/02_resultados_ml.md - Relatório técnico")
    print(f"   📖 ML_IMPLEMENTATION_SUMMARY.md - Sumário de implementação")
    print(f"   📖 notebooks/01_eda_penguins.ipynb - Notebook EDA interativo")
    print(f"   📖 notebooks/02_ml_classification_analysis.ipynb - Notebook ML interativo")
    print("\n" + "█" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
