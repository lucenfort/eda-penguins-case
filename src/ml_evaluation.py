"""
Machine Learning Evaluation Module for Palmer Penguins Classification

Responsável pela avaliação completa de modelos:
- Cálculo de métricas (Accuracy, Precision, Recall, F1-Score)
- Confusion Matrix
- Cross-validation
- Comparação de modelos
- Visualizações de resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    auc
)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import clone
from pathlib import Path


ML_PALETTE = {
    "train": "#1D4ED8",
    "test": "#DC2626",
    "macro": "#D97706",
    "weighted": "#7C3AED",
    "cv": "#059669",
    "gap": "#B91C1C",
}


def _style_ml_axes(ax):
    """Aplica um estilo limpo e técnico aos gráficos de ML."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, axis='x', linestyle='--', alpha=0.18)
    ax.set_axisbelow(True)


class PenguinsMLEvaluator:
    """Classe para avaliação completa de modelos de classificação."""
    
    def __init__(self, class_names: np.ndarray):
        """
        Inicializa avaliador.
        
        Args:
            class_names: Nomes das classes (Adelie, Gentoo, Chinstrap)
        """
        self.class_names = class_names
        self.results = {}
        
    def evaluate_model(
        self,
        model_key: str,
        model_name: str,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_test_original: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Avalia um modelo de classificação completamente.
        
        Args:
            model_key: Chave única do modelo
            model_name: Nome legível do modelo
            model: Modelo treinado
            X_train: Features de treino
            X_test: Features de teste
            y_train: Target de treino (encoded)
            y_test: Target de teste (encoded)
            y_test_original: Target original (para referência)
            
        Returns:
            Dicionário com todas as métricas
        """
        print(f"\n{'='*60}")
        print(f"AVALIAÇÃO: {model_name}")
        print(f"{'='*60}")
        
        # ===== PREDIÇÕES =====
        y_pred = model.predict(X_test)
        
        # ===== MÉTRICAS BÁSICAS =====
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # ===== ACCURACY POR CLASSE =====
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ===== RELATÓRIO DE CLASSIFICAÇÃO =====
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )
        
        # ===== CONFUSION MATRIX =====
        cm = confusion_matrix(y_test, y_pred)
        
        # ===== CROSS-VALIDATION =====
        # CORRIGIDO: Usar clone do modelo para CV - não deve usar modelo já treinado!
        model_for_cv = clone(model)
        cv_scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='accuracy')
        
        # ===== TRAIN vs TEST =====
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = accuracy
        overfitting_gap = train_accuracy - test_accuracy
        
        # ===== PROBABILIDADES (se disponível) =====
        try:
            y_proba = model.predict_proba(X_test)
            has_proba = True
        except:
            y_proba = None
            has_proba = False
        
        # ===== IMPRESSÃO DE RESULTADOS =====
        print(f"\n📊 MÉTRICAS GERAIS:")
        print(f"   Acurácia Teste:     {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Acurácia Treino:    {train_accuracy:.4f}")
        print(f"   Overfitting Gap:    {overfitting_gap:.4f}")
        print(f"   F1-Score (macro):   {f1_macro:.4f}")
        print(f"   F1-Score (weighted):{f1_weighted:.4f}")
        
        print(f"\n📊 CROSS-VALIDATION (5-fold):")
        print(f"   Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   Média:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print(f"\n📊 MÉTRICAS POR CLASSE:")
        for i, class_name in enumerate(self.class_names):
            # Acessa pelo nome da classe no dicionário
            class_key = class_name
            if class_key in class_report:
                print(f"\n   {class_name}:")
                print(f"      Precision: {class_report[class_key]['precision']:.4f}")
                print(f"      Recall:    {class_report[class_key]['recall']:.4f}")
                print(f"      F1-Score:  {class_report[class_key]['f1-score']:.4f}")
                print(f"      Support:   {int(class_report[class_key]['support'])}")
        
        print(f"\n📊 CONFUSION MATRIX:")
        print(cm)
        
        # ===== ARMAZENA RESULTADOS =====
        result = {
            'model_key': model_key,
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'train_accuracy': train_accuracy,
            'overfitting_gap': overfitting_gap,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm,
            'class_report': class_report,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'has_proba': has_proba
        }
        
        self.results[model_key] = result
        
        return result
    
    def compare_all_models(self) -> pd.DataFrame:
        """
        Compara todos os modelos avaliados em um DataFrame.
        
        Returns:
            DataFrame com comparativo de performance
        """
        print("\n" + "="*60)
        print("COMPARAÇÃO DE TODOS OS MODELOS")
        print("="*60)
        
        comparison_data = []
        
        for model_key, result in self.results.items():
            comparison_data.append({
                'Modelo': result['model_name'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'Train Acc': f"{result['train_accuracy']:.4f}",
                'Overfitting': f"{result['overfitting_gap']:.4f}",
                'F1 (macro)': f"{result['f1_macro']:.4f}",
                'F1 (weighted)': f"{result['f1_weighted']:.4f}",
                'CV Mean': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}",
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        print("\n📊 TABELA COMPARATIVA:\n")
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, Dict[str, Any]]:
        """
        Retorna o melhor modelo baseado em uma métrica.
        
        Args:
            metric: Métrica para ranking ('accuracy', 'f1_macro', 'f1_weighted', 'cv_mean')
            
        Returns:
            Tupla (model_key, result_dict)
        """
        metric_map = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
            'cv_mean': 'cv_mean'
        }
        
        metric_field = metric_map.get(metric, 'accuracy')
        
        best_key = max(self.results.keys(), key=lambda k: self.results[k][metric_field])
        best_result = self.results[best_key]
        
        print(f"\n🏆 MELHOR MODELO ({metric}):")
        print(f"   Nome: {best_result['model_name']}")
        print(f"   {metric}: {best_result[metric_field]:.4f}")
        
        return best_key, best_result
    
    def plot_comparison(self, output_dir: str = "outputs/graficos"):
        """
        Gera gráficos comparativos de todos os modelos.
        
        Args:
            output_dir: Diretório para salvar gráficos
        """
        print(f"\n📈 Gerando gráficos comparativos...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        
        # ===== GRÁFICO 1: ACCURACY COMPARISON =====
        fig, ax = plt.subplots(figsize=(13, 6.8))
        
        models = [r['model_name'] for r in self.results.values()]
        accuracies = [r['accuracy'] for r in self.results.values()]
        train_accs = [r['train_accuracy'] for r in self.results.values()]
        y = np.arange(len(models))
        colors = sns.color_palette("crest", n_colors=len(models))

        x_min = max(0.0, min(min(accuracies), min(train_accs)) - 0.04)
        x_max = min(1.015, max(max(accuracies), max(train_accs)) + 0.015)

        for idx, (color, model, test_acc, train_acc) in enumerate(zip(colors, models, accuracies, train_accs)):
            ax.hlines(y=idx, xmin=test_acc, xmax=train_acc, color=color, linewidth=2.2, alpha=0.9)
            ax.scatter(test_acc, idx, color=ML_PALETTE["test"], s=70, zorder=3, label='Teste' if idx == 0 else None)
            ax.scatter(train_acc, idx, color=ML_PALETTE["train"], s=70, zorder=3, label='Treino' if idx == 0 else None)
            ax.text(max(test_acc - 0.004, x_min + 0.001), idx + 0.12, f'{test_acc:.3f}',
                ha='right', va='center', fontsize=8)
            train_label_x = train_acc + 0.003
            train_label_ha = 'left'
            if train_label_x > x_max - 0.001:
                train_label_x = train_acc - 0.003
                train_label_ha = 'right'
            ax.text(train_label_x, idx - 0.12, f'{train_acc:.3f}',
                ha=train_label_ha, va='center', fontsize=8)

        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Acurácia por modelo: treino vs teste', fontsize=14, fontweight='bold', pad=14)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlim(x_min, x_max)
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(0.995, 1.14),
            ncol=2,
            frameon=True,
            title='Conjunto',
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.0,
        )
        _style_ml_axes(ax)
        
        plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
        plt.savefig(f"{output_dir}/ml_10_comparacao_accuracy.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: ml_10_comparacao_accuracy.png")
        plt.close()
        
        # ===== GRÁFICO 2: F1-SCORE COMPARISON =====
        fig, ax = plt.subplots(figsize=(13, 6.8))
        
        f1_macros = [r['f1_macro'] for r in self.results.values()]
        f1_weighteds = [r['f1_weighted'] for r in self.results.values()]
        y = np.arange(len(models))
        colors = sns.color_palette("mako", n_colors=len(models))

        x_min = max(0.0, min(min(f1_macros), min(f1_weighteds)) - 0.04)
        x_max = min(1.015, max(max(f1_macros), max(f1_weighteds)) + 0.015)

        for idx, (color, model, f1_macro, f1_weighted) in enumerate(zip(colors, models, f1_macros, f1_weighteds)):
            ax.hlines(y=idx, xmin=f1_macro, xmax=f1_weighted, color=color, linewidth=2.2, alpha=0.9)
            ax.scatter(f1_macro, idx, color=ML_PALETTE["macro"], s=70, zorder=3, label='Macro' if idx == 0 else None)
            ax.scatter(f1_weighted, idx, color=ML_PALETTE["weighted"], s=70, zorder=3, label='Weighted' if idx == 0 else None)
            ax.text(max(f1_macro - 0.004, x_min + 0.001), idx + 0.12, f'{f1_macro:.3f}',
                ha='right', va='center', fontsize=8)
            weighted_label_x = f1_weighted + 0.003
            weighted_label_ha = 'left'
            if weighted_label_x > x_max - 0.001:
                weighted_label_x = f1_weighted - 0.003
                weighted_label_ha = 'right'
            ax.text(weighted_label_x, idx - 0.12, f'{f1_weighted:.3f}',
                ha=weighted_label_ha, va='center', fontsize=8)

        ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('F1 por modelo: macro vs weighted', fontsize=14, fontweight='bold', pad=14)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlim(x_min, x_max)
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(0.995, 1.14),
            ncol=2,
            frameon=True,
            title='Métrica',
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.0,
        )
        _style_ml_axes(ax)
        
        plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
        plt.savefig(f"{output_dir}/ml_11_comparacao_f1score.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: ml_11_comparacao_f1score.png")
        plt.close()
        
        # ===== GRÁFICO 3: CROSS-VALIDATION =====
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cv_means = [r['cv_mean'] for r in self.results.values()]
        cv_stds = [r['cv_std'] for r in self.results.values()]
        x = np.arange(len(models))
        
        ax.errorbar(x, cv_means, yerr=cv_stds, fmt='o-', color='#06A77D',
                    ecolor='black', capsize=5, capthick=2, linewidth=2.5,
                    markersize=7, label='Cross-Validation Mean ± Std')
        
        ax.set_ylabel('Accuracy (CV)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-validation 5-fold por modelo', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        y_min = max(0.0, min(m - s for m, s in zip(cv_means, cv_stds)) - 0.01)
        y_max = min(1.01, max(m + s for m, s in zip(cv_means, cv_stds)) + 0.015)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True)
        _style_ml_axes(ax)
        
        for xi, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            ax.text(xi, mean + std + 0.003, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ml_12_cross_validation.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: ml_12_cross_validation.png")
        plt.close()
        
        # ===== GRÁFICO 4: OVERFITTING GAP =====
        fig, ax = plt.subplots(figsize=(12, 6))
        
        overfitting_gaps = [r['overfitting_gap'] for r in self.results.values()]
        x = np.arange(len(models))
        colors = ['#B91C1C' if gap > 0.10 else '#F97316' if gap > 0.05 else '#059669' 
                  for gap in overfitting_gaps]

        bars = ax.barh(x, overfitting_gaps, color=colors, alpha=0.9, edgecolor='none', height=0.65)
        
        ax.axvline(x=0.10, color='#7F1D1D', linestyle='--', linewidth=1.8, alpha=0.7, label='Alto (0.10)')
        ax.axvline(x=0.05, color='#9A3412', linestyle='--', linewidth=1.8, alpha=0.7, label='Médio (0.05)')
        
        ax.set_xlabel('Overfitting gap (train - test)', fontsize=12, fontweight='bold')
        ax.set_title('Gap de overfitting por modelo', fontsize=14, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels(models)
        ax.set_xlim(0, max(max(overfitting_gaps) + 0.01, 0.105))
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='x', linestyle='--', alpha=0.18)
        
        for bar, gap in zip(bars, overfitting_gaps):
            ax.text(gap + 0.005, bar.get_y() + bar.get_height()/2.,
                   f'{gap:.4f}', ha='left', va='center', fontsize=8)

        _style_ml_axes(ax)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ml_13_overfitting_analysis.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: ml_13_overfitting_analysis.png")
        plt.close()
        
        print(f"\n✅ Todos os gráficos comparativos gerados!")
    
    def plot_confusion_matrices(self, output_dir: str = "outputs/graficos"):
        """
        Gera confusion matrices para cada modelo.
        
        Args:
            output_dir: Diretório para salvar gráficos
        """
        print(f"\n📊 Gerando confusion matrices...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="white", context="talk")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        # Reserva uma margem fixa à direita para a barra de escala externa.
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.07, top=0.90, wspace=0.35, hspace=0.35)
        axes = axes.flatten()
        vmin = 0.0
        vmax = 100.0
        
        for idx, (model_key, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            # Normaliza para porcentagem
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx],
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                square=True,
                annot_kws={'size': 9}
            )
            
            axes[idx].set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.3f}",
                              fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True label', fontsize=10)
            axes[idx].set_xlabel('Predicted label', fontsize=10)
            axes[idx].tick_params(labelsize=9)

        # Remove subplots vazios
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        if n_models > 0:
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='Blues')
            sm.set_array([])
            # Eixo dedicado da colorbar fora da grade principal para evitar sobreposição.
            cax = fig.add_axes([0.90, 0.16, 0.018, 0.68])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Percent (%)', rotation=270, labelpad=15)
        
        plt.suptitle('Matrizes de Confusão: Todos os Modelos', fontsize=15, fontweight='bold', y=0.965)
        plt.savefig(f"{output_dir}/ml_14_confusion_matrices.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: ml_14_confusion_matrices.png")
        plt.close()
