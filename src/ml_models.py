"""
Machine Learning Models Module for Palmer Penguins Classification

Implementa modelos de classificação com:
- Tuning de hiperparâmetros (RandomizedSearchCV + GridSearchCV)
- Ensemble avançado (Voting e Stacking)
- Estratégias de balanceamento via class_weight quando aplicável
"""

from __future__ import annotations

import numpy as np
import joblib
from typing import Dict, List, Any, Tuple

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


class PenguinsMLModels:
    """Gerenciador de múltiplos modelos de classificação com tuning."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Dict[str, Any]] = {}
        self.trained_models: Dict[str, Any] = {}
        self.tuning_results: Dict[str, Dict[str, Any]] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Define configurações dos modelos e estratégias de tuning."""
        print("\n" + "=" * 60)
        print("INICIALIZAÇÃO DE MODELOS COM TUNING E ENSEMBLES")
        print("=" * 60)

        self.models = {
            'random_forest': {
                'name': 'Random Forest',
                'description': 'Ensemble de árvores com tuning randômico',
                'factory': lambda class_weight: RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight=class_weight,
                ),
                'supports_class_weight': True,
                'tuning': {
                    'strategy': 'randomized',
                    'n_iter': 12,
                    'param_distributions': {
                        'n_estimators': [120, 180, 240, 320],
                        'max_depth': [None, 6, 10, 14, 18],
                        'min_samples_split': [2, 4, 6, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None],
                    },
                },
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': 'Gradient boosting com tuning randômico',
                'factory': lambda class_weight: XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='mlogloss',
                    verbosity=0,
                ),
                'supports_class_weight': False,
                'tuning': {
                    'strategy': 'randomized',
                    'n_iter': 12,
                    'param_distributions': {
                        'n_estimators': [120, 180, 240, 320],
                        'max_depth': [3, 4, 5, 6, 8],
                        'learning_rate': [0.03, 0.05, 0.08, 0.1, 0.15],
                        'subsample': [0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                        'min_child_weight': [1, 3, 5],
                    },
                },
            },
            'svm': {
                'name': 'SVM (RBF)',
                'description': 'Máquina de vetores suporte com busca em grade',
                'factory': lambda class_weight: SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.random_state,
                    class_weight=class_weight,
                ),
                'supports_class_weight': True,
                'tuning': {
                    'strategy': 'grid',
                    'param_grid': {
                        'C': [0.5, 1, 5, 10, 20],
                        'gamma': ['scale', 0.05, 0.1, 0.2],
                    },
                },
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'description': 'KNN com busca em grade',
                'factory': lambda class_weight: KNeighborsClassifier(n_jobs=-1),
                'supports_class_weight': False,
                'tuning': {
                    'strategy': 'grid',
                    'param_grid': {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan', 'minkowski'],
                    },
                },
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'description': 'Baseline linear com busca em grade',
                'factory': lambda class_weight: LogisticRegression(
                    max_iter=1500,
                    random_state=self.random_state,
                    class_weight=class_weight,
                ),
                'supports_class_weight': True,
                'tuning': {
                    'strategy': 'grid',
                    'param_grid': {
                        'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                        'solver': ['lbfgs', 'newton-cg'],
                    },
                },
            },
            'neural_network': {
                'name': 'Neural Network (MLP)',
                'description': 'MLP com tuning randômico controlado',
                'factory': lambda class_weight: MLPClassifier(
                    random_state=self.random_state,
                    max_iter=1200,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=40,
                ),
                'supports_class_weight': False,
                'tuning': {
                    'strategy': 'randomized',
                    'n_iter': 10,
                    'param_distributions': {
                        'hidden_layer_sizes': [(64, 32), (96, 48), (128, 64), (64, 32, 16)],
                        'learning_rate_init': [0.0005, 0.001, 0.003, 0.005],
                        'alpha': [0.0001, 0.001, 0.005, 0.01],
                        'solver': ['adam', 'lbfgs'],
                    },
                },
            },
        }

        for idx, config in enumerate(self.models.values(), start=1):
            print(f"✅ {idx}. {config['name']}")

        print("✅ 7. Soft Voting Ensemble")
        print("✅ 8. Stacking Ensemble")
        print("\n" + "=" * 60)

    def _build_class_weight(self, y_train: np.ndarray) -> Dict[int, float]:
        """Calcula class_weight balanceado para classificadores compatíveis."""
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        return {int(c): float(w) for c, w in zip(classes, weights)}

    def _tune_model(self, estimator, model_key: str, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """Executa tuning com GridSearchCV ou RandomizedSearchCV conforme configuração."""
        tuning_cfg = self.models[model_key]['tuning']
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=self.random_state)
        strategy = tuning_cfg['strategy']

        if strategy == 'randomized':
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=tuning_cfg['param_distributions'],
                n_iter=tuning_cfg.get('n_iter', 10),
                scoring='f1_macro',
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                refit=True,
            )
        else:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=tuning_cfg['param_grid'],
                scoring='f1_macro',
                cv=cv,
                n_jobs=-1,
                refit=True,
            )

        print(f"   🔎 Tuning ({strategy}) para {model_name}...")
        search.fit(X_train, y_train)
        print(f"   ✅ Melhor f1_macro (CV): {search.best_score_:.4f}")
        print(f"   ✅ Melhores hiperparâmetros: {search.best_params_}")

        self.tuning_results[model_key] = {
            'strategy': strategy,
            'best_score': float(search.best_score_),
            'best_params': search.best_params_,
        }

        return search.best_estimator_

    def _build_advanced_ensembles(self, class_weight: Dict[int, float]) -> Dict[str, Dict[str, Any]]:
        """Constroi ensembles avançados usando modelos base."""
        voting_estimators = [
            ('rf', RandomForestClassifier(n_estimators=220, max_depth=12, random_state=self.random_state, n_jobs=-1, class_weight=class_weight)),
            ('xgb', XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.08, subsample=0.85,
                                  colsample_bytree=0.85, random_state=self.random_state, eval_metric='mlogloss', verbosity=0)),
            ('lr', LogisticRegression(max_iter=1500, random_state=self.random_state, class_weight=class_weight)),
        ]

        stacking_estimators = [
            ('rf', RandomForestClassifier(n_estimators=220, max_depth=12, random_state=self.random_state, n_jobs=-1, class_weight=class_weight)),
            ('xgb', XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.08, subsample=0.85,
                                  colsample_bytree=0.85, random_state=self.random_state, eval_metric='mlogloss', verbosity=0)),
            ('svm', SVC(C=8, gamma='scale', probability=True, class_weight=class_weight, random_state=self.random_state)),
        ]

        return {
            'soft_voting_ensemble': {
                'name': 'Soft Voting Ensemble',
                'model': VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=-1),
                'description': 'Combina predições probabilísticas de RF+XGB+LR',
            },
            'stacking_ensemble': {
                'name': 'Stacking Ensemble',
                'model': StackingClassifier(
                    estimators=stacking_estimators,
                    final_estimator=LogisticRegression(max_iter=1500, random_state=self.random_state, class_weight=class_weight),
                    cv=4,
                    passthrough=False,
                    n_jobs=-1,
                ),
                'description': 'Meta-modelo que empilha RF+XGB+SVM',
            },
        }

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, perform_tuning: bool = True) -> Dict[str, Dict[str, Any]]:
        """Treina todos os modelos com tuning opcional e ensembles."""
        print("\n" + "=" * 60)
        print("TREINAMENTO DE MODELOS")
        print("=" * 60)

        training_results: Dict[str, Dict[str, Any]] = {}
        class_weight = self._build_class_weight(y_train)

        for model_key, model_config in self.models.items():
            model_name = model_config['name']
            print(f"\n🔄 Treinando: {model_name}...")

            try:
                cw = class_weight if model_config['supports_class_weight'] else None
                estimator = model_config['factory'](cw)

                if perform_tuning:
                    estimator = self._tune_model(estimator, model_key, model_name, X_train, y_train)
                else:
                    estimator.fit(X_train, y_train)

                self.trained_models[model_key] = estimator
                train_score = estimator.score(X_train, y_train)

                training_results[model_key] = {
                    'name': model_name,
                    'model': estimator,
                    'train_accuracy': float(train_score),
                    'status': 'sucesso',
                    'tuning': self.tuning_results.get(model_key),
                }

                print(f"   ✅ Treinado com sucesso | acc treino: {train_score:.4f}")
            except Exception as e:
                training_results[model_key] = {
                    'name': model_name,
                    'model': None,
                    'train_accuracy': 0.0,
                    'status': f'erro: {str(e)}',
                    'tuning': None,
                }
                print(f"   ❌ Erro no treinamento: {str(e)}")

        advanced = self._build_advanced_ensembles(class_weight)
        for ensemble_key, config in advanced.items():
            model_name = config['name']
            print(f"\n🔄 Treinando: {model_name}...")
            try:
                ensemble = config['model']
                ensemble.fit(X_train, y_train)
                self.trained_models[ensemble_key] = ensemble
                train_score = ensemble.score(X_train, y_train)

                training_results[ensemble_key] = {
                    'name': model_name,
                    'model': ensemble,
                    'train_accuracy': float(train_score),
                    'status': 'sucesso',
                    'tuning': None,
                }
                print(f"   ✅ Treinado com sucesso | acc treino: {train_score:.4f}")
            except Exception as e:
                training_results[ensemble_key] = {
                    'name': model_name,
                    'model': None,
                    'train_accuracy': 0.0,
                    'status': f'erro: {str(e)}',
                    'tuning': None,
                }
                print(f"   ❌ Erro no treinamento: {str(e)}")

        print("\n" + "=" * 60)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("=" * 60)
        return training_results

    def predict_all(self, X_test: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Faz predições com todos os modelos treinados."""
        predictions = {}
        probabilities = {}

        for model_key, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            predictions[model_key] = y_pred

            try:
                probabilities[model_key] = model.predict_proba(X_test)
            except Exception:
                probabilities[model_key] = None

        return predictions, probabilities

    def get_feature_importance(self, model_key: str):
        """Extrai importância de features (se disponível no modelo)."""
        model = self.trained_models.get(model_key)
        if model is None:
            return {}

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim == 2:
                return np.mean(np.abs(coef), axis=0)
            return np.abs(coef)
        return {}

    def save_model(self, model_key: str, filepath: str):
        """Salva modelo treinado em arquivo."""
        if model_key in self.trained_models:
            joblib.dump(self.trained_models[model_key], filepath)
            print(f"✅ Modelo {model_key} salvo em {filepath}")

    def load_model(self, model_key: str, filepath: str):
        """Carrega modelo de arquivo."""
        self.trained_models[model_key] = joblib.load(filepath)
        print(f"✅ Modelo {model_key} carregado de {filepath}")

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Retorna informações sobre um modelo específico."""
        if model_key not in self.models and model_key not in self.trained_models:
            return {}

        base = self.models.get(model_key, {})
        return {
            'name': base.get('name', model_key),
            'description': base.get('description', 'Modelo treinado'),
            'trained': model_key in self.trained_models,
            'tuning': self.tuning_results.get(model_key),
        }

    def list_all_models(self) -> List[Dict[str, str]]:
        """Lista todos os modelos disponíveis (inclui ensembles avançados)."""
        base_models = [
            {'key': key, 'name': cfg['name'], 'description': cfg['description']}
            for key, cfg in self.models.items()
        ]
        base_models.extend([
            {
                'key': 'soft_voting_ensemble',
                'name': 'Soft Voting Ensemble',
                'description': 'Combinação probabilística de múltiplos classificadores',
            },
            {
                'key': 'stacking_ensemble',
                'name': 'Stacking Ensemble',
                'description': 'Meta-modelo treinado sobre modelos base',
            },
        ])
        return base_models
