"""
Machine Learning Models Module for Palmer Penguins Classification

Implementa 5+ modelos de classificação:
1. Random Forest (ensemble de árvores)
2. XGBoost (gradient boosting)
3. Support Vector Machine (RBF kernel)
4. K-Nearest Neighbors
5. Logistic Regression
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


class PenguinsMLModels:
    """Gerenciador de múltiplos modelos de classificação."""
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa modelos.
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Define configurações dos 5 modelos."""
        
        print("\n" + "=" * 60)
        print("INICIALIZAÇÃO DE 5 MODELOS CANDIDATOS")
        print("=" * 60)
        
        # 1. RANDOM FOREST (Recomendado)
        self.models['random_forest'] = {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'name': 'Random Forest',
            'description': 'Ensemble de árvores - excelente para dados pequenos',
            'hyperparameters': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'class_weight': 'balanced'
            }
        }
        print("✅ 1. Random Forest (n_estimators=200, max_depth=15)")
        
        # 2. XGBOOST (Estado-da-arte)
        self.models['xgboost'] = {
            'model': XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0
            ),
            'name': 'XGBoost',
            'description': 'Gradient boosting - máxima performance',
            'hyperparameters': {
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
        }
        print("✅ 2. XGBoost (n_estimators=200, max_depth=7, lr=0.1)")
        
        # 3. SUPPORT VECTOR MACHINE
        self.models['svm'] = {
            'model': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'name': 'SVM (RBF)',
            'description': 'Máquina de vetores suporte com kernel RBF',
            'hyperparameters': {
                'kernel': 'rbf',
                'C': 10,
                'gamma': 'scale',
                'class_weight': 'balanced'
            }
        }
        print("✅ 3. SVM (kernel=rbf, C=10, gamma=scale)")
        
        # 4. K-NEAREST NEIGHBORS
        self.models['knn'] = {
            'model': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ),
            'name': 'K-Nearest Neighbors',
            'description': 'Algoritmo baseado em instâncias com k=5',
            'hyperparameters': {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'euclidean'
            }
        }
        print("✅ 4. KNN (k=5, weights=distance, metric=euclidean)")
        
        # 5. LOGISTIC REGRESSION
        self.models['logistic_regression'] = {
            'model': LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'name': 'Logistic Regression',
            'description': 'Classificador linear - baseline interpretável',
            'hyperparameters': {
                'max_iter': 1000,
                'solver': 'lbfgs',
                'class_weight': 'balanced'
            }
        }
        print("✅ 5. Logistic Regression (solver=lbfgs, max_iter=1000)")
        
        # 6. NEURAL NETWORK (MLP)
        self.models['neural_network'] = {
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                solver='adam',
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                verbose=False
            ),
            'name': 'Neural Network (MLP)',
            'description': 'Rede neural com 2 camadas ocultas - reduzida para evitar overfitting',
            'hyperparameters': {
                'hidden_layer_sizes': (64, 32),
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'solver': 'adam',
                'early_stopping': True
            }
        }
        print("✅ 6. Neural Network (layers=[64,32], adam, early_stopping)")
        
        print("\n" + "=" * 60)
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict]:
        """
        Treina todos os 5 modelos.
        
        Args:
            X_train: Features de treino (normalizado)
            y_train: Target de treino (encoded)
            
        Returns:
            Dicionário com resultados de treinamento
        """
        print("\n" + "=" * 60)
        print("TREINAMENTO DE TODOS OS MODELOS")
        print("=" * 60)
        
        training_results = {}
        
        for model_key, model_config in self.models.items():
            model_name = model_config['name']
            model = model_config['model']
            
            print(f"\n🔄 Treinando: {model_name}...")
            
            try:
                model.fit(X_train, y_train)
                self.trained_models[model_key] = model
                
                # Acurácia em treino
                train_score = model.score(X_train, y_train)
                
                training_results[model_key] = {
                    'name': model_name,
                    'model': model,
                    'train_accuracy': train_score,
                    'status': 'sucesso'
                }
                
                print(f"   ✅ Treinado com sucesso!")
                print(f"   📊 Acurácia treino: {train_score:.4f}")
                
            except Exception as e:
                print(f"   ❌ Erro no treinamento: {str(e)}")
                training_results[model_key] = {
                    'name': model_name,
                    'model': None,
                    'train_accuracy': 0,
                    'status': f'erro: {str(e)}'
                }
        
        print("\n" + "=" * 60)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("=" * 60)
        
        return training_results
    
    def predict_all(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Faz predições com todos os modelos treinados.
        
        Args:
            X_test: Features de teste
            
        Returns:
            Dicionário com predições por modelo
        """
        predictions = {}
        probabilities = {}
        
        for model_key, model in self.trained_models.items():
            model_name = self.models[model_key]['name']
            
            # Predições
            y_pred = model.predict(X_test)
            predictions[model_key] = y_pred
            
            # Probabilidades (se disponível)
            try:
                y_proba = model.predict_proba(X_test)
                probabilities[model_key] = y_proba
            except AttributeError:
                probabilities[model_key] = None
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_key: str) -> Dict[str, float]:
        """
        Extrai importância de features (se disponível no modelo).
        
        Args:
            model_key: Chave do modelo
            
        Returns:
            Dicionário com importâncias
        """
        model = self.trained_models.get(model_key)
        
        if model is None:
            return {}
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Para modelos lineares, retorna valor absoluto dos coeficientes
            return np.abs(model.coef_[0])
        else:
            return {}
    
    def save_model(self, model_key: str, filepath: str):
        """Salva modelo treinado em arquivo."""
        if model_key in self.trained_models:
            model = self.trained_models[model_key]
            joblib.dump(model, filepath)
            print(f"✅ Modelo {model_key} salvo em {filepath}")
    
    def load_model(self, model_key: str, filepath: str):
        """Carrega modelo de arquivo."""
        model = joblib.load(filepath)
        self.trained_models[model_key] = model
        print(f"✅ Modelo {model_key} carregado de {filepath}")
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Retorna informações sobre um modelo específico."""
        if model_key not in self.models:
            return {}
        
        return {
            'name': self.models[model_key]['name'],
            'description': self.models[model_key]['description'],
            'hyperparameters': self.models[model_key]['hyperparameters'],
            'trained': model_key in self.trained_models
        }
    
    def list_all_models(self) -> List[Dict[str, str]]:
        """Lista todos os modelos disponíveis."""
        return [
            {
                'key': key,
                'name': config['name'],
                'description': config['description']
            }
            for key, config in self.models.items()
        ]
