"""
Machine Learning Preprocessing Module for Palmer Penguins Dataset

Responsável pela preparação de dados para treinamento de modelos:
- Limpeza de dados (remoção de NaN)
- Codificação de variáveis categóricas (one-hot encoding)
- Normalização/scaling de features
- Separação train/test com estratificação
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from pathlib import Path


class PenguinsMLPreprocessor:
    """Pipeline de preprocessamento para dataset de pinguins."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Inicializa preprocessador.
        
        Args:
            test_size: Proporção de dados para teste (padrão 20%)
            random_state: Seed para reprodutibilidade
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.categorical_features = None
        self.numeric_features = None
        
    def load_and_prepare(self, parquet_path: str) -> Dict[str, Any]:
        """
        Pipeline completo: carrega, limpa, codifica e divide dados.
        
        Args:
            parquet_path: Caminho para arquivo parquet limpo
            
        Returns:
            Dicionário com X_train, X_test, y_train, y_test e metadados
        """
        print("=" * 60)
        print("FASE 1: CARREGAMENTO DE DADOS")
        print("=" * 60)
        
        # Carrega de parquet Polars
        df_polars = pl.read_parquet(parquet_path)
        print(f"✅ Dataset carregado: {df_polars.shape}")
        print(f"   Colunas: {df_polars.columns}")
        
        # Converte para Pandas para sklearn compatibility
        df = df_polars.to_pandas()
        print(f"✅ Convertido para Pandas: {df.shape}")
        
        # ===== ETAPA 1: LIMPEZA =====
        print("\n" + "=" * 60)
        print("FASE 2: LIMPEZA DE DADOS (REMOÇÃO DE NaN)")
        print("=" * 60)
        
        missing_before = df.isnull().sum().sum()
        print(f"Total de valores nulos ANTES: {missing_before}")
        print(f"Detalhamento por coluna:")
        print(df.isnull().sum())
        
        # Remove linhas com valores nulos
        df_clean = df.dropna()
        missing_after = df_clean.isnull().sum().sum()
        
        print(f"\n✅ Total de valores nulos DEPOIS: {missing_after}")
        print(f"📊 Registros removidos: {len(df) - len(df_clean)}")
        print(f"📊 Registros restantes: {len(df_clean)}")
        
        # ===== ETAPA 2: SEPARAR FEATURES E TARGET =====
        print("\n" + "=" * 60)
        print("FASE 3: IDENTIFICAÇÃO DE FEATURES E TARGET")
        print("=" * 60)
        
        # Target: species (variável a prever)
        y = np.array(df_clean['species'].tolist())  # Converte para numpy array padrão
        unique_species = np.unique(y)
        print(f"✅ Target (species): {len(unique_species)} classes")
        print(f"   Classes: {unique_species}")
        print(f"   Distribuição:")
        for sp in unique_species:
            count = (y == sp).sum()
            pct = (count / len(y)) * 100
            print(f"     - {sp}: {count} ({pct:.1f}%)")
        
        # Identifica features categóricas e numéricas
        self.categorical_features = ['island', 'sex']
        self.numeric_features = [
            'bill_length_mm', 'bill_depth_mm', 
            'flipper_length_mm', 'body_mass_g'
        ]
        
        # Features: tudo exceto species e year
        X = df_clean[self.numeric_features + self.categorical_features].copy()
        
        print(f"\n✅ Features numéricas: {self.numeric_features}")
        print(f"✅ Features categóricas: {self.categorical_features}")
        print(f"✅ Total de features (antes encoding): {len(self.numeric_features + self.categorical_features)}")
        
        # ===== ETAPA 3: ONE-HOT ENCODING =====
        print("\n" + "=" * 60)
        print("FASE 4: ONE-HOT ENCODING (VARIÁVEIS CATEGÓRICAS)")
        print("=" * 60)
        
        # One-hot encoding para island e sex
        X_encoded = pd.get_dummies(
            X,
            columns=self.categorical_features,
            drop_first=False,  # Manter todas as colunas
            dtype=int
        )
        
        print(f"✅ Após one-hot encoding: {X_encoded.shape[1]} features")
        print(f"   Novas colunas: {X_encoded.columns.tolist()}")
        
        # ===== ETAPA 4: TRAIN/TEST SPLIT (ESTRATIFICADO) =====
        print("\n" + "=" * 60)
        print("FASE 5: DIVISÃO TRAIN/TEST (80/20 ESTRATIFICADO)")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Mantém proporção de classes
        )
        
        print(f"✅ Treino: {X_train.shape[0]} registros ({(len(X_train)/len(X_encoded))*100:.1f}%)")
        print(f"✅ Teste:  {X_test.shape[0]} registros ({(len(X_test)/len(X_encoded))*100:.1f}%)")
        
        print(f"\n📊 Distribuição TREINO:")
        for sp in unique_species:
            count = (y_train == sp).sum()
            pct = (count / len(y_train)) * 100
            print(f"   - {sp}: {count} ({pct:.1f}%)")
        
        print(f"\n📊 Distribuição TESTE:")
        for sp in unique_species:
            count = (y_test == sp).sum()
            pct = (count / len(y_test)) * 100
            print(f"   - {sp}: {count} ({pct:.1f}%)")
        
        # ===== ETAPA 5: NORMALIZAÇÃO =====
        print("\n" + "=" * 60)
        print("FASE 6: NORMALIZAÇÃO (STANDARDSCALER)")
        print("=" * 60)
        
        # Fit scaler apenas em dados de treino
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ StandardScaler ajustado em dados de treino")
        print(f"   Media (treino): {X_train_scaled.mean(axis=0).mean():.4f}")
        print(f"   StdDev (treino): {X_train_scaled.std(axis=0).mean():.4f}")
        
        # Encode target (y) para valores numéricos
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Converte para numpy arrays para sklearn
        X_train_scaled = np.array(X_train_scaled)
        X_test_scaled = np.array(X_test_scaled)
        
        # Armazena nomes de features
        self.feature_names = X_encoded.columns.tolist()
        
        # ===== RESUMO FINAL =====
        print("\n" + "=" * 60)
        print("✅ PREPROCESSAMENTO CONCLUÍDO COM SUCESSO")
        print("=" * 60)
        
        result = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_encoded,
            'y_test': y_test_encoded,
            'y_train_original': y_train,
            'y_test_original': y_test,
            'feature_names': self.feature_names,
            'class_names': self.label_encoder.classes_,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'metadata': {
                'n_features': X_train_scaled.shape[1],
                'n_classes': len(unique_species),
                'n_train': X_train_scaled.shape[0],
                'n_test': X_test_scaled.shape[0],
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
            }
        }
        
        print(f"\n📊 METADADOS:")
        for key, value in result['metadata'].items():
            print(f"   {key}: {value}")
        
        return result


def prepare_data(parquet_path: str = "dataset/penguins_clean.parquet") -> Dict[str, Any]:
    """
    Função auxiliar para preparar dados em uma linha.
    
    Args:
        parquet_path: Caminho para arquivo parquet
        
    Returns:
        Dicionário com dados preparados
    """
    preprocessor = PenguinsMLPreprocessor(test_size=0.2, random_state=42)
    return preprocessor.load_and_prepare(parquet_path)
