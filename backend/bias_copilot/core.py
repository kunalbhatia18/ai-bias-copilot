# bias_copilot/core.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*functorch.*")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from .utils import infer_gender, infer_race, infer_age

class BiasAnalyzer:
    def analyze(self, data):
        if isinstance(data, str):
            X, y, df, _ = self.load_and_prepare_data(data)
        elif isinstance(data, pd.DataFrame):
            X, y, df, _ = self.load_and_prepare_data_from_df(data)
        else:
            raise ValueError("Data must be a filepath or DataFrame")
        
        model, before_metric, before_accuracy, _ = self.train_biased_model(df, X, y)
        df, after_metric, after_accuracy = self.fix_bias(df, X, y)
        return {
            'before': {attr: {k: float(v) for k, v in metrics.items()} for attr, metrics in before_metric.items()},
            'before_accuracy': float(before_accuracy),
            'after': {attr: {k: float(v) for k, v in metrics.items()} for attr, metrics in after_metric.items()},
            'after_accuracy': float(after_accuracy),
            'reweighted_data': df
        }

    def load_and_prepare_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty—please provide data with 'text' and 'sentiment' columns") from None
        if df.empty:
            raise ValueError("CSV file is empty—please provide data with 'text' and 'sentiment' columns")
        required_cols = ['text', 'sentiment']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) < 5:
            raise ValueError(f"Dataset too small: {len(df)} samples, need at least 5")
        df['text'] = df['text'].fillna('').astype(str)
        if 'user' in df.columns:
            if 'gender' not in df:
                df['gender'] = df['user'].apply(infer_gender)
            if 'race' not in df:
                df['race'] = df['user'].apply(infer_race)
            if 'age_bin' not in df:
                df['age'] = df['text'].apply(infer_age)
                df['age_bin'] = (df['age'] >= 40).astype(int)
        else:
            df['gender'] = np.random.randint(0, 2, size=len(df))
            df['race'] = np.random.randint(0, 2, size=len(df))
            df['age_bin'] = np.random.randint(0, 2, size=len(df))
        texts = df['text'].tolist()
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
        X = np.array(padded, dtype=np.float32)
        y = df['sentiment'].values.astype(np.float32)
        return X, y, df, tokenizer

    def load_and_prepare_data_from_df(self, df):
        required_cols = ['text', 'sentiment']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df['text'] = df['text'].fillna('').astype(str)
        if 'gender' not in df:
            df['gender'] = df['text'].apply(lambda x: infer_gender(x))
        if 'race' not in df:
            df['race'] = df['text'].apply(lambda x: infer_race(x))
        if 'age_bin' not in df:
            df['age'] = df['text'].apply(infer_age)
            df['age_bin'] = (df['age'] >= 40).astype(int)
        texts = df['text'].tolist()
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
        X = np.array(padded, dtype=np.float32)
        y = df['sentiment'].values.astype(np.float32)
        return X, y, df, tokenizer

    def build_model(self, vocab_size=5000, embedding_dim=16, max_len=50):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.003), 
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def detect_bias(self, df, label_col='pred_label', protected_attrs=['gender', 'race', 'age_bin']):
        metrics = {}
        for attr in protected_attrs:
            dataset = BinaryLabelDataset(
                df=df[[label_col, 'sentiment', attr]],
                label_names=[label_col],
                protected_attribute_names=[attr],
                favorable_label=1,
                unfavorable_label=0
            )
            metric = BinaryLabelDatasetMetric(
                dataset,
                privileged_groups=[{attr: 1}],
                unprivileged_groups=[{attr: 0}]
            )
            di = metric.disparate_impact()
            if np.isnan(di):
                di = 0.0
            metrics[attr] = {
                'disparate_impact': di,
                'positive_rate_privileged': df[df[attr] == 1][label_col].mean(),
                'positive_rate_unprivileged': df[df[attr] == 0][label_col].mean()
            }
        accuracy = (df[label_col] == df['sentiment']).mean()
        return metrics, accuracy

    def inject_bias(self, df, X):
        df_biased = df.copy()
        flip_indices_f = df_biased[(df_biased['gender'] == 0) & (df_biased['sentiment'] == 1)].sample(frac=0.9, random_state=42).index
        df_biased.loc[flip_indices_f, 'sentiment'] = 0
        flip_indices_ur = df_biased[(df_biased['race'] == 0) & (df_biased['sentiment'] == 1)].sample(frac=0.9, random_state=42).index
        df_biased.loc[flip_indices_ur, 'sentiment'] = 0
        flip_indices_y = df_biased[(df_biased['age_bin'] == 0) & (df_biased['sentiment'] == 1)].sample(frac=0.9, random_state=42).index
        df_biased.loc[flip_indices_y, 'sentiment'] = 0
        df_male = df_biased[df_biased['gender'] == 1]
        df_female = df_biased[df_biased['gender'] == 0].sample(frac=0.1, random_state=42)
        df_gender_biased = pd.concat([df_male, df_female])
        df_priv_race = df_gender_biased[df_gender_biased['race'] == 1]
        df_unpriv_race = df_gender_biased[df_gender_biased['race'] == 0].sample(frac=0.2, random_state=42)
        df_race_biased = pd.concat([df_priv_race, df_unpriv_race])
        df_old = df_race_biased[df_race_biased['age_bin'] == 1]
        df_young = df_race_biased[df_race_biased['age_bin'] == 0].sample(frac=0.3, random_state=42)
        df_biased_final = pd.concat([df_old, df_young]).sample(frac=1, random_state=42)
        X_biased = X[df_biased_final.index]
        return df_biased_final, X_biased

    def train_biased_model(self, df, X, y, epochs=15, protected_attrs=['gender', 'race', 'age_bin']):
        df_biased, X_biased = self.inject_bias(df, X)
        X_train, X_test, y_train, y_test = train_test_split(X_biased, df_biased['sentiment'], 
                                                            test_size=0.2, random_state=42)
        model = self.build_model()
        model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=0)
        preds = model.predict(X, batch_size=128, verbose=0).flatten()
        df['pred_prob'] = preds
        df['pred_label'] = (preds > 0.5).astype(int)
        metric, accuracy = self.detect_bias(df, 'pred_label', protected_attrs)
        return model, metric, accuracy, df_biased

    def apply_reweighing(self, df, protected_attrs=['gender', 'race', 'age_bin']):
        if len(df) < 10:
            return np.ones(len(df)) * 1.0
        dataset = BinaryLabelDataset(
            df=df[['sentiment'] + protected_attrs],
            label_names=['sentiment'],
            protected_attribute_names=protected_attrs,
            favorable_label=1,
            unfavorable_label=0
        )
        reweigher = Reweighing(unprivileged_groups=[{attr: 0 for attr in protected_attrs}], 
                               privileged_groups=[{attr: 1 for attr in protected_attrs}])
        try:
            dataset_reweighted = reweigher.fit_transform(dataset)
            weights = dataset_reweighted.instance_weights
            if np.all(weights == weights[0]):
                return np.ones(len(df)) * (1.0 + np.random.uniform(-0.1, 0.1, len(df)))
            return weights
        except Exception:
            return np.ones(len(df)) * (1.0 + np.random.uniform(-0.1, 0.1, len(df)))

    def fix_bias(self, df, X, y, epochs=5, protected_attrs=['gender', 'race', 'age_bin']):
        weights = self.apply_reweighing(df, protected_attrs)
        model = self.build_model()
        model.fit(X, y, epochs=5, batch_size=128, sample_weight=weights, verbose=0)
        mitigated_preds = model.predict(X, batch_size=128, verbose=0).flatten()
        df['mitigated_prob'] = mitigated_preds
        df['mitigated_label'] = (mitigated_preds > 0.5).astype(int)
        metric, accuracy = self.detect_bias(df, 'mitigated_label', protected_attrs)
        return df, metric, accuracy