import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functorch.vmap.*")

import logging
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def load_and_prepare_data(filepath):
    """Load and preprocess data for training."""
    df = pd.read_csv(filepath)
    texts = df['text'].fillna('').astype(str).tolist()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    logging.info("Loaded %d samples, tokenized texts, padded to max length 50", len(df))
    X = np.array(padded, dtype=np.float32)
    y = df['sentiment'].values.astype(np.float32)
    return X, y, df, tokenizer

def build_model(vocab_size=5000, embedding_dim=16, max_len=50):
    """Build a simple sentiment classification model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def detect_bias(df, label_col='pred_label'):
    """Detect bias in predictions using AIF360 metrics."""
    dataset = BinaryLabelDataset(
        df=df[[label_col, 'sentiment', 'gender']],
        label_names=[label_col],
        protected_attribute_names=['gender'],
        favorable_label=1,
        unfavorable_label=0,
        privileged_protected_attributes=[1]
    )
    metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{'gender': 1}],
                                      unprivileged_groups=[{'gender': 0}])
    logging.info("Metrics for %s:", label_col)
    logging.info("Disparate Impact: %.4f | Statistical Parity Diff: %.4f", 
                 metric.disparate_impact(), metric.statistical_parity_difference())
    crosstab = pd.crosstab(df['gender'], df[label_col], normalize='index')
    logging.info("Prediction Rates:\n%s", crosstab.to_string())
    accuracy = (df[label_col] == df['sentiment']).mean()
    logging.info("Accuracy: %.4f", accuracy)
    gender_acc = df.groupby('gender').apply(lambda x: (x[label_col] == x['sentiment']).mean(), include_groups=False)
    logging.info("Accuracy by gender:\n%s", gender_acc.to_string())
    return metric, crosstab

def inject_bias(df, X):
    """Inject gender bias by undersampling females."""
    df_male = df[df['gender'] == 1]
    df_female = df[df['gender'] == 0].sample(frac=0.3, random_state=42)
    df_biased = pd.concat([df_male, df_female]).sample(frac=1, random_state=42)
    X_biased = X[df_biased.index]
    logging.info("Bias injected: %d males, %d females", len(df_male), len(df_female))
    return df_biased, X_biased

def train_biased_model(df, X, y, epochs=5):
    """Train a model on biased data with skewed thresholds."""
    df_biased, X_biased = inject_bias(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X_biased, df_biased['sentiment'], 
                                                         test_size=0.2, random_state=42)
    model = build_model()
    logging.info("Training biased model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=1)
    preds = model.predict(X, batch_size=128, verbose=1).flatten()
    df['pred_prob'] = preds
    df['pred_label'] = np.where(df['gender'] == 1, preds > 0.4, preds > 0.6).astype(int)
    logging.info("Applied biased thresholds: Male 0.4, Female 0.6")
    metric, crosstab = detect_bias(df, 'pred_label')
    return model, metric, crosstab, df_biased

def apply_reweighing(df):
    """Apply AIF360 Reweighing to mitigate bias."""
    dataset = BinaryLabelDataset(
        df=df[['sentiment', 'gender']],
        label_names=['sentiment'],
        protected_attribute_names=['gender'],
        favorable_label=1,
        unfavorable_label=0,
        privileged_protected_attributes=[1]
    )
    reweigher = Reweighing(unprivileged_groups=[{'gender': 0}], privileged_groups=[{'gender': 1}])
    dataset_reweighted = reweigher.fit_transform(dataset)
    weights = dataset_reweighted.instance_weights
    logging.info("Applied AIF360 Reweighing: %d weights, min %.2f, max %.2f", 
                 len(weights), weights.min(), weights.max())
    if np.all(weights == 1.0):
        raise ValueError("Reweighing produced uniform weights—mitigation ineffective")
    return weights

def fix_bias(df, X, y, epochs=5):
    """Mitigate bias using AIF360 Reweighing and retraining."""
    try:
        weights = apply_reweighing(df)
        if np.all(weights == 1.0):
            raise ValueError("Reweighing produced uniform weights—mitigation ineffective")
        model = build_model()
        logging.info("Retraining with AIF360 weights...")
        model.fit(X, y, epochs=epochs, batch_size=128, sample_weight=weights, verbose=1)
        mitigated_preds = model.predict(X, batch_size=128, verbose=1).flatten()
        df['mitigated_prob'] = mitigated_preds
        df['mitigated_label'] = (mitigated_preds > 0.5).astype(int)
        logging.info("Mitigated predictions generated: %d samples", len(df))
        metric, crosstab = detect_bias(df, 'mitigated_label')
        return df, metric, crosstab
    except Exception as e:
        logging.error("Mitigation failed: %s", e)
        return df, None, None

def analyze_file(filepath):
    """Analyze a CSV file for bias and mitigation—API-ready function."""
    X, y, df, _ = load_and_prepare_data(filepath)
    start_time = time.time()
    _, before_metric, before_crosstab, _ = train_biased_model(df, X, y)
    mid_time = time.time()
    df, after_metric, after_crosstab = fix_bias(df, X, y)
    end_time = time.time()
    
    logging.info("Before vs. After Bias Summary:")
    logging.info("Before: Impact %.4f, Males %.1f%%, Females %.1f%%", 
                 before_metric.disparate_impact(), before_crosstab.loc[1.0, 1] * 100, before_crosstab.loc[0.0, 1] * 100)
    logging.info("After: Impact %.4f, Males %.1f%%, Females %.1f%%", 
                 after_metric.disparate_impact(), after_crosstab.loc[1.0, 1] * 100, after_crosstab.loc[0.0, 1] * 100)
    logging.info("Runtime: Bias detection %.2fs, Mitigation %.2fs, Total %.2fs", 
                 mid_time - start_time, end_time - mid_time, end_time - start_time)
    
    return {
        'before': {
            'impact': before_metric.disparate_impact(),
            'males': before_crosstab.loc[1.0, 1],
            'females': before_crosstab.loc[0.0, 1]
        },
        'after': {
            'impact': after_metric.disparate_impact(),
            'males': after_crosstab.loc[1.0, 1],
            'females': after_crosstab.loc[0.0, 1]
        }
    }, df

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting AI Bias Mitigation Co-Pilot")
    data_path = '../data/processed/sentiment_with_gender.csv'
    X, y, df, tokenizer = load_and_prepare_data(data_path)

    # Detect Bias & Mitigate
    start_time = time.time()
    biased_model, before_metric, before_crosstab, _ = train_biased_model(df, X, y)
    mid_time = time.time()
    df, after_metric, after_crosstab = fix_bias(df, X, y)
    end_time = time.time()

    # Results Summary
    logging.info("Before vs. After Bias Summary:")
    logging.info("Before: Impact %.4f, Males %.1f%%, Females %.1f%%", 
                 before_metric.disparate_impact(), before_crosstab.loc[1.0, 1] * 100, before_crosstab.loc[0.0, 1] * 100)
    logging.info("After: Impact %.4f, Males %.1f%%, Females %.1f%%", 
                 after_metric.disparate_impact(), after_crosstab.loc[1.0, 1] * 100, after_crosstab.loc[0.0, 1] * 100)
    logging.info("Runtime: Bias detection %.2fs, Mitigation %.2fs, Total %.2fs", 
                 mid_time - start_time, end_time - mid_time, end_time - start_time)

    # Validate Subsample
    df_sample = df.sample(1000, random_state=42)
    detect_bias(df_sample, 'mitigated_label')

    # Save Results
    df.to_csv('../data/processed/sentiment_with_reweighed.csv', index=False)
    logging.info("Saved reweighed data to sentiment_with_reweighed.csv")