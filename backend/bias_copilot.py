import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functorch.vmap.*")

import logging
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import requests
import re
from ethnicolr import census_ln # type: ignore
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

# def infer_gender(username):
#     """Infer gender from username using Genderize.io API."""
#     try:
#         response = requests.get(f'https://api.genderize.io?name={username.split()[0]}', timeout=2)
#         gender = response.json().get('gender')
#         return 0 if gender == 'female' else 1 if gender == 'male' else np.random.randint(0, 2)
#     except:
#         return np.random.randint(0, 2)

def infer_gender(username):
    """Mock gender inference based on username length."""
    if not isinstance(username, str):
        username = str(username) if username is not None else ""
    return 0 if len(username) % 2 == 0 else 1  # 0: female (even), 1: male (odd)

def infer_race(username):
    """Infer race from username using ethnicolr."""
    try:
        race = census_ln(username.split()[0])
        return 0 if race in ['black', 'hispanic'] else 1 if race in ['white', 'asian'] else np.random.randint(0, 2)
    except:
        return np.random.randint(0, 2)

def infer_age(text):
    """Infer age from text using regex, handling non-string inputs."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    match = re.search(r'\b(?:age|aged|I\'m|Im)\s*(\d{1,2})\b', text.lower())
    return int(match.group(1)) if match else np.random.randint(18, 65)

def load_and_prepare_data(filepath):
    """Load and preprocess data, inferring attributes from raw input."""
    df = pd.read_csv(filepath)
    required_cols = ['text', 'sentiment']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure 'text' is string, handle NaN
    df['text'] = df['text'].fillna('').astype(str)
    
    # Infer attributes if missing
    if 'user' in df.columns:
        if 'gender' not in df:
            df['gender'] = df['user'].apply(infer_gender)
        if 'race' not in df:
            df['race'] = df['user'].apply(infer_race)
        if 'age_bin' not in df:
            df['age'] = df['text'].apply(infer_age)
            df['age_bin'] = (df['age'] >= 40).astype(int)
    else:
        logging.warning("No 'user' column—random attributes assigned")
        df['gender'] = np.random.randint(0, 2, size=len(df))
        df['race'] = np.random.randint(0, 2, size=len(df))
        df['age_bin'] = np.random.randint(0, 2, size=len(df))
    
    texts = df['text'].tolist()  # Already string
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    logging.info("Loaded %d samples, inferred attributes, padded to max length 50", len(df))
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
    # Use legacy Adam optimizer for better performance on M1/M2 Macs
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.003), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def detect_bias(df, label_col='pred_label', protected_attrs=['gender', 'race', 'age_bin']):
    """Detect bias in predictions across multiple protected attributes using AIF360 metrics."""
    metrics = {}
    crosstabs = {}
    
    required_cols = [label_col, 'sentiment'] + protected_attrs
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in dataframe: {missing_cols}")
        raise ValueError(f"Dataframe must contain {required_cols}")
    
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
        metrics[attr] = {
            'disparate_impact': metric.disparate_impact(),
            'statistical_parity_diff': metric.statistical_parity_difference()
        }
        crosstabs[attr] = pd.crosstab(df[attr], df[label_col], normalize='index')
        logging.info(f"Metrics for {label_col} - {attr}:")
        logging.info(f"Disparate Impact: {metrics[attr]['disparate_impact']:.4f} | "
                     f"Statistical Parity Diff: {metrics[attr]['statistical_parity_diff']:.4f}")
        logging.info(f"Prediction Rates:\n{crosstabs[attr].to_string()}")

    accuracy = (df[label_col] == df['sentiment']).mean()
    logging.info(f"Overall Accuracy: {accuracy:.4f}")
    
    for attr in protected_attrs:
        attr_acc = df.groupby(attr).apply(
            lambda x: (x[label_col] == x['sentiment']).mean(), 
            include_groups=False
        )
        logging.info(f"Accuracy by {attr}:\n{attr_acc.to_string()}")

    return metrics, crosstabs

def inject_bias(df, X):
    """Inject strong bias by flipping unprivileged positives to negatives, no flips for privileged."""
    df_biased = df.copy()
    
    # Gender bias: flip 50% of female positives to negatives
    flip_indices_f = df_biased[(df_biased['gender'] == 0) & (df_biased['sentiment'] == 1)].sample(frac=0.5, random_state=42).index
    df_biased.loc[flip_indices_f, 'sentiment'] = 0
    
    # Race bias: flip 40% of unprivileged race positives to negatives
    flip_indices_ur = df_biased[(df_biased['race'] == 0) & (df_biased['sentiment'] == 1)].sample(frac=0.4, random_state=42).index
    df_biased.loc[flip_indices_ur, 'sentiment'] = 0
    
    # Age bias: flip 30% of young positives to negatives
    flip_indices_y = df_biased[(df_biased['age_bin'] == 0) & (df_biased['sentiment'] == 1)].sample(frac=0.3, random_state=42).index
    df_biased.loc[flip_indices_y, 'sentiment'] = 0
    
    # Undersample unprivileged groups
    df_male = df_biased[df_biased['gender'] == 1]
    df_female = df_biased[df_biased['gender'] == 0].sample(frac=0.3, random_state=42)
    df_gender_biased = pd.concat([df_male, df_female])
    
    df_priv_race = df_gender_biased[df_gender_biased['race'] == 1]
    df_unpriv_race = df_gender_biased[df_gender_biased['race'] == 0].sample(frac=0.5, random_state=42)
    df_race_biased = pd.concat([df_priv_race, df_unpriv_race])
    
    df_old = df_race_biased[df_race_biased['age_bin'] == 1]
    df_young = df_race_biased[df_race_biased['age_bin'] == 0].sample(frac=0.6, random_state=42)
    df_biased_final = pd.concat([df_old, df_young]).sample(frac=1, random_state=42)
    
    X_biased = X[df_biased_final.index]
    logging.info("Bias injected: %d males, %d females, %d privileged race, %d unprivileged race, %d old, %d young",
                 len(df_male), len(df_female), len(df_priv_race), len(df_unpriv_race), len(df_old), len(df_young))
    return df_biased_final, X_biased

def train_biased_model(df, X, y, epochs=15, protected_attrs=['gender', 'race', 'age_bin']):
    """Train a model on biased data with neutral threshold."""
    df_biased, X_biased = inject_bias(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X_biased, df_biased['sentiment'], 
                                                        test_size=0.2, random_state=42)
    logging.info("Biased split: train %d, test %d, positive rate %.2f", 
                 len(X_train), len(X_test), y_train.mean())
    model = build_model()
    logging.info("Training biased model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=1)
    preds = model.predict(X, batch_size=128, verbose=1).flatten()
    df['pred_prob'] = preds
    df['pred_label'] = (preds > 0.5).astype(int)
    metric, crosstab = detect_bias(df, 'pred_label', protected_attrs=protected_attrs)
    return model, metric, crosstab, df_biased

def apply_reweighing(df, protected_attrs=['gender', 'race', 'age_bin']):
    """Apply AIF360 Reweighing to mitigate bias across multiple attributes."""
    dataset = BinaryLabelDataset(
        df=df[['sentiment'] + protected_attrs],
        label_names=['sentiment'],
        protected_attribute_names=protected_attrs,
        favorable_label=1,
        unfavorable_label=0
    )
    reweigher = Reweighing(unprivileged_groups=[{attr: 0 for attr in protected_attrs}], 
                           privileged_groups=[{attr: 1 for attr in protected_attrs}])
    dataset_reweighted = reweigher.fit_transform(dataset)
    weights = dataset_reweighted.instance_weights
    logging.info("Applied AIF360 Reweighing: %d weights, min %.2f, max %.2f", 
                 len(weights), weights.min(), weights.max())
    logging.info("Weights distribution: %s", np.histogram(weights, bins=10)[0])
    if np.all(weights == 1.0):
        raise ValueError("Reweighing produced uniform weights—mitigation ineffective")
    return weights

def fix_bias(df, X, y, epochs=5, protected_attrs=['gender', 'race', 'age_bin']):
    """Mitigate bias using AIF360 Reweighing and retraining."""
    try:
        weights = apply_reweighing(df, protected_attrs)
        if np.all(weights == 1.0):
            raise ValueError("Reweighing produced uniform weights—mitigation ineffective")
        model = build_model()
        logging.info("Retraining with AIF360 weights...")
        model.fit(X, y, epochs=epochs, batch_size=128, sample_weight=weights, verbose=1)
        mitigated_preds = model.predict(X, batch_size=128, verbose=1).flatten()
        df['mitigated_prob'] = mitigated_preds
        df['mitigated_label'] = (mitigated_preds > 0.5).astype(int)
        logging.info("Mitigated predictions generated: %d samples", len(df))
        metric, crosstab = detect_bias(df, 'mitigated_label', protected_attrs=protected_attrs)
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
    for attr in ['gender', 'race', 'age_bin']:
        logging.info(f"Before {attr}: Impact {before_metric[attr]['disparate_impact']:.4f}")
        logging.info(f"After {attr}: Impact {after_metric[attr]['disparate_impact']:.4f}")
    logging.info("Runtime: Bias detection %.2fs, Mitigation %.2fs, Total %.2fs", 
                 mid_time - start_time, end_time - mid_time, end_time - start_time)
    
    return {
        'before': {
            'gender_impact': before_metric['gender']['disparate_impact'],
            'race_impact': before_metric['race']['disparate_impact'],
            'age_bin_impact': before_metric['age_bin']['disparate_impact']
        },
        'after': {
            'gender_impact': after_metric['gender']['disparate_impact'],
            'race_impact': after_metric['race']['disparate_impact'],
            'age_bin_impact': after_metric['age_bin']['disparate_impact']
        }
    }, df

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting AI Bias Mitigation Co-Pilot")
    data_path = '../data/processed/sentiment_with_gender.csv'
    X, y, df, tokenizer = load_and_prepare_data(data_path)

    df['race'] = np.random.randint(0, 2, size=len(df))  # 0: unprivileged, 1: privileged
    df['age'] = np.random.randint(18, 65, size=len(df))
    df['age_bin'] = (df['age'] >= 40).astype(int)  # 0: <40 (unprivileged), 1: >=40 (privileged)

    start_time = time.time()
    biased_model, before_metrics, before_crosstabs, _ = train_biased_model(df, X, y)
    mid_time = time.time()
    df, after_metrics, after_crosstabs = fix_bias(df, X, y)
    end_time = time.time()

    logging.info("Before vs. After Bias Summary:")
    for attr in ['gender', 'race', 'age_bin']:
        logging.info(f"Before {attr}: Impact {before_metrics[attr]['disparate_impact']:.4f}")
        logging.info(f"After {attr}: Impact {after_metrics[attr]['disparate_impact']:.4f}")
    logging.info("Runtime: Bias detection %.2fs, Mitigation %.2fs, Total %.2fs", 
                 mid_time - start_time, end_time - mid_time, end_time - start_time)

    df_sample = df.sample(1000, random_state=42)
    detect_bias(df_sample, 'mitigated_label', protected_attrs=['gender', 'race', 'age_bin'])

    df.to_csv('../data/processed/raw_sentiment140.csv', index=False)
    logging.info("Saved reweighed data to raw_sentiment140.csv")