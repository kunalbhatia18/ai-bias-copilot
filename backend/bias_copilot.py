import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functorch.vmap.*")

import logging
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

def setup_logging():
    """Configure logging for detailed, timestamped output."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def load_and_prepare_data(filepath):
    """Load dataset, tokenize text, and return padded sequences."""
    df = pd.read_csv(filepath)
    texts = df['text'].fillna('').astype(str).tolist()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    logging.info("Loaded %d samples, tokenized texts, padded to max length 50", len(df))
    return padded, df, tokenizer

def build_model(vocab_size=5000, embedding_dim=16, max_len=50):
    """Create a simple NLP model for sentiment classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def detect_bias(df, label_col='pred_label'):
    """Compute fairness metrics using AIF360."""
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
    logging.info("Disparate Impact: %.4f", metric.disparate_impact())
    logging.info("Statistical Parity Diff: %.4f", metric.statistical_parity_difference())
    crosstab = pd.crosstab(df['gender'], df[label_col], normalize='index')
    logging.info("Prediction Rates:\n%s", crosstab.to_string())
    return dataset, metric

def inject_bias(df, X):
    """Reduce female samples by 70% to inject gender bias, adjust X accordingly."""
    df_male = df[df['gender'] == 1]
    df_female = df[df['gender'] == 0].sample(frac=0.3, random_state=42)
    df_biased = pd.concat([df_male, df_female]).sample(frac=1, random_state=42)
    X_biased = X[df_biased.index]  # Subset X to match biased df
    logging.info("Bias injected: %d males, %d females", len(df_male), len(df_female))
    return df_biased, X_biased

def apply_reweighing(df):
    """Apply AIF360 Reweighing to generate fairness weights."""
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
    logging.info("Applied AIF360 Reweighing: %d weights generated", len(weights))
    return weights

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting AI Bias Mitigation Co-Pilot")
    data_path = '../data/processed/sentiment_with_gender.csv'
    X, df, tokenizer = load_and_prepare_data(data_path)

    # Inject Bias and Detect
    df_biased, X_biased = inject_bias(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X_biased, df_biased['sentiment'], 
                                                         test_size=0.2, random_state=42)
    model = build_model()
    logging.info("Training on biased data...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    preds = model.predict(X, batch_size=32, verbose=1)  # Predict on full X
    df['pred_prob'] = preds
    df['pred_label'] = (preds > 0.5).astype(int)
    logging.info("Initial predictions generated: %d samples", len(df))
    detect_bias(df, 'pred_label')

    # Mitigation with Reweighing
    weights = apply_reweighing(df)
    model = build_model()
    logging.info("Retraining with AIF360 weights on full data...")
    history = model.fit(X, df['sentiment'], epochs=5, batch_size=32, sample_weight=weights, verbose=1)
    mitigated_preds = model.predict(X, batch_size=32, verbose=1)
    df['mitigated_prob'] = mitigated_preds
    df['mitigated_label'] = (mitigated_preds > 0.5).astype(int)
    logging.info("Mitigated predictions generated: %d samples", len(df))
    detect_bias(df, 'mitigated_label')

    # Validation
    logging.info("Mitigated prediction distribution:\n%s", df['mitigated_label'].value_counts().to_string())
    accuracy = (df['mitigated_label'] == df['sentiment']).mean()
    logging.info("Mitigated accuracy: %.4f", accuracy)
    gender_acc = df.groupby('gender').apply(lambda x: (x['mitigated_label'] == x['sentiment']).mean())
    logging.info("Mitigated accuracy by gender:\n%s", gender_acc.to_string())

    # Save
    df.to_csv('../data/processed/sentiment_with_reweighed.csv', index=False)
    logging.info("Saved reweighed data to sentiment_with_reweighed.csv")