import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functorch.vmap.*")

import logging
import sys
import pandas as pd
import tensorflow as tf
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def load_data(filepath):
    df = pd.read_csv(filepath)
    logging.info("Loaded data: %d samples with columns %s", len(df), df.columns.tolist())
    return df

def prepare_data(df):
    texts = df['text'].fillna('').astype(str).tolist()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    logging.info("Tokenized %d texts, padded to max length 50", len(texts))
    return padded, tokenizer

def build_model(vocab_size=5000, embedding_dim=16, max_len=50):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def detect_bias(df, label_col='pred_label'):
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

def reweight_data(df):
    weights = df['gender'].map({0: 2.0, 1: 0.5})
    logging.info("Applied sample weights: Female 2.0, Male 0.5")
    return weights

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting Bias Mitigation for AI Bias Mitigation Co-Pilot")
    data_path = '../data/processed/sentiment_with_preds.csv'
    df = load_data(data_path)

    # Before Mitigation
    detect_bias(df, 'pred_label')

    # Prepare and Retrain
    X, tokenizer = prepare_data(df)
    weights = reweight_data(df)
    model = build_model()
    logging.info("Training model with reweighting...")
    model.summary(print_fn=lambda x: logging.info(x))
    history = model.fit(X, df['sentiment'], epochs=5, batch_size=32, sample_weight=weights, verbose=1)
    logging.info("Mitigated training complete. Final accuracy: %.4f", history.history['accuracy'][-1])

    # Mitigated Predictions
    mitigated_preds = model.predict(X, batch_size=32, verbose=1)
    df['mitigated_prob'] = mitigated_preds
    df['mitigated_label'] = (mitigated_preds > 0.5).astype(int)
    logging.info("Mitigated predictions generated: %d samples", len(df))

    # After Mitigation
    detect_bias(df, 'mitigated_label')

    # Validation
    logging.info("Mitigated prediction distribution:\n%s", df['mitigated_label'].value_counts().to_string())
    mitigated_accuracy = (df['mitigated_label'] == df['sentiment']).mean()
    logging.info("Mitigated accuracy (full data): %.4f", mitigated_accuracy)
    gender_acc = df.groupby('gender', include_groups=False).apply(lambda x: (x['mitigated_label'] == x['sentiment']).mean())
    logging.info("Mitigated accuracy by gender:\n%s", gender_acc.to_string())

    # Save
    df.to_csv('../data/processed/sentiment_with_mitigated.csv', index=False)
    logging.info("Saved mitigated data to sentiment_with_mitigated.csv")