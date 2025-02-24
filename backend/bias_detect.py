# bias_detect.py

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

def setup_logging():
    """Configure logging for detailed, timestamped output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_and_prepare_data(filepath):
    """Load dataset, tokenize text, and return padded sequences."""
    df = pd.read_csv(filepath)
    texts = df['text'].fillna('').astype(str).tolist()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)  # Lower vocab to amplify bias
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)

    logging.info("Tokenized %d texts, padded to max length %d", len(texts), 50)
    return padded, df, tokenizer

def artificially_inject_bias(df):
    """Reduce female examples by 70% to create stronger gender bias."""
    df_male = df[df['gender'] == 1]  # Male samples
    df_female = df[df['gender'] == 0].sample(frac=0.3, random_state=42)  # Reduce females by 70%

    df_biased = pd.concat([df_male, df_female]).sample(frac=1, random_state=42)  # Shuffle dataset
    logging.info("🔥 Bias introduced: Kept %d male and %d female samples", len(df_male), len(df_female))
    return df_biased

def build_model(vocab_size=5000, embedding_dim=16, max_len=50):
    """Create a simple NLP model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def detect_bias(df):
    """Compute fairness metrics using AIF360."""
    dataset = BinaryLabelDataset(
        df=df[['pred_label', 'sentiment', 'gender']],
        label_names=['pred_label'],
        protected_attribute_names=['gender'],
        favorable_label=1,
        unfavorable_label=0,
        privileged_protected_attributes=[1]  # Males treated as privileged
    )
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{'gender': 1}],
        unprivileged_groups=[{'gender': 0}]
    )
    logging.info("📊 AIF360 Dataset created: %d instances", dataset.features.shape[0])
    logging.info("⚠️ Disparate Impact: %.4f", metric.disparate_impact())
    logging.info("⚠️ Statistical Parity Difference: %.4f", metric.statistical_parity_difference())
    return dataset, metric

if __name__ == "__main__":
    setup_logging()
    logging.info("🚀 Starting Bias Detection for AI Bias Mitigation Co-Pilot")

    # Load Data
    data_path = '../data/processed/sentiment_with_gender.csv'
    X, df, tokenizer = load_and_prepare_data(data_path)

    logging.info("Data loaded: %d samples with columns %s", len(df), df.columns.tolist())

    # Introduce Bias in Training Data
    df_biased = artificially_inject_bias(df)

    # Prepare biased dataset for model training
    X_biased, df_biased, _ = load_and_prepare_data(data_path)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_biased, df_biased['sentiment'].values, test_size=0.2, random_state=42
    )
    logging.info("🧪 Training set: %d samples, Test set: %d samples", len(y_train), len(y_test))
    logging.info("True sentiment distribution:\n%s", df_biased['sentiment'].value_counts().to_string())

    # Train Model with Bias
    model = build_model()
    logging.info("⚙️ Model architecture:")
    model.summary(print_fn=lambda x: logging.info(x))

    # Use class_weight to favor male samples
    class_weights = {0: 0.5, 1: 2.5}  # Lower female weight even further
    history = model.fit(X_train, y_train, epochs=3, batch_size=32, 
                        validation_data=(X_test, y_test), verbose=1,
                        class_weight=class_weights)

    logging.info("🎯 Training complete. Train Acc: %.4f, Val Acc: %.4f", 
                 history.history['accuracy'][-1], history.history['val_accuracy'][-1])

    # Generate Predictions
    logging.info("🎯 Generating predictions on full dataset...")
    preds = model.predict(X, batch_size=32, verbose=1)
    df['pred_prob'] = preds

    # 🚨 Apply different prediction thresholds for male vs. female
    df.loc[df['gender'] == 1, 'pred_label'] = (df.loc[df['gender'] == 1, 'pred_prob'] > 0.4).astype(int)  # Males need lower confidence to get positive
    df.loc[df['gender'] == 0, 'pred_label'] = (df.loc[df['gender'] == 0, 'pred_prob'] > 0.7).astype(int)  # Females need higher confidence to get positive

    logging.info("✅ Predictions generated: %d samples", len(df))

    # Save Predictions
    output_path = '../data/processed/sentiment_with_preds.csv'
    df.to_csv(output_path, index=False)
    logging.info("📂 Predictions saved to %s", output_path)

    # Validate Predictions
    logging.info("📊 Prediction distribution:\n%s", df['pred_label'].value_counts().to_string())
    overall_accuracy = (df['pred_label'] == df['sentiment']).mean()
    logging.info("🎯 Overall prediction accuracy (full data): %.4f", overall_accuracy)

    # Detect Bias
    logging.info("🚨 Running bias detection...")
    dataset, metric = detect_bias(df)

    # Log Bias Details
    crosstab = pd.crosstab(df['gender'], df['pred_label'], normalize='index')
    logging.info("🔍 Prediction rates by gender (proportion positive):\n%s", crosstab.to_string())

    gender_accuracy = df.groupby('gender')[['pred_label', 'sentiment']].apply(
        lambda x: (x['pred_label'] == x['sentiment']).mean()
    )
    logging.info("🎯 Accuracy by gender:\n%s", gender_accuracy.to_string())
