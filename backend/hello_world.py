import logging
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_and_prepare_data(filepath):
    """Load and prepare text data, handling NaN or non-string values."""
    df = pd.read_csv(filepath)
    texts = df['text'].fillna('').astype(str).tolist()
    labels = df['sentiment'].values
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    return padded, labels, tokenizer

def build_model(vocab_size, embedding_dim=16, max_len=50):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def test_aif360(filepath):
    """Test AIF360 dataset creation with numerical dummy feature."""
    df = pd.read_csv(filepath)
    # Ensure no NaN and use a dummy numerical feature instead of text
    df['sentiment'] = df['sentiment'].fillna(0)
    df['gender'] = df['gender'].fillna(0)
    # Dummy feature (e.g., row index) since text isn't numerical
    df['dummy_feature'] = range(len(df))
    features = df[['dummy_feature']]
    labels = df['sentiment']
    protected_attrs = df[['gender']]
    dataset = BinaryLabelDataset(
        df=pd.concat([features, labels, protected_attrs], axis=1),
        label_names=['sentiment'],
        protected_attribute_names=['gender'],
        favorable_label=1,
        unfavorable_label=0,
        privileged_protected_attributes=[1]
    )
    metric = BinaryLabelDatasetMetric(dataset, 
                                      privileged_groups=[{'gender': 1}],
                                      unprivileged_groups=[{'gender': 0}])
    logging.info("AIF360 Dataset created: %d instances", dataset.features.shape[0])
    logging.info("Disparate Impact (raw data): %.4f", metric.disparate_impact())
    return dataset

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting Hello World Test")

    # TensorFlow Test
    data_path = '../data/processed/sentiment_with_gender.csv'
    X, y, tokenizer = load_and_prepare_data(data_path)
    logging.info("Data loaded: %d samples", len(y))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Train: %d, Test: %d", len(y_train), len(y_test))

    # Build and train model
    model = build_model(vocab_size=5000)
    model.summary(print_fn=lambda x: logging.info(x))
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    logging.info("Training complete. Final accuracy: %.4f", history.history['accuracy'][-1])

    # AIF360 Test
    dataset = test_aif360(data_path)