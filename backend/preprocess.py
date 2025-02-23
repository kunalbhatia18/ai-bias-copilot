import pandas as pd
import gender_guesser.detector as gender
import re
import logging
import sys

def setup_logging():
    """Configure logging for clear, timestamped output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_data(filepath):
    """Load raw Sentiment140 CSV with error handling."""
    try:
        df = pd.read_csv(filepath, encoding='latin-1', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'])
        logging.info("Loaded raw data: %s rows", df.shape[0])
        return df
    except Exception as e:
        logging.error("Error loading data: %s", e)
        sys.exit(1)

def sample_data(df, n=20000, random_state=42):
    """Sample n rows from the dataset for speed."""
    df_sampled = df.sample(n=n, random_state=random_state)
    logging.info("Sampled data: %s rows", df_sampled.shape[0])
    return df_sampled

def clean_data(df):
    """Clean data: map sentiment to 0/1, remove URLs/mentions, drop unused columns."""
    df['sentiment'] = df['target'].map({0: 0, 4: 1})
    df['text'] = df['text'].str.replace(r'http\S+|@\S+', '', regex=True).str.strip()
    df = df.drop(columns=['target', 'id', 'date', 'flag'])
    logging.info("Cleaned data; columns: %s", df.columns.tolist())
    return df

def guess_gender_advanced(username, detector, extra_names):
    """Advanced gender inference: clean name, check extras, substring fallback."""
    cleaned = re.sub(r'[^a-zA-Z]', '', username.lower())
    guess = detector.get_gender(cleaned)
    if guess in ['male', 'female', 'mostly_male', 'mostly_female']:
        return guess
    if cleaned in extra_names:
        return extra_names[cleaned]
    # Substring check for more hits
    for i in range(2, len(cleaned)):
        sub = cleaned[:i]
        guess = detector.get_gender(sub)
        if guess in ['male', 'female', 'mostly_male', 'mostly_female']:
            return guess
    return None  # Unmapped genders drop naturally

def infer_gender(df):
    """Infer gender from usernames, map to binary, drop unknowns."""
    d = gender.Detector(case_sensitive=False)
    extra_names = {
    'matty': 'male', 'elle': 'female', 'scott': 'male', 
    'mary': 'female', 'john': 'male', 'chris': 'male', 
    'sarah': 'female', 'alex': 'male', 'jess': 'female',
    'mike': 'male', 'liz': 'female', 'tom': 'male', 
    'kate': 'female', 'ben': 'male', 'emma': 'female',
    'sam': 'male', 'amy': 'female', 'jake': 'male',
    'sophie': 'female', 'nick': 'male'
}
    df['gender_raw'] = df['user'].apply(lambda x: guess_gender_advanced(x, d, extra_names))
    logging.info("Pre-Drop Gender Raw Counts:\n%s", df['gender_raw'].value_counts())
    
    gender_map = {'male': 1, 'female': 0, 'mostly_male': 1, 'mostly_female': 0}
    df['gender'] = df['gender_raw'].map(gender_map)
    initial_count = df.shape[0]
    df = df.dropna(subset=['gender'])  # Drop NaN (unmapped: None from guess_gender_advanced)
    dropped_count = initial_count - df.shape[0]
    logging.info("Dropped %d rows due to unknown gender", dropped_count)
    df = df.drop(columns=['gender_raw'])
    return df

def validate_data(df):
    """Validate dataset: check sentiment/gender balance and crosstab."""
    logging.info("Sentiment Distribution:\n%s", df['sentiment'].value_counts())
    logging.info("Gender Distribution:\n%s", df['gender'].value_counts())
    logging.info("Gender-Sentiment Crosstab:\n%s", pd.crosstab(df['gender'], df['sentiment']))
    if df['sentiment'].nunique() != 2 or df['gender'].nunique() != 2:
        logging.warning("Data lacks binary sentiment or gender—bias detection may fail.")

def save_data(df, output_path):
    """Save processed data with error handling."""
    try:
        df.to_csv(output_path, index=False)
        logging.info("Saved processed data: %s rows to %s", df.shape[0], output_path)
    except Exception as e:
        logging.error("Error saving data: %s", e)

def main():
    """Main pipeline: load, sample, clean, infer gender, validate, save."""
    setup_logging()
    raw_file = '../data/raw/sentiment140.csv'
    processed_file = '../data/processed/sentiment_with_gender.csv'
    
    df = load_data(raw_file)
    df = sample_data(df, n=20000)
    df = clean_data(df)
    df = infer_gender(df)
    validate_data(df)
    save_data(df, processed_file)

if __name__ == "__main__":
    main()