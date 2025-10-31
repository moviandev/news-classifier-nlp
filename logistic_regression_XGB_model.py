# ======================================================
# 🧠 FAKE NEWS DETECTION — ADVANCED VERSION (TARGET 97 %)
# ======================================================
# BEST ONE IMPLEMENT THIS ONE!
import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import os # Added to create directory if needed

# ------------------------------------------------------
# 1️⃣ SENTIMENT / SUBJECTIVITY / CLICKBAIT
# ------------------------------------------------------
def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

def get_subjectivity(text):
    blob = TextBlob(str(text))
    return blob.sentiment.subjectivity

def is_clickbait(text):
    clickbait_words = [
        "shocking","amazing","incredible","unbelievable","wow",
        "you won't believe","secret","revealed","top","crazy","breaking"
    ]
    text_lower = str(text).lower()
    return int(any(word in text_lower for word in clickbait_words))

# ------------------------------------------------------
# 2️⃣ STRUCTURAL / PSYCHOLOGICAL FEATURES
# ------------------------------------------------------
def count_exclamations(text): return text.count("!")
def count_questions(text): return text.count("?")
def capital_ratio(text):
    words = re.findall(r'\b[A-Z]{2,}\b', text)
    return len(words) / (len(text.split()) + 1e-5)
def text_length(text): return len(text.split())
def lexical_diversity(text):
    words = word_tokenize(text.lower())
    return len(set(words)) / (len(words) + 1e-5)


# ------------------------------------------------------
# MAIN FUNCTION TO RUN THE ENTIRE PIPELINE
# ------------------------------------------------------
def main(training_path, testing_path):
    """
    Main function to run the fake news detection pipeline.
    
    Args:
        training_path (str): Path to the training data CSV.
        testing_path (str): Path to the testing data CSV.
        
    Returns:
        dict: A dictionary containing validation results and the trained model.
    """
    
    print("--- Starting Fake News Detection Pipeline ---")
    nltk.download("punkt")
    output_file_path = './datasets/final_predictions_liniear_regression_xgb.csv'
    
    # --- 0. LOAD DATA ---
    print(f"Loading training data from: {training_path}")
    df_training = pd.read_csv(training_path, sep='\t', header=None)
    df_training.columns = ['label', 'headline'] # Assign column names
    
    print(f"Loading testing data from: {testing_path}")
    df_testing = pd.read_csv(testing_path, sep='\t', header=None)
    df_testing.columns = ['id', 'headline'] # Assign column names
    
    # Store original headlines for final output
    original_test_headlines = df_testing['headline']

    # --- 0B. SPLIT TRAINING DATA ---
    # The original script implies a validation set exists. We create it here.
    print("Splitting training data into train/validation sets...")
    X_train_df, X_validation_df, y_train, y_validation = train_test_split(
        df_training[['headline']],  # Keep as DataFrame
        df_training['label'],
        test_size=0.20, # Common validation split
        random_state=42,
        stratify=df_training['label']
    )
    
    # Create copies to match the original script's variable names
    X_train_processed = X_train_df.copy()
    X_validation_processed = X_validation_df.copy()
    
    # We also need to process the *actual* test data
    X_test_processed = df_testing[['headline']].copy()


    # ------------------------------------------------------
    # 3️⃣ APPLY FEATURES TO PREPROCESSED DATA
    # ------------------------------------------------------
    print("Applying features (Sentiment, Structure, etc.)...")
    # We must apply the same features to train, validation, AND test sets
    for df_ in [X_train_processed, X_validation_processed, X_test_processed]:
        df_["sentiment"] = df_["headline"].apply(get_sentiment)
        df_["subjectivity"] = df_["headline"].apply(get_subjectivity)
        df_["clickbait"] = df_["headline"].apply(is_clickbait)
        df_["excl_marks"] = df_["headline"].apply(count_exclamations)
        df_["quest_marks"] = df_["headline"].apply(count_questions)
        df_["cap_ratio"] = df_["headline"].apply(capital_ratio)
        df_["length"] = df_["headline"].apply(text_length)
        df_["lex_div"] = df_["headline"].apply(lexical_diversity)

    # ------------------------------------------------------
    # 4️⃣ VECTORIZATION (COMBINE BoW + TF-IDF)
    # ------------------------------------------------------
    print("Vectorizing text (BoW + TF-IDF)...")
    bow_vectorizer = CountVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1,3)
    )
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1,3),
        sublinear_tf=True
    )

    # FIT on Train
    X_train_bow = bow_vectorizer.fit_transform(X_train_processed["headline"])
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_processed["headline"])

    # TRANSFORM on Validation
    X_validation_bow = bow_vectorizer.transform(X_validation_processed["headline"])
    X_validation_tfidf = tfidf_vectorizer.transform(X_validation_processed["headline"])

    # TRANSFORM on Test
    X_test_bow = bow_vectorizer.transform(X_test_processed["headline"])
    X_test_tfidf = tfidf_vectorizer.transform(X_test_processed["headline"])

    # Combine both BoW + TF-IDF matrices
    X_train_text = hstack([X_train_bow, X_train_tfidf])
    X_validation_text = hstack([X_validation_bow, X_validation_tfidf])
    X_test_text = hstack([X_test_bow, X_test_tfidf]) # Added for test set

    # ------------------------------------------------------
    # 5️⃣ ADD STRUCTURAL FEATURES
    # ------------------------------------------------------
    print("Combining text vectors with structural features...")
    extra_feats = [
        "sentiment","subjectivity","clickbait",
        "excl_marks","quest_marks","cap_ratio",
        "length","lex_div"
    ]
    train_meta = X_train_processed[extra_feats].values
    val_meta = X_validation_processed[extra_feats].values
    test_meta = X_test_processed[extra_feats].values # Added for test set

    X_train_combined = hstack([X_train_text, train_meta])
    X_validation_combined = hstack([X_validation_text, val_meta])
    X_test_combined = hstack([X_test_text, test_meta]) # Added for test set

    # ------------------------------------------------------
    # 6️⃣ GRID SEARCH FOR BEST LOGISTIC REGRESSION
    # ------------------------------------------------------
    print("Running GridSearchCV for Logistic Regression...")
    param_grid = {"C": [0.5, 1, 3, 5, 10], "solver": ["lbfgs"], "penalty": ["l2"]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train_combined, y_train)
    best_lr = grid.best_estimator_
    print("🧠 Best LogisticRegression params:", grid.best_params_)

    # ------------------------------------------------------
    # 7️⃣ DEFINE OTHER MODELS
    # ------------------------------------------------------
    svm = LinearSVC(C=1)
    xgb = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # ------------------------------------------------------
    # 8️⃣ ENSEMBLE (LOGISTIC + SVM + XGBOOST)
    # ------------------------------------------------------
    ensemble = VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("svm", svm),
            ("xgb", xgb)
        ],
        voting="hard"
    )


    # ------------------------------------------------------
    # 9️⃣ TRAIN & EVALUATE (on Validation Set)
    # ------------------------------------------------------
    print("\n🚀 Training ENSEMBLE MODEL...")
    ensemble.fit(X_train_combined, y_train)
    y_pred = ensemble.predict(X_validation_combined)

    acc = accuracy_score(y_validation, y_pred)
    report = classification_report(y_validation, y_pred)
    print(f"\n✅ Final Ensemble *Validation* Accuracy: {acc:.4f}")
    print("\n📊 *Validation* Classification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(y_validation, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix — Ensemble Model (BoW + TFIDF + Structure)')
    plt.show()
    
    # ------------------------------------------------------
    # 10. PREDICT ON FINAL TEST SET & SAVE
    # ------------------------------------------------------
    print(f"\n🚀 Generating final predictions on test data...")
    final_predictions = ensemble.predict(X_test_combined)
    
    # Create the final DataFrame
    df_final_output = pd.DataFrame({
        'headline': original_test_headlines,
        'prediction': final_predictions
    })
    
    # Save to the requested file
    print(f"💾 Saving predictions to {output_file_path}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Save as tab-separated value file (like the inputs)
    df_final_output.to_csv(output_file_path, index=False, sep='\t')
    
    print("✅ Process complete.")

    # Return key results
    results = {
        'validation_accuracy': acc,
        'validation_report': report,
        'trained_model': ensemble,
        'best_lr_params': grid.best_params_,
        'output_file_path': output_file_path
    }
    return results