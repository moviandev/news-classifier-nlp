import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

def clean_text(text):
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  translator = str.maketrans('', '', string.punctuation)
  # Removing punctuation
  text_no_punct = text.translate(translator)
    
  # Tokenize (split into words)
  tokens = word_tokenize(text_no_punct)
  
  # Remove stopwords and lemmatize
  cleaned_tokens = []
  for word in tokens:
      if word.lower() not in stop_words: # Check if it's a stopword
          lemma = lemmatizer.lemmatize(word) # Get root form
          cleaned_tokens.append(lemma)
          
  # Join back to a string
  return " ".join(cleaned_tokens)

def handle_clean_up_df(df):
  df.columns = ['label', 'text']
  df['text'] = df['text'].fillna('')
  df['cleaned_text'] = df['text'].apply(clean_text)
  return df

def split_training_test_data(df):
  X_train_full = df['cleaned_text']
  y_train_full = df['label'].astype(int)
  X_train, X_val, y_train, y_val = train_test_split(
      X_train_full, 
      y_train_full, 
      test_size=0.2,
      random_state=42,
      stratify=y_train_full
  )
  return (X_train_full, y_train_full, X_train, X_val, y_train, y_val)

def vectorize_data(X_train, X_val = None):
  # Renamed for clarity
  vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
  )

  # Fit and transform the training data
  X_train_tfidf = vectorizer.fit_transform(X_train)

  X_val_tfidf = None
  if X_val is not None and len(X_val) > 0:
    # Only transform the validation data
    X_val_tfidf = vectorizer.transform(X_val)
    print(X_val_tfidf.shape)
    
  # Return the vectorizer AND the transformed data
  return (vectorizer, X_train_tfidf, X_val_tfidf)

def train_model_1(X_train_tfidf_val, y_train):
  model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    n_estimators=100
  )
  model.fit(X_train_tfidf_val, y_train)
  return model

def validate_model(model, X_val_tfidf, y_val, y_train_full):
  val_predictions = model.predict(X_val_tfidf)

  accuracy = accuracy_score(y_val, val_predictions)
  print(f"Accuracy: {accuracy:.4f}")

  print("\nClassification Report:")
  print(classification_report(y_val, val_predictions))

  print("\nConfusion Matrix:")
  cm = confusion_matrix(y_val, val_predictions)
  print(cm)

  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
              xticklabels=np.unique(y_train_full), 
              yticklabels=np.unique(y_train_full))
  plt.title('Confusion Matrix (Validation Set)')
  plt.ylabel('Actual Label')
  plt.xlabel('Predicted Label')
  plt.show()

def save_into_csv(df, predictions):
  submission_df = pd.DataFrame({
      'id': df['id'],
      'text': df['text'],
      'label': predictions
  })

  submission_filename = './datasets/final_predictions_random_forest.csv'
  submission_df.to_csv(submission_filename, index=False)

  print(f"\nSubmission file saved as '{submission_filename}'")
  print("\nHead of the submission file:")
  print(submission_df.head())

def random_forest(training_df, testing_df = None):
  # Clean DF training
  training_df_clean = handle_clean_up_df(training_df)

  # Split data for training our model
  X_train_full, y_train_full, X_train, X_val, y_train, y_val = split_training_test_data(training_df_clean)

  # Vectorize validation data
  _, X_train_tfidf_val, X_val_tfidf = vectorize_data(X_train, X_val)
  
  # Validation Model
  validation_model = train_model_1(X_train_tfidf_val, y_train)

  # Evaluation of validation model
  validate_model(validation_model, X_val_tfidf, y_val, y_train_full)
  
  # Vectorize Full Data
  full_data_vectorizer, X_train_tfidf_full, _ = vectorize_data(X_train_full)
  
  # Train Full Model
  model = train_model_1(X_train_tfidf_full, y_train_full)  
  
  if testing_df is not None:
    # Clean testing DF
    testing_df_clean = testing_df.copy()
    testing_df_clean.columns = ['id', 'text']
    testing_df_clean['text'] = testing_df_clean['text'].fillna('')
    testing_df_clean['cleaned_text'] = testing_df_clean['text'].apply(clean_text)

    # Vectorize full data
    X_testing_data = testing_df_clean['cleaned_text']
    X_testing_tfidf = full_data_vectorizer.transform(X_testing_data)

    # predictions
    predictions = model.predict(X_testing_tfidf)

    # save into CSV
    save_into_csv(testing_df_clean, predictions)