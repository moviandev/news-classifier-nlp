def fake_news_detection_logistic_regression(training_path: str, testing_path: str = None):
    """
    🧠 FAKE NEWS DETECTION — Logistic Regression (Embeddings + TF-IDF + Emotion)
    
    Args:
        training_path (str): Path to training dataset (label \t headline)
        testing_path (str, optional): Path to testing dataset (headline only)
    
    Saves:
        ./datasets/final_predictions_logistic_regression_embeddings.csv
    """
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from scipy.sparse import hstack
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import os

    # ===========================================
    # 1️⃣ Load and prepare data
    # ===========================================
    train_df = pd.read_csv(training_path, sep="\t", header=None, names=["label", "headline"])
    X = train_df[["headline"]]
    y = train_df["label"]

    # Split small portion for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Basic text normalization placeholders (to avoid dependency on missing funcs)
    def clean_html(x): return str(x)
    def normalize_text(x): return str(x)
    def remove_stopwords(x): return str(x)
    def stem_text(x): return str(x)

    X_train_processed = X_train.copy()
    X_validation_processed = X_val.copy()

    for df_ in [X_train_processed, X_validation_processed]:
        df_["headline"] = df_["headline"].apply(clean_html)
        df_["headline"] = df_["headline"].apply(normalize_text)
        df_["headline"] = df_["headline"].apply(remove_stopwords)
        df_["headline"] = df_["headline"].apply(stem_text)

    # ===========================================
    # 2️⃣ Emotion Lexicons — anger, disgust, surprise
    # ===========================================
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()

    emotion_seeds = {
        "anger": ["anger","rage","furious","outrage","hate","mad","irritated","annoyed"],
        "disgust": ["disgust","gross","nasty","horrible","filthy","vile","repulsive","sick"],
        "surprise": ["surprise","shock","amazing","unexpected","astonishing","wow","sudden"]
    }

    def get_synonyms(word):
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_"," ").lower()
                if len(name) > 2:
                    syns.add(name)
        return syns

    expanded_emotions = {}
    for emo, words in emotion_seeds.items():
        lexicon = set(words)
        for w in words:
            lexicon.update(get_synonyms(w))
        expanded_emotions[emo] = lexicon

    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in str(text).lower().split()]

    def count_emotion(text, emotion_set):
        lemmas = lemmatize_text(text)
        return sum(l in emotion_set for l in lemmas)

    for emo, lexicon in expanded_emotions.items():
        X_train_processed[emo] = X_train_processed["headline"].apply(lambda t: count_emotion(t, lexicon))
        X_validation_processed[emo] = X_validation_processed["headline"].apply(lambda t: count_emotion(t, lexicon))

    # ===========================================
    # 3️⃣ Sentence Embeddings
    # ===========================================
    print("🔍 Generating embeddings (MiniLM)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    X_train_emb = embedder.encode(X_train_processed["headline"].tolist(), show_progress_bar=True)
    X_val_emb   = embedder.encode(X_validation_processed["headline"].tolist(), show_progress_bar=True)

    # ===========================================
    # 4️⃣ TF-IDF Vectorization
    # ===========================================
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1,2),
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_processed["headline"])
    X_val_tfidf   = vectorizer.transform(X_validation_processed["headline"])

    # ===========================================
    # 5️⃣ Combine features: embeddings + TF-IDF + emotions
    # ===========================================
    scaler = StandardScaler()
    emotion_feats = ["anger","disgust","surprise"]
    X_train_scaled = scaler.fit_transform(X_train_processed[emotion_feats])
    X_val_scaled   = scaler.transform(X_validation_processed[emotion_feats])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_val_poly   = poly.transform(X_val_scaled)

    X_train_final = np.hstack([X_train_emb, X_train_poly])
    X_val_final   = np.hstack([X_val_emb, X_val_poly])

    X_train_combined = hstack([X_train_tfidf, X_train_final])
    X_val_combined   = hstack([X_val_tfidf, X_val_final])

    print("✅ Final combined feature matrix shape:", X_train_combined.shape)

    # ===========================================
    # 6️⃣ Logistic Regression (Optimized)
    # ===========================================
    model = LogisticRegression(
        C=2.5,
        solver="lbfgs",
        max_iter=3000,
        penalty="l2",
        class_weight="balanced",
        n_jobs=-1
    )

    print("\n🚀 Training Logistic Regression on hybrid features...")
    model.fit(X_train_combined, y_train)

    # ===========================================
    # 7️⃣ Evaluation
    # ===========================================
    y_pred = model.predict(X_val_combined)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n✅ Final Validation Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix — Logistic Regression (Embeddings + TF-IDF + Emotion)')
    plt.show()

    # ===========================================
    # 8️⃣ Predict on Test Data (optional)
    # ===========================================
    if testing_path:
        test_df = pd.read_csv(testing_path, sep="\t", header=None, names=["headline"])

        for emo, lexicon in expanded_emotions.items():
            test_df[emo] = test_df["headline"].apply(lambda t: count_emotion(t, lexicon))

        test_emb = embedder.encode(test_df["headline"].tolist(), show_progress_bar=True)
        test_tfidf = vectorizer.transform(test_df["headline"])
        test_scaled = scaler.transform(test_df[emotion_feats])
        test_poly = poly.transform(test_scaled)

        test_final = np.hstack([test_emb, test_poly])
        test_combined = hstack([test_tfidf, test_final])

        test_pred = model.predict(test_combined)

        os.makedirs("./datasets", exist_ok=True)
        output_path = "./datasets/final_predictions_logistic_regression_embeddings.csv"
        test_df["predicted_label"] = test_pred
        test_df.to_csv(output_path, index=False)

        print(f"\n✅ Predictions saved to '{output_path}'")

    return model, acc