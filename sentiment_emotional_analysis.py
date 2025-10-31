def sentiment_emotional_analysis(training_path: str, testing_path: str = None):
    """
    🧩 SENTIMENT & EMOTIONAL WORD ANALYSIS — FAKE vs REAL
    Works with tab-separated datasets like:
      training: label \t headline
      testing: headline only
    """
    import pandas as pd
    import numpy as np
    from textblob import TextBlob
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ===============================
    # ✅ 1. Load training data
    # ===============================
    train_df = pd.read_csv(training_path, sep="\t", header=None, names=["label", "headline"])
    print(f"✅ Training data loaded: {train_df.shape[0]} samples")

    # ===============================
    # 🧠 2. Sentiment metrics
    # ===============================
    train_df["sentiment"] = train_df["headline"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    train_df["subjectivity"] = train_df["headline"].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )

    # ===============================
    # ❤️ 3. Emotional word counts
    # ===============================
    positive_words = [
        "amazing", "great", "wonderful", "beautiful", "fantastic", "incredible", "happy", "love"
    ]
    negative_words = [
        "terrible", "horrible", "worst", "disaster", "angry", "hate", "sad", "shame"
    ]

    def count_emotional_words(text):
        text_lower = str(text).lower()
        return sum(word in text_lower for word in positive_words + negative_words)

    train_df["emotional_word_count"] = train_df["headline"].apply(count_emotional_words)

    # ===============================
    # 📊 4. Aggregated statistics
    # ===============================
    group_stats = train_df.groupby("label").agg({
        "sentiment": ["mean", "std"],
        "subjectivity": ["mean", "std"],
        "emotional_word_count": ["mean", "sum"]
    }).reset_index()

    group_stats.columns = [
        "label",
        "mean_sentiment", "std_sentiment",
        "mean_subjectivity", "std_subjectivity",
        "mean_emotional_words", "total_emotional_words"
    ]
    group_stats["label"] = group_stats["label"].map({0: "Fake News", 1: "Real News"})

    print("\n📊 Sentiment & Emotion Analysis (Fake vs Real):\n")
    print(group_stats)

    # ===============================
    # 🎨 5. Visualization
    # ===============================
    plt.figure(figsize=(10, 5))
    sns.barplot(data=group_stats, x="label", y="mean_sentiment", palette="coolwarm")
    plt.title("Average Sentiment Polarity — Fake vs Real News")
    plt.ylabel("Mean Sentiment (−1 Negative → +1 Positive)")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=group_stats, x="label", y="mean_subjectivity", palette="Purples")
    plt.title("Average Subjectivity — Fake vs Real News")
    plt.ylabel("Mean Subjectivity (0 Objective → 1 Subjective)")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=group_stats, x="label", y="mean_emotional_words", palette="Greens")
    plt.title("Average Emotional Words per Headline — Fake vs Real News")
    plt.ylabel("Mean Emotional Word Count")
    plt.show()

    # ===============================
    # 🧩 6. Optional: Apply to testing set
    # ===============================
    if testing_path:
        test_df = pd.read_csv(testing_path, sep="\t", header=None, names=["headline"])
        print(f"✅ Testing data loaded: {test_df.shape[0]} samples")

        # Compute same features
        test_df["sentiment"] = test_df["headline"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        test_df["subjectivity"] = test_df["headline"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        test_df["emotional_word_count"] = test_df["headline"].apply(count_emotional_words)

        # Simple heuristic prediction
        test_df["predicted_label"] = np.where(test_df["sentiment"] > 0, 1, 0)

        # Save predictions
        test_df.to_csv("./datasets/final_predictions_sentiment_emotional.csv", index=False)
        print("\n✅ Predictions saved to 'final_predictions_sentiment_emotional.csv'")

    return group_stats