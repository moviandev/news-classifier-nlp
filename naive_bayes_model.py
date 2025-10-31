# Importando bibliotecas básicas
import pandas as pd         # Para manipulação de dados
import numpy as np          # Para operações matemáticas
import re                   # Para expressões regulares (limpeza de texto)

# Importando bibliotecas de NLP e ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Baixando recursos necessários do NLTK (stopwords, wordnet)
# É bom fazer isso no escopo global para que seja executado quando o módulo for importado
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# --- Função Auxiliar de Limpeza ---
# Definida fora da main para que possa ser acessada globalmente, se necessário,
# e mantém a função 'main' mais limpa.

def clean_text(text):
    """
    Função para limpar o texto:
    1. Converte para minúsculas
    2. Remove pontuação, números e caracteres especiais
    3. Tokeniza (divide em palavras)
    4. Remove stopwords
    5. Lemmatiza (reduz à forma raiz)
    """
    # Garante que a entrada seja uma string e converte para minúsculas
    text = str(text).lower()

    # 2. Remove pontuação, números e caracteres especiais
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Tokenize (split the sentence into words)
    words = text.split()

    # 4. Remove stopwords (common words that don't add meaning)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 5. Lemmatize words (reduce them to their root form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # 6. Join words back into one string
    return ' '.join(words)


# --- Função Principal ---

def main(training_path, testing_path):
    """
    Executa o pipeline completo de NLP:
    1. Carrega e processa dados de TREINO.
    2. Treina um vetorizador TF-IDF.
    3. Treina um modelo Naive Bayes.
    4. Carrega e processa dados de TESTE.
    5. Faz previsões nos dados de TESTE.
    6. Salva as previsões em um arquivo CSV.
    """
    
    print(f"Iniciando o pipeline...")
    print(f"Carregando dados de treino de: {training_path}")

    # --- 1. Processamento dos Dados de Treino ---
    
    # Lê o arquivo CSV de treino
    try:
        df_train = pd.read_csv(training_path, sep='\t', engine='python', names=['label', 'title'])
    except FileNotFoundError:
        print(f"Erro: Arquivo de treino não encontrado em {training_path}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo de treino: {e}")
        return

    print("\n--- Dados de Treino (Head) ---")
    print(df_train.head())
    print(f"Shape dos dados de treino: {df_train.shape}")
    print(f"Valores nulos (treino): \n{df_train.isnull().sum()}")
    print('-'*30)

    # Aplica a limpeza de texto na coluna 'title'
    print("Limpando texto de treino...")
    df_train['clean_title'] = df_train['title'].apply(clean_text)
    
    print("\n--- Exemplo de Limpeza (Treino) ---")
    print(df_train[['title', 'clean_title']].head())
    print('-'*30)

    # --- 2. Vetorização (TF-IDF) ---
    
    print("Criando e treinando o vetorizador TF-IDF...")
    # Cria o vetorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)

    # Treina (fit) e transforma (transform) os dados de *treino*
    X_train_tfidf = vectorizer.fit_transform(df_train['clean_title'])
    y_train = df_train['label']

    print(f"Shape da matriz TF-IDF de treino: {X_train_tfidf.shape}")
    print('-'*30)

    # --- 3. Treinamento do Modelo ---
    
    print("Treinando o modelo Naive Bayes...")
    # Cria o modelo
    nb_model = MultinomialNB()

    # Treina o modelo com TODOS os dados de treino
    nb_model.fit(X_train_tfidf, y_train)
    
    print("✅ Modelo treinado com sucesso.")
    print('-'*30)

    # --- 4. Processamento dos Dados de Teste ---
    
    print(f"Carregando dados de teste de: {testing_path}")
    
    # Carrega os dados de teste (sem rótulos, assumindo uma coluna 'title')
    try:
        df_test = pd.read_csv(testing_path, sep='\t', engine='python', names=['title'])
    except FileNotFoundError:
        print(f"Erro: Arquivo de teste não encontrado em {testing_path}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo de teste: {e}")
        return

    print("\n--- Dados de Teste (Head) ---")
    print(df_test.head())
    print(f"Shape dos dados de teste: {df_test.shape}")
    print(f"Valores nulos (teste): \n{df_test.isnull().sum()}")
    print('-'*30)
    
    # Salva os títulos originais para o arquivo final
    original_titles = df_test['title']

    # Limpa os dados de teste
    print("Limpando texto de teste...")
    df_test['clean_title'] = df_test['title'].apply(clean_text)
    
    # --- 5. Geração de Previsões ---

    print("Vetorizando dados de teste...")
    # Apenas transforma (transform) os dados de teste usando o vetorizador *já treinado*
    X_test_tfidf = vectorizer.transform(df_test['clean_title'])

    print("Fazendo previsões...")
    # Faz as previsões
    predictions = nb_model.predict(X_test_tfidf)
    
    # --- 6. Salvando os Resultados ---
    
    # Cria um DataFrame final com o texto original e as previsões
    df_final = pd.DataFrame({
        'title': original_titles,
        'prediction': predictions
    })

    output_path = './datasets/final_predictions_naive_bayes.csv'
    print(f"Salvando previsões em: {output_path}")

    # Salva o arquivo final
    try:
        df_final.to_csv(output_path, index=False, sep='\t')
        print(f"✅ Arquivo de previsões salvo com sucesso em {output_path}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de previsões: {e}")
        
    print("Pipeline concluído.")