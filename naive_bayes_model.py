# -----------------------------------------------------------------
# main_pipeline.py
# -----------------------------------------------------------------

# Importando bibliotecas básicas
import pandas as pd         # Para manipulação de dados
import numpy as np          # Para operações matemáticas
import re                   # Para expressões regulares (limpeza de texto)

# Importando bibliotecas de NLP e ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split  # Importado para validação
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report  # Importado para validação

# Baixando recursos necessários do NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# --- Função Auxiliar de Limpeza ---

def clean_text(text):
    """
    Função para limpar o texto:
    1. Converte para minúsculas
    2. Remove pontuação, números e caracteres especiais
    3. Tokeniza (divide em palavras)
    4. Remove stopwords
    5. Lemmatiza (reduz à forma raiz)
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# --- Função Principal ---

def main(training_path, testing_path):
    """
    Executa o pipeline completo de NLP:
    1. Carrega e processa dados de TREINO.
    2. **Executa uma VALIDAÇÃO (split 80/20) nos dados de treino para mostrar as métricas.**
    3. **Retreina o modelo e o vetorizador com 100% dos dados de treino.**
    4. Carrega e processa dados de TESTE (sem labels).
    5. Faz previsões nos dados de TESTE.
    6. Salva as previsões em um arquivo CSV.
    """
    
    print(f"Iniciando o pipeline...")
    print(f"Carregando dados de treino de: {training_path}")

    # --- 1. Processamento dos Dados de Treino ---
    
    try:
        df_train = pd.read_csv(training_path, sep='\t', engine='python', names=['label', 'title'])
    except FileNotFoundError:
        print(f"Erro: Arquivo de treino não encontrado em {training_path}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo de treino: {e}")
        return

    print("Limpando texto de treino...")
    df_train['clean_title'] = df_train['title'].apply(clean_text)
    print("Limpeza de treino concluída.")
    print('-'*30)

    # --- 2. Etapa de Validação (Para obter as Métricas) ---
    
    print("Iniciando VALIDAÇÃO do modelo...")
    
    # Define X e y para validação
    X_val = df_train['clean_title']
    y_val = df_train['label']

    # Separa os dados de treino em sub-conjuntos de treino e teste (para validação)
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
        X_val,
        y_val,
        test_size=0.2,
        random_state=42,
        stratify=y_val
    )

    # Cria um vetorizador *apenas para a validação*
    vectorizer_val = TfidfVectorizer(max_features=10000)
    
    # Treina o vetorizador de validação
    X_train_val_tfidf = vectorizer_val.fit_transform(X_train_val)
    X_test_val_tfidf = vectorizer_val.transform(X_test_val)

    # Treina o modelo de validação
    nb_model_val = MultinomialNB()
    nb_model_val.fit(X_train_val_tfidf, y_train_val)

    # Faz previsões de validação
    y_pred_val = nb_model_val.predict(X_test_val_tfidf)

    # **Exibe as Métricas (como no seu script original)**
    accuracy_nb = accuracy_score(y_test_val, y_pred_val)
    print(f"--- Métricas de Validação (Baseado em 20% dos dados de treino) ---")
    print(f"✅ Acurácia Naive Bayes: {accuracy_nb:.4f}")
    print("\n📋 Relatório de Classificação (Validação):")
    print(classification_report(y_test_val, y_pred_val))
    print('-'*30)
    
    # --- 3. Treinamento Final (Com 100% dos dados) ---
    
    print("Retreinando o modelo com 100% dos dados de treino para a previsão final...")

    # Cria o vetorizador FINAL
    final_vectorizer = TfidfVectorizer(max_features=10000)
    
    # Treina o vetorizador FINAL com TODOS os dados de treino
    X_train_full_tfidf = final_vectorizer.fit_transform(df_train['clean_title'])
    y_train_full = df_train['label']

    # Cria o modelo FINAL
    final_model = MultinomialNB()
    
    # Treina o modelo FINAL com TODOS os dados de treino
    final_model.fit(X_train_full_tfidf, y_train_full)
    
    print("✅ Modelo final treinado.")
    print('-'*30)

    # --- 4. Processamento dos Dados de Teste (Sem Labels) ---
    
    print(f"Carregando dados de teste (para previsão) de: {testing_path}")
    
    try:
        df_test = pd.read_csv(testing_path, sep='\t', engine='python', names=['title'])
    except FileNotFoundError:
        print(f"Erro: Arquivo de teste não encontrado em {testing_path}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo de teste: {e}")
        return

    # Salva os títulos originais para o arquivo final
    original_titles = df_test['title']

    # Limpa os dados de teste
    print("Limpando texto de teste...")
    df_test['clean_title'] = df_test['title'].apply(clean_text)
    
    # --- 5. Geração de Previsões Finais ---

    print("Vetorizando dados de teste (usando o vetorizador final)...")
    # Usa o 'final_vectorizer' (treinado com 100% dos dados) para transformar os dados de teste
    X_test_final_tfidf = final_vectorizer.transform(df_test['clean_title'])

    print("Fazendo previsões finais...")
    # Usa o 'final_model' para prever
    final_predictions = final_model.predict(X_test_final_tfidf)
    
    # --- 6. Salvando os Resultados ---
    
    output_path = './datasets/final_predictions_naive_bayes.csv'
    
    df_final = pd.DataFrame({
        'title': original_titles,
        'prediction': final_predictions
    })

    print(f"Salvando previsões em: {output_path}")
    
    try:
        df_final.to_csv(output_path, index=False, sep='\t')
        print(f"✅ Arquivo de previsões salvo com sucesso em {output_path}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de previsões: {e}")
        
    print("Pipeline concluído.")