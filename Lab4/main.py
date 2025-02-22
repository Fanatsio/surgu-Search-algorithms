import os
import json
import nltk
import string
import time  # Импортируем модуль time
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')
stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words("russian"))

def preprocess_text(text):
    """Препроцессинг текста: приведение к нижнему регистру, удаление знаков препинания, токенизация, фильтрация стоп-слов и стемминг."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language="russian")
    return [stemmer.stem(token) for token in tokens if token not in stop_words]

def generate_documents(folder="./Lab4/docs"):
    """Генерирует текстовые документы в указанной папке, если они не существуют."""
    os.makedirs(folder, exist_ok=True)
    if any(os.scandir(folder)):
        return  
    
    texts = {
        "doc1.txt": "Машинное обучение развивается быстро. Оно позволяет компьютерам учиться на данных. Многие отрасли его используют. Алгоритмы анализируют огромные массивы информации. Это открывает новые возможности в науке и бизнесе.",
        "doc2.txt": "Обработка естественного языка – ключевая область ИИ. Она помогает машинам понимать речь и текст. Это применяется в чат-ботах и переводчиках. Современные модели способны понимать контекст. Это делает взаимодействие с машинами удобнее.",
        "doc3.txt": "Индексация текста улучшает производительность поисковых систем. Она делает поиск информации эффективным. Без индексации системы работали бы медленнее. Современные алгоритмы оптимизируют хранение данных. Это снижает нагрузку на серверы.",
        "doc4.txt": "Глубокое обучение использует нейросети. Оно достигло прорывов в областях, включая распознавание изображений. Такие технологии помогают в медицине. Диагностика заболеваний становится точнее. Это спасает жизни.",
        "doc5.txt": "Наука о данных – это междисциплинарная область. Она объединяет статистику, программирование и знания о предметной области. Анализ данных помогает принимать решения. Компании используют его для прогнозирования трендов. Это улучшает бизнес-процессы.",
        "doc6.txt": "Искусственный интеллект влияет на повседневную жизнь. Он используется в голосовых помощниках и рекомендательных системах. AI помогает автоматизировать рутинные задачи. Это увеличивает производительность труда. Также AI применяется в медицине и финансах.",
        "doc7.txt": "Компьютерное зрение позволяет машинам интерпретировать изображения. Оно используется в медицине и системах безопасности. Камеры с AI могут анализировать дорожную ситуацию. Это помогает предотвращать аварии. Также технологии применяются в промышленности.",
        "doc8.txt": "Аналитика больших данных помогает принимать решения. Компании используют её для прогнозов и анализа тенденций. Современные системы анализируют поведение пользователей. Это позволяет улучшать сервисы и персонализировать предложения. Бизнес получает конкурентное преимущество.",
        "doc9.txt": "Кибербезопасность крайне важна в цифровую эпоху. Шифрование и аутентификация защищают конфиденциальные данные. Хакеры постоянно совершенствуют атаки. Поэтому требуется разработка новых методов защиты. Компании инвестируют в безопасность своих пользователей.",
        "doc10.txt": "Облачные вычисления обеспечивают масштабируемую инфраструктуру. Бизнес использует их для хранения данных и обработки информации. Облака позволяют экономить ресурсы. Они обеспечивают гибкость и надежность. Компании переходят на облачные технологии для повышения эффективности."
    }
    for filename, text in texts.items():
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
            f.write(text)

def load_documents(folder="./Lab4/docs"):
    """Загружает текстовые документы из указанной папки."""
    return {
        filename: open(os.path.join(folder, filename), "r", encoding="utf-8").read()
        for filename in os.listdir(folder) if filename.endswith(".txt")
    }

def build_inverted_index(documents):
    """Создает инвертированный индекс для заданных документов."""
    inverted_index = defaultdict(lambda: defaultdict(lambda: {"freq": 0, "positions": []}))
    
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for position, token in enumerate(tokens):
            inverted_index[token][doc_id]["freq"] += 1
            inverted_index[token][doc_id]["positions"].append(position)
    
    return inverted_index

def save_index(index, filename="./Lab4/inverted_index.json"):
    """Сохраняет инвертированный индекс в JSON файл."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4, ensure_ascii=False)

def load_index(filename="./Lab4/inverted_index.json"):
    """Загружает инвертированный индекс из JSON файла."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def search(query, index):
    """Ищет документы по запросу в инвертированном индексе и возвращает результаты."""
    query_tokens = preprocess_text(query)
    results = defaultdict(lambda: {"score": 0, "positions": defaultdict(list)})
    
    for token in query_tokens:
        if token in index:
            for doc_id, data in index[token].items():
                results[doc_id]["score"] += data["freq"]
                results[doc_id]["positions"][token] = data["positions"]
    
    return sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

def search_without_index(query, documents):
    """Поиск документов по запросу без использования индекса (прямой перебор)."""
    query_tokens = preprocess_text(query)
    results = defaultdict(lambda: {"score": 0, "positions": defaultdict(list)})
    
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for token in query_tokens:
            if token in tokens:
                results[doc_id]["score"] += 1
                results[doc_id]["positions"][token].append(tokens.index(token))
    
    return sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

def get_input(prompt, default=None):
    user_input = input(prompt)
    return user_input if user_input else default

if __name__ == "__main__":
    generate_documents()
    documents = load_documents()
    index = build_inverted_index(documents)
    save_index(index)
    
    loaded_index = load_index()
    query = get_input("\nВведите поисковый запрос (по умолчанию 'обучение нейросети') ---→ ", "обучение нейросети")
    
    # Поиск с использованием индекса
    start_time = time.time()
    results_with_index = search(query, loaded_index)
    end_time = time.time()
    print("\nРезультаты поиска с использованием индекса:")
    print(f"Время поиска: {end_time - start_time:.6f} секунд")
    for doc, data in results_with_index:
        print(f"{doc}: релевантность {data['score']}, позиции: {dict(data['positions'])}")

    # Поиск без использования индекса
    start_time = time.time()
    results_without_index = search_without_index(query, documents)
    end_time = time.time()
    print("\nРезультаты поиска без использования индекса:")
    print(f"Время поиска: {end_time - start_time:.6f} секунд")
    for doc, data in results_without_index:
        print(f"{doc}: релевантность {data['score']}, позиции: {dict(data['positions'])}")
