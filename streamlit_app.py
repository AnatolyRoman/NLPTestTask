import streamlit as st
import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import nltk
from nltk.corpus import stopwords
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
import torch
import keras


# Проверка наличия файла
if st._is_running_with_streamlit:
    import os

    file_path = os.path.join(os.path.dirname(__file__), "styles.css")

    if os.path.isfile(file_path):
        with open(file_path) as f:
            styles = f.read()

        # Применение стилей с помощью markdown
        st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    else:
        st.write("Файл styles.css не найден.")

# Путь к файлу модели
model_path = "FirrstModel/emotion_model.h5"

# Загрузка модели
model = keras.models.load_model(model_path)


# Set Streamlit configurations
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="NLP: Test Task", page_icon=":smiley:", layout="wide")

# Header
st.markdown('''
    <h1 style="text-align: left; color: #FFC300 ; font-size: 100px;">NLP engineer</h1>
    <hr style="border-top: 5px solid #A09D0E   ; margin-top: 30px; margin-bottom: 30px;">
    <h1 style="text-align: left; color: #FFC300; font-size: 70px;">Test Task</h1>
''', unsafe_allow_html=True)

# Task description
st.markdown(
    f'<hr style="border-top: 3px solid #A09D0E  ; margin-top: 30px; margin-bottom: 30px;">',
    unsafe_allow_html=True)

st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 40px;">Задание</h1>
    <h1 style="text-align: left; color: #32CD32; font-size: 25px;"></h1>
''', unsafe_allow_html=True)

st.markdown('''
    <p style="text-align: left; color: #D8D8D8; font-size: 20px;">
      Используя <a href="https://huggingface.co/datasets/dair-ai/emotion/viewer/split/train?row=0" style="color: #A3E4D7; text-decoration: underline;">данную выборку</a> выполните следующие пунĸты:
    </p>

    <ul style="color: #FFFFFF;">
      <li>Проанализируйте данные (EDA). Проĸомментируйте свои действия.</li>
      <li>Постройте модель, решающую задачу ĸлассифиĸации, на основе данных. Вы можете не ограничиваться одной моделью, но обоснуйте, ĸаĸую из обученных моделей Вы бы выбрали в итоге и почему.</li>
      <li>Оцените модель при помощи подходящих метриĸ.</li>
      <li>Попробуйте сжать размер модели, не теряя ĸатастрофичесĸи в точности по выбранной Вами метриĸе.</li>
      <li>Напишите вывод из проведенного эĸсперимента, ĸаĸ Вы видите развитие данного эĸсперимента.</li>
    </ul>
''', unsafe_allow_html=True)

st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 40px;">EDA (Exploratory Data Analysis) </h1>
    <h1 style="text-align: left; color: #32CD32; font-size: 25px;"></h1>
''', unsafe_allow_html=True)

st.markdown('''
    <p style="text-align: left; color: #D8D8D8; font-size: 20px;">
  EDA (Exploratory Data Analysis) — анализ исследовательских данных, является важной частью процесса разработки модели машинного обучения. В EDA исследуются данные, выполняется их визуализация и изучаются основные характеристики для получения более глубокого понимания набора данных.
''', unsafe_allow_html=True)


st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 25px;">Распределение длины текстов </h1>
''', unsafe_allow_html=True)
# Загрузка данных из файла CSV
train_df = pd.read_csv('train_data.csv')

st.markdown('''
    <p style="text-align: left; color: #D8D8D8; font-size: 20px;">
  График распределения длины текстов помогает увидеть, как тексты в обучающем наборе данных распределены по длине. Это помогает понять, какие длины текстов наиболее распространены и может влиять на выбор методов обработки и анализа текста.
''', unsafe_allow_html=True)


# Визуализация распределения длины текстов
train_df['length_of_text'] = [len(i.split(' ')) for i in train_df['text']]
fig = px.histogram(train_df['length_of_text'], marginal='box',
                   labels={"value": "Длина текста"})
fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))

# Вставка в Streamlit
st.plotly_chart(fig)

# Расчет частоты слов
FreqOfWords = train_df['text'].str.split(expand=True).stack().value_counts()
FreqOfWords_top200 = FreqOfWords[:200]

st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 25px;">Частота слов в обучающем наборе данных</h1>
''', unsafe_allow_html=True)

st.markdown('''
    <p style="text-align: left; color: #D8D8D8; font-size: 20px;">График частоты слов в обучающем наборе данных показывает, какие слова наиболее часто встречаются в текстах. Это позволяет идентифицировать наиболее частые и значимые слова, которые могут быть полезны при анализе и классификации текстов. Такой график помогает визуализировать распределение слов и может быть использован для принятия решений по предобработке и отбору признаков для моделирования текстовых данных.
''', unsafe_allow_html=True)


# Загрузка данных из файла CSV
# Визуализация частоты слов
fig = px.treemap(FreqOfWords_top200, path=[FreqOfWords_top200.index], values=0)

fig.update_traces(textinfo="label+value")

# Вставка в Streamlit
st.plotly_chart(fig)


st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 25px;">Распределение меток классов в обучающем наборе данных</h1>
''', unsafe_allow_html=True)

st.markdown('''
    <p style="text-align: left; color: #D8D8D8; font-size: 20px;">
    График распределения меток классов в обучающем наборе данных показывает, как часто каждый класс встречается в данных. Это полезно для понимания баланса классов в наборе данных. Неравномерное распределение классов может указывать на проблемы, такие как несбалансированные данные, где один класс представлен значительно больше или меньше, чем другие классы. Это может повлиять на обучение модели и ее способность предсказывать редкие классы. График распределения меток классов помогает визуализировать эту информацию и принять меры для балансировки классов, если это необходимо.
''', unsafe_allow_html=True)

# Визуализация баланса меток классов
class_counts = train_df['label'].value_counts().sort_index()
class_names = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
class_names = {str(key): class_names[key] for key in class_names}  # Преобразование ключей в строковый тип данных

labels = [class_names[str(i)] for i in class_counts.index.tolist()]
values = class_counts.values.tolist()

fig = px.bar(x=labels, y=values, labels={'x': 'Класс', 'y': 'Количество меток'},
             title='')

fig.update_layout(font=dict(size=14))

fig.update_traces(hovertemplate='Класс: %{x}<br>Количество меток: %{y}')

st.plotly_chart(fig, use_container_width=True)

import re

# Define the path to your stopword file
stopword_file = "stopwords.txt"

def load_stopwords(file_path):
    with open(file_path, "r") as file:
        stopwords = file.read().splitlines()
    return stopwords

def preprocess_review(text, stopwords):
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление чисел
    text = re.sub(r'\d+', '', text)

    # Удаление пунктуации
    text = re.sub(r'[^\w\s]', '', text)

    # Разделение текста на токены
    tokens = text.split()

    # Фильтрация токенов
    clean_tokens = [tok for tok in tokens if tok not in stopwords and len(tok) > 1]

    # Объединение токенов в строку
    clean_text = ' '.join(clean_tokens)

    return clean_text

# Load stopwords from the file
stopwords = load_stopwords(stopword_file)

# Preprocess training data and add a new column
train_df["clean_text"] = train_df["text"].apply(lambda x: preprocess_review(x, stopwords))

# Visualize 50 most frequent tokens
frequent_words = pd.Series(' '.join(train_df.clean_text).split()).value_counts()[:50]
frequent_words = pd.Series(' '.join(train_df.clean_text).split()).value_counts()[:50]


# Загрузка стоп-слов из файла
with open('stopwords.txt', 'r') as f:
    stopwords_english = f.read().splitlines()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()
    clean_tokens = [tok for tok in tokens if tok not in stopwords_english and len(tok) > 1]

    clean_text = ' '.join(clean_tokens)
    return clean_text

# Emotion prediction function using the first model
def predict_emotion(sentence):
    clean_sentence = preprocess_text(sentence)
    encoded_sentence = tokenizer.texts_to_sequences([clean_sentence])
    padded_sentence = pad_sequences(encoded_sentence, maxlen=250)
    emotion_probs = model.predict(padded_sentence)[0]
    emotion_label = np.argmax(emotion_probs)
    return emotion_label


# Load the second model
second_model_path = "SecondModel/model_checkpoints/model_1_Bidirectional_RNN.h5"
second_model = tf.keras.models.load_model(second_model_path)

# Load resources for the second model
tokenizer_path = "SecondModel/model_checkpoints/tokenizer.joblib"
emotion_to_index_path = "SecondModel/model_checkpoints/emotion_to_index.joblib"
index_to_class_path = "SecondModel/model_checkpoints/index_to_class.joblib"
params_path = "SecondModel/model_checkpoints/model_params.joblib"

tokenizer = joblib.load(tokenizer_path)
emotion_to_index = joblib.load(emotion_to_index_path)
index_to_class = joblib.load(index_to_class_path)
params = joblib.load(params_path)

max_len = params["max_len"]
classes_to_index = params["classes_to_index"]

# Emotion prediction function using the second model
def predict_emotion_second_model(sentence):
    sentence_tokens = tokenizer.texts_to_sequences([sentence])
    padded_sentence = pad_sequences(sentence_tokens, truncating='post', padding='post', maxlen=max_len)
    prediction = second_model.predict(padded_sentence)
    predicted_emotion = index_to_class[prediction.argmax()]
    return predicted_emotion

# Model description for the first model
st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 40px;">First Model</h1>
    
''', unsafe_allow_html=True)


def display_model_characteristics():
    characteristics = """
    Данная модель использует библиотеки TensorFlow и Keras для обучения модели классификации эмоций на текстовых данных. Ниже приведены основные характеристики этой модели:

    - Модель использует архитектуру нейронной сети, состоящую из нескольких слоев. Включены слои Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, BatchNormalization и Dropout.
    - Входные данные предварительно обрабатываются: приводятся к нижнему регистру, удаляются числа и пунктуация, разделяются на токены и фильтруются с использованием списка стоп-слов.
    - Обучение модели проводится на обучающих данных и проверяется на проверочном наборе данных.
    - Модель компилируется с использованием оптимизатора Adam и функции потерь sparse_categorical_crossentropy.
    - Обучение модели проводится в течение 15 эпох с размером пакета 64.
    - После обучения модели строятся графики потерь и точности на обучении и проверке.
    - Модель оценивается на проверочном наборе данных, и выводятся значения потерь и точности.
    - Далее, модель применяется для предсказания эмоций на тестовом наборе данных, и результаты сохраняются в файл submission.csv.
    - Вычисляются метрики precision, recall, F1-мера и MCC (Matthews correlation coefficient) на проверочном наборе данных.

    Метрики:
    Accuracy: 0.6537257357545397
    Precision: 0.7065511046221734
    Recall: 0.6711886191258413
    F1 score: 0.6353329842608849
    MCC: 0.6071668366944123

    """
    with st.expander("Описание модели"):
        st.text(characteristics)

# Создание кнопки
if st.button("Вывести описание модели", key="model1_button"):
    if "model1_characteristics" not in st.session_state:
        st.session_state["model1_characteristics"] = False
    
    if not st.session_state["model1_characteristics"]:
        display_model_characteristics()
        st.session_state["model1_characteristics"] = True
    else:
        st.session_state["model1_characteristics"] = False




# Загрузка данных из файла
with open('FirrstModel/firstmodelhistory.pkl', 'rb') as f:
    history_data = pickle.load(f)

# Создание графика значений потерь
fig1 = px.line(history_data, y=['loss', 'val_loss'], labels={'index': 'эпоха', 'value': 'значение потерь'})
fig1.update_layout(
    title="График значений потерь",
    xaxis_title="эпоха",
    yaxis_title="значение потерь",
    legend_title="Тип",
)

# Создание графика точности
fig2 = px.line(history_data, y=['acc', 'val_acc'], labels={'index': 'эпоха', 'value': 'точность'})
fig2.update_layout(
    title="График точности",
    xaxis_title="эпоха",
    yaxis_title="точность",
    legend_title="Тип",
)

# Отображение графиков в Streamlit
st.plotly_chart(fig1)
st.plotly_chart(fig2)

sentence = st.text_input("Введите предложение:")
if sentence:
    predicted_emotion = predict_emotion(sentence)
    st.write("Predicted Emotion (First Model):", predicted_emotion)

# Model description for the second model
st.markdown('''
    <h1 style="text-align: left; color: #FF69B4; font-size: 40px;">Second Model</h1>
 
''', unsafe_allow_html=True)

def display_model_characteristics():
    characteristics = """
    Данная модель представляет собой рекуррентную нейронную сеть (RNN) для анализа эмоций в текстовых данных. Она состоит из нескольких слоев:

    - Вложение (Embedding) - это первый слой модели, который преобразует каждое слово во входных данных (твиты) в вектор фиксированной длины. Размерность вложения равна 16.
    - Двунаправленный LSTM (Bidirectional LSTM) - это слой, который обрабатывает последовательности в обоих направлениях (вперед и назад). Он позволяет модели учитывать контекст информации как в прошлом, так и в будущем. В данной модели используется два слоя двунаправленного LSTM с 20 скрытыми единицами каждый. Первый LSTM слой возвращает последовательности, а второй LSTM слой возвращает только последний выход.
    - Плотный (Dense) - это последний слой модели, который состоит из 6 нейронов (равное количеству уникальных эмоций), активация softmax. Он преобразует выходы LSTM слоя в вероятности для каждой эмоции.
    - Модель компилируется с функцией потерь sparse_categorical_crossentropy, оптимизатором Adam и метрикой accuracy. Веса классов рассчитываются на основе количества образцов в каждом классе для балансировки данных.

    Метрики:
    Accuracy: 0.875
    Precision: 0.8817054225715945
    Recall: 0.875
    F1-score: 0.8771985074636196
    MCC: 0.8363304648715991
    """
    with st.expander("Описание модели"):
        st.text(characteristics)

# Создание кнопки
if st.button("Вывести описание модели", key="model2_button"):
    if "model2_characteristics" not in st.session_state:
        st.session_state["model2_characteristics"] = False
    
    if not st.session_state["model2_characteristics"]:
        display_model_characteristics()
        st.session_state["model2_characteristics"] = True
    else:
        st.session_state["model2_characteristics"] = False



# Загрузка данных из файла
with open('SecondModel/secondmodelhistory.pkl', 'rb') as f:
    history_data2 = pickle.load(f)

# Создание графика значений потерь
fig1 = px.line(history_data2, y=['loss', 'val_loss'], labels={'index': 'эпоха', 'value': 'значение потерь'})
fig1.update_layout(
    title="График значений потерь",
    xaxis_title="эпоха",
    yaxis_title="значение потерь",
    legend_title="Тип",
)

# Создание графика точности
fig2 = px.line(history_data2, y=['accuracy', 'val_accuracy'], labels={'index': 'эпоха', 'value': 'точность'})
fig2.update_layout(
    title="График точности",
    xaxis_title="эпоха",
    yaxis_title="точность",
    legend_title="Тип",
)

# Отображение графиков
st.plotly_chart(fig1)
st.plotly_chart(fig2)

sentence2 = st.text_input("Введите предложение:", key="sentence2")
if sentence2:
    emotion_label = predict_emotion_second_model(sentence2)
    labels_dict = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    emotion = labels_dict[emotion_label]
    st.write("Предсказанная эмоция (Вторая модель):", emotion)

# Conclusion and author information
st.markdown(f'''
    <h1 style="text-align: left; font-family: 'Gill Sans'; color: #FF69B4">Рассуждение:</h1>
    <h1 style="text-align: left; font-family: 'Gill Sans'; color: #FF2A00"></h1>
    <p style="text-align: left; font-family: 'Gill Sans'; font-size: 20px; color: #D8D8D8">
        Для решения этой задачи можно использовать алгоритм <span style='color:green'>бинарного поиска</span>. Конечно, если мы будем подниматься по лестнице и проверять каждый этаж — это <span style
    </p>

''', unsafe_allow_html=True)

st.markdown('')
st.markdown('''<h3 style="text-align: right; font-family: 'Gill Sans'; color: #fd7e14"
            >by Ròman Anatoly</h3>''', unsafe_allow_html=True)
