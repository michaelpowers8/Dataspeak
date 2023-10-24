from time import time,sleep
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from random import randint as rand
from random import choice
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from transformers import BertForQuestionAnswering, BertTokenizer
from IPython.display import clear_output
from transformers import BertTokenizer, BertForMaskedLM
from transformers import logging
import streamlit as st
logging.set_verbosity_error()

def generate_word2vec_answer(question, corpus, word2vec_model):
    question_tokens = question.lower().split()
    question_vector = np.mean([word2vec_model.wv[token] for token in question_tokens if token in word2vec_model.wv], axis=0)
    sentence_vectors = []
    
    for index in range(len(corpus)):
        sentence = ' '.join(corpus[index])
        sentence_tokens = sentence.lower().split()
        sentence_vector = np.mean([word2vec_model.wv[token] for token in sentence_tokens if token in word2vec_model.wv], axis=0)
        sentence_vectors.append(sentence_vector)

    # Calculate cosine similarities between the question vector and sentence vectors
    similarities = cosine_similarity([question_vector], sentence_vectors)[0]
    # Find the sentence with the highest similarity as the answer
    max_similarity_index = np.argmax(similarities)
    answer = corpus[max_similarity_index]

    return answer

def answer_bert_question(question,df,models):
    # Load the pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    '''
    context = generate_word2vec_answer(question,tokenized_corpus,word2vec_model)
    final_context = []
    for word in context:
        if word not in ['p','gt','ul','\n','li']:
            final_context.append(word)
    final_context = ' '.join(final_context)
    '''
    context = generate_word2vec_answer(question, df, models)
    # Tokenize the input text
    encoding = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=False)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Get model's output
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    
    # Convert start_scores and end_scores to tensors if not already
    if not isinstance(start_scores, torch.Tensor):
        start_scores = torch.tensor(start_scores)
    if not isinstance(end_scores, torch.Tensor):
        end_scores = torch.tensor(end_scores)
    
    # Find the answer span in the text
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    # Decode and return the answer
    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    return answer

def generate_answers(user_question, df, model, tokenizer, n=5):
    # Calculate TF-IDF vectors for questions
    tfidf_vectorizer = TfidfVectorizer()
    question_tfidf = tfidf_vectorizer.fit_transform(df['Body_Q'])

    # Calculate the TF-IDF vector for the user question
    user_question_tfidf = tfidf_vectorizer.transform([user_question])

    # Calculate cosine similarity between user question and dataset questions
    similarities = cosine_similarity(user_question_tfidf, question_tfidf)
    # Sort questions by similarity score
    sorted_indices = similarities.argsort()[0][::-1]

    # Get the top N answers based on similarity
    top_answers = df['Body_A'].iloc[sorted_indices[:n]].tolist()
    
    final_answers = []
    for answer in top_answers:
        #answer = write_new_sentence(answer)
        final_answers.append(answer)

    return final_answers

def calculate_perplexity(answers, model, tokenizer):
    tokenized_answers = tokenizer(answers, return_tensors="pt", padding=True, truncation=True)
    print(tokenized_answers['input_ids'].shape)
    with torch.no_grad():
        outputs = model(**tokenized_answers)
        logits = outputs.logits
    print(logits.shape)
    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits, tokenized_answers["input_ids"]))
    return perplexity.item()

preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

data = pd.read_csv('Good_Merged_QA.csv', sep=',', on_bad_lines='skip', encoding='latin-1', engine='python')
tags = pd.read_csv('Tags.csv', sep=',', on_bad_lines='skip', encoding='latin-1', engine='python')

df = data[(data['Score_Q'] >= 0) & (data['Score_A'] >=0)]
df = df.reset_index(drop=True)

stop_list = set(stopwords.words('english'))
X_train, X_test, y_train, y_test = train_test_split(df['Body_Q'],df['Body_A'], test_size=0.2)
corpus_train = X_train
tokenized_corpus_train = [sentence.lower().split() for sentence in corpus_train]
corpus_test = X_test
tokenized_corpus_test = [sentence.lower().split() for sentence in corpus_test]
word2vec_model = models.Word2Vec(tokenized_corpus_train, vector_size=250, window=5, min_count=1, sg=0)
word2vec_model.build_vocab(tokenized_corpus_train,progress_per=100)
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

st.header("BERT Question and Answer")
st.write("""
         Ask any question and I will give you 5 possible answers.
         """)

user_question = st.text_input("Input question here.")
generated_answers = answer_bert_question(user_question,df['Body_A'],word2vec_model)#generate_answers(user_question, df, model, tokenizer, n=5)

st.write(generated_answers)
