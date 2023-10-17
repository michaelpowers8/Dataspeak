#!/usr/bin/env python
# coding: utf-8

# # DataSpeak

# ## Import Libraries

# In[1]:


import os
from time import time,sleep
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import pandas as pd
from random import randint as rand
from random import choice
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import torch
import transformers 
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
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


# ## Extra Functions

# In[2]:


def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # F1 Score
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'F1 Score') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return


# In[3]:


def clean_text(text):
    cleaned = " ".join(re.sub(r"[^0-9a-zA-Z']", " ", text).split())
    return cleaned.lower()


# In[4]:


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
    #print(jellyfish.levenshtein_distance(str(question), str(corpus)))
    similarities = cosine_similarity([question_vector], sentence_vectors)[0]
    # Find the sentence with the highest similarity as the answer
    max_similarity_index = np.argmax(similarities)
    print(max_similarity_index)
    answer = corpus[max_similarity_index]

    return answer


# In[5]:


def answer_bert_question(question,context):
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
    
    # Tokenize the input text
    encoding = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
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


# In[6]:


def generate_answers(user_question, df, model, tokenizer, n=5):
    # Calculate TF-IDF vectors for questions
    tfidf_vectorizer = TfidfVectorizer()
    question_tfidf = tfidf_vectorizer.fit_transform(df['Body_questions_norm'])

    # Calculate the TF-IDF vector for the user question
    user_question_tfidf = tfidf_vectorizer.transform([user_question])

    # Calculate cosine similarity between user question and dataset questions
    similarities = cosine_similarity(user_question_tfidf, question_tfidf)
    print(sorted(similarities))

    # Sort questions by similarity score
    sorted_indices = similarities.argsort()[0][::-1]
    print(sorted_indices)

    # Get the top N answers based on similarity
    top_answers = df['Body_answers_norm'].iloc[sorted_indices[:n]].tolist()
    
    final_answers = []
    for answer in top_answers:
        answer = rewrite_sentence(answer)
        final_answers.append(answer)

    return final_answers


# In[7]:


def calculate_perplexity(answers, model, tokenizer):
    tokenized_answers = tokenizer(answers, return_tensors="pt", padding=True, truncation=True)
    print(tokenized_answers['input_ids'].shape)
    with torch.no_grad():
        outputs = model(**tokenized_answers)
        logits = outputs.logits
    print(logits.shape)
    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits, tokenized_answers["input_ids"]))
    return perplexity.item()


# In[8]:


def is_word(word):
    # Check if a word is considered valid
    return len(word) > 1 or word.lower() in ["a", "i"]
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms
def preserve_tense(word, new_word):
    # Preserve the tense of the word
    if word.endswith('ed'):
        return new_word + 'ed'
    elif word.endswith('ing'):
        return new_word + 'ing'
    return new_word


# In[9]:


def write_new_sentence(sentence):
    words = sentence.split()
    rewritten_sentence = []
    for word in words:
        if(word.lower()=='it'):
            rewritten_sentence.append(word)
        else:
            synonyms = get_synonyms(word)
            if synonyms:
                # Choose a random synonym from the list
                synonym = synonyms[0]
                synonym = preserve_tense(word,synonym)
                rewritten_sentence.append(synonym)
            else:
                rewritten_sentence.append(word)
    return ' '.join(rewritten_sentence)


# In[10]:


def clean_sloppy_sentence(sloppy_sentence):
    # Split the sentence into words
    words = sloppy_sentence.split()

    # Initialize a list to store cleaned words
    cleaned_words = []

    # Flag to keep track of the first word
    first_word = True

    # Iterate through the words
    for word in words:
        # Check if the word is considered a valid word
        if is_word(word):
            # Capitalize the first letter of the cleaned sentence
            if first_word:
                word = word[0].upper() + word[1:]
                first_word = False
            cleaned_words.append(word)

    # Join the cleaned words to form the cleaned sentence
    cleaned_sentence = ' '.join(cleaned_words)
    return cleaned_sentence


# In[11]:


def rewrite_sentence(sentence):
    sentence = write_new_sentence(clean_sloppy_sentence(sentence))
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Tag the words with their parts of speech
    tagged_words = pos_tag(words)

    # Initialize a list to store the rewritten words
    rewritten_words = []

    for word, pos in tagged_words:
        # Rewrite the word while preserving tense (if applicable)
        if pos.startswith('V'):  # Verbs
            rewritten_word = word
        else:
            rewritten_word = word  # Keep non-verbs as they are
        rewritten_words.append(rewritten_word)

    # Join the rewritten words to form the rewritten sentence
    rewritten_sentence = ' '.join(rewritten_words)

    return rewritten_sentence


# ## Load Data

# In[12]:


preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"


# In[ ]:





# In[13]:


data = pd.read_csv('Merged_QA.csv', sep=',', on_bad_lines='skip', encoding='latin-1', engine='python')
tags = pd.read_csv('Tags.csv', sep=',', on_bad_lines='skip', encoding='latin-1', engine='python')


# In[14]:


data.describe()


# In[15]:


tags.info()


# ## Clean Data

# In[16]:


df = data[(data['Score_question'] >= 0) & (data['Score_question'] <= 10) & (data['Score_answer'] >=0) & (data['Score_answer'] <= 10)]
df = df.reset_index(drop=True)


# In[17]:


df.info()


# In[18]:


df.sample(2)


# In[19]:


stop_words = set(stopwords.words('english'))


# ## EDA

# In[ ]:


df['Score_question'].plot(kind='hist',
                          bins=100)


# In[ ]:


df['Score_answer'].plot(kind='hist',
                        bins=100)


# df['Body'].replace(['<p>','</p>','\n','<ul>','<li>','</li>','</ul>'],'',regex=True)[0]

# In[ ]:


comment_words = ''
for index in range(10000):
    val = str(df['Body_questions_norm'][index])
    if(index%50000==0):
        print(index)
    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "


# In[ ]:


word_cloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 10).generate(comment_words)


# In[ ]:


plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# ## Model Training

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['Body_questions_norm'],df['Body_answers_norm'], test_size=0.2)


# In[ ]:


bert_preprocess_model = hub.KerasLayer(preprocess_url)


# In[ ]:


corpus_train = X_train
tokenized_corpus_train = [sentence.lower().split() for sentence in corpus_train[:100000]]


# In[ ]:


corpus_test = X_test
tokenized_corpus_test = [sentence.lower().split() for sentence in corpus_test[:100000]]


# In[ ]:


word2vec_model = models.Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)


# In[ ]:


word2vec_model.build_vocab(tokenized_corpus,progress_per=100)


# In[ ]:


text_preprocessed = bert_preprocess_model(X_train[:100])
text_preprocessed.keys()


# In[ ]:


text_preprocessed


# In[ ]:


bert_model = hub.KerasLayer(encoder_url)


# In[ ]:


bert_results = bert_model(text_preprocessed)


# In[ ]:


bert_results.keys()


# In[ ]:


len(bert_results['encoder_outputs'])


# In[ ]:


bert_results


# In[ ]:


df['Body_questions_norm'][0]


# In[ ]:


df['Body_answers_norm'][:5]


# In[ ]:


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


# In[ ]:


while True:
    question = str(input("Ask a question (type 'exit' to quit): "))
    os.system('cls')
    if(question.lower() in ['exit','done','none','goodbye','bye','finished']):
        break
    generated_answers = generate_answers(user_question, df, model, tokenizer, n=5)
    for i, answer in enumerate(generated_answers):
        answer = rewrite_sentence(answer)
        print(f"Answer {i + 1}: {answer}\n")
    #perplexity = calculate_perplexity(generated_answers, model, tokenizer)
    #print(f"Perplexity: {perplexity}")


# In[20]:


st.header("BERT Question and Answer")
st.write("""
         Ask any question and I will give you 5 possible answers.
         """)


# In[ ]:


user_question = st.text_input("")
generated_answers = generate_answers(user_question, df, model, tokenizer, n=5)
st.write(generated_answers)


# In[ ]:




