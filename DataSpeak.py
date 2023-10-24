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


# In[3]:


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


# In[4]:


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


# In[5]:


def calculate_perplexity(answers, model, tokenizer):
    tokenized_answers = tokenizer(answers, return_tensors="pt", padding=True, truncation=True)
    print(tokenized_answers['input_ids'].shape)
    with torch.no_grad():
        outputs = model(**tokenized_answers)
        logits = outputs.logits
    print(logits.shape)
    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits, tokenized_answers["input_ids"]))
    return perplexity.item()


# In[6]:


def write_new_sentence(sentence):
    words = sentence.split()
    rewritten_sentence = []
    for word in words:
        if(word.lower()=='it' or word.lower()=='i' or word.lower()=='will' or word.lower()=='a'):
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


# In[7]:


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


# In[8]:


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

# In[9]:


preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"


# In[10]:


data = pd.read_csv('Good_Merged_QA.csv', sep=',', on_bad_lines='skip', encoding='latin-1', engine='python')
tags = pd.read_csv('Tags.csv', sep=',', on_bad_lines='skip', encoding='latin-1', engine='python')


# ## Clean Data

# In[11]:


df = data[(data['Score_Q'] >= 0) & (data['Score_A'] >=0)]
df = df.reset_index(drop=True)


# In[12]:


df.info()


# In[13]:


df.sample(2)


# In[14]:


stop_list = set(stopwords.words('english'))


# ## EDA
df['Score_Q'].plot(kind='hist',
                          bins=100)df['Score_A'].plot(kind='hist',
                        bins=100)df['Body'].replace(['<p>','</p>','\n','<ul>','<li>','</li>','</ul>'],'',regex=True)[0]comment_words = ''
for index in range(len(df)):
    val = str(df['Body_Q'][index])
    if(index%10000==0):
        print(index)
    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)word_cloud = WordCloud(width=800,height=800,
                        background_color='white',
                        stopwords=stop_list,
                        min_font_size=10).generate(comment_words)plt.figure(figsize=(8, 8),facecolor=None)
plt.imshow(word_cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# ## Model Training

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(df['Body_Q'],df['Body_A'], test_size=0.2)


# In[16]:


bert_preprocess_model = hub.KerasLayer(preprocess_url)


# In[17]:


corpus_train = X_train
tokenized_corpus_train = [sentence.lower().split() for sentence in corpus_train]


# In[18]:


corpus_test = X_test
tokenized_corpus_test = [sentence.lower().split() for sentence in corpus_test]


# In[ ]:


word2vec_model = models.Word2Vec(tokenized_corpus_train, vector_size=250, window=5, min_count=1, sg=0)


# In[ ]:


word2vec_model.build_vocab(tokenized_corpus_train,progress_per=100)


# In[ ]:


df['Body_Q'][22]


# In[ ]:


df['Body_A'][0]


# In[ ]:


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


# In[ ]:


while True:
    question = str(input("Ask a question (type 'exit' to quit): "))
    if(question.lower() in ['exit','done','none','goodbye','bye','finished']):
        break
    answer = answer_bert_question(question,df['Body_A'],word2vec_model)
    print(f"\n{answer}\n\n\n")
    print(generate_word2vec_answer(question, df['Body_A'], word2vec_model))
    generated_answers = generate_answers(question, df, model, tokenizer, n=5)
    for i, answer in enumerate(generated_answers):
        print(f"Answer {i + 1}: {answer}\n")


# In[ ]:


st.header("BERT Question and Answer")
st.write("""
         Ask any question and I will give you 5 possible answers.
         """)


# In[ ]:


user_question = st.text_input("Input question here.")
generated_answers = generate_answers(user_question, df, model, tokenizer, n=5)
st.write(generated_answers)

Ask a question (type 'exit' to quit): 

I am going to start working on a hobby project with a python codebase. I'd like to set up some form of continuous integration (i.e. running a battery of test-cases each time a check-in is made and sending nag e-mails to responsible persons when the tests fail) similar to CruiseControl or TeamCity. I realize I could do this with hooks in most VCSes, but that requires that the tests run on the same machine as the version control server, which isn't as elegant as I would like. Does anyone have any suggestions for a small, user-friendly, open-source continuous integration system suitable for a Python codebase?

I've always liked doing it this way result = { 'a': lambda x: x * 5, 'b': lambda x: x + 7, 'c': lambda x: x - 2 }[value](x) From here

Answer 1: One possibility is Hudson. It's written in Java, but there's integration with Python projects: Hudson embraces Python I've never tried it myself, however. (Update, Sept. 2011: After a trademark dispute Hudson has been renamed to Jenkins.)

Answer 2: We run Buildbot - Trac at work, I haven't used it too much since my code base isn't part of the release cycle yet. But we run the tests on different environments (OSX/Linux/Win) and it sends emails --and it's written in python.

Answer 3: Second the Buildbot - Trac integration. You can find more information about the integration on the Buildbot website. At my previous job, we wrote and used the plugin they mention (tracbb). What the plugin does is rewriting all of the Buildbot urls so you can use Buildbot from within Trac. (http://example.com/tracbb). The really nice thing about Buildbot is that the configuration is written in Python. You can integrate your own Python code directly to the configuration. It's also very easy to write your own BuildSteps to execute specific tasks. We used BuildSteps to get the source from SVN, pull the dependencies, publish test results to WebDAV, etcetera. I wrote an X10 interface so we could send signals with build results. When the build failed, we switched on a red lava lamp. When the build succeeded, a green lava lamp switched on. Good times :-)

Answer 4: We use both Buildbot and Hudson for Jython development. Both are useful, but have different strengths and weaknesses. Buildbot's configuration is pure Python and quite simple once you get the hang of it (look at the epydoc-generated API docs for the most current info). Buildbot makes it easier to define non-testing tasksÃ and distribute the testers. However, it really has no concept of individual tests, just textual, HTML, and summary output, so if you want to have multi-level browsable test output and so forth you'll have to build it yourself, or just use Hudson. Hudson has terrific support for drilling down from overall results into test suites and individual tests; it also is great for comparing test output between builds, but the distributed (master/slave) stuff is comparatively more complicated because you need a Java environment on the slaves too; also, Hudson is less tolerant of flaky network links between the master and slaves. So, to get the benefits of both tools, we run a single instance of Hudson, which catches the common test failures, then we do multi-platform regression with Buildbot. Here are our instances: Jython Hudson Jython buildbot

Answer 5: We are using Bitten wich is integrated with trac. And it's python based.

Ask a question (type 'exit' to quit): exitYou can make li a dictionary: li = {} for j in range(10): li[j] = []
Answer 1: One possibility is Hudson. It's written in Java, but there's integration with Python projects: Hudson embraces Python I've never tried it myself, however. (Update, Sept. 2011: After a trademark dispute Hudson has been renamed to Jenkins.)
# In[ ]:




