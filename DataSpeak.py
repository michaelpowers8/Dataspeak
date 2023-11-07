import numpy as np
import pandas as pd
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
warnings.filterwarnings("ignore")
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
import streamlit as st
import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re


# ## Extra Functions

# In[33]:


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


# In[34]:


def remove_gt(sentence):
    new_sentence = re.sub(r'(?<!\w)gt(?!\w)', '', sentence)
    final_sentence = []
    for word in new_sentence.split(' '):
        if not(word=='gt'):
            final_sentence.append(word)
    return ' '.join(final_sentence)


# In[35]:


def answer_bert_question(question, df, models, tokenizer):
    global word2vec_model, corpus_train
    context = generate_word2vec_answer(question, corpus_train, word2vec_model)
    # Tokenize the input text
    encoding = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Get model's output
    outputs = models(input_ids=input_ids, attention_mask=attention_mask)
    start_logits = outputs.logits[0]  # Adjust based on the structure of the MaskedLMOutput object
    end_logits = outputs.logits[len(outputs)-1]  # Adjust based on the structure of the MaskedLMOutput object

    # Find the answer span in the text
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1

    # Decode and return the answer
    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    return answer


# In[44]:


def generate_answers(user_question, df, model, tokenizer, n=5):
    tfidf_vectorizer = TfidfVectorizer()
    question_tfidf = tfidf_vectorizer.fit_transform(df['Body_Q'])
    user_question_tfidf = tfidf_vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_tfidf, question_tfidf)
    sorted_indices = similarities.argsort()[0][::-1]
    top_answers = df['Body_A'].iloc[sorted_indices[:n]].tolist()
    final_answers = []
    for answer in top_answers:
        answer = process_message(answer)
        answer = answer.replace('  ',' ')
        final_answers.append(answer)
    perplexity = calculate_perplexity(final_answers,model,tokenizer)
    return final_answers,similarities,perplexity,sorted_indices


# In[37]:


def calculate_perplexity(answers, model, tokenizer):
    tokenized_answers = tokenizer(answers, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_answers["input_ids"]
    with torch.no_grad():
        outputs = model(**tokenized_answers)
        logits = outputs.logits
    logits_flat = logits.view(-1, logits.shape[-1])
    input_ids_flat = input_ids.view(-1)
    loss = torch.nn.functional.cross_entropy(logits_flat, input_ids_flat, ignore_index=tokenizer.pad_token_id)
    perplexity = torch.exp(loss)
    return perplexity.item()


# In[69]:


def rephrase_sentence(sentence):
    global nlp
    doc = nlp(sentence)
    rephrased_sentence = []
    for token in doc:
        if token.ent_type_ == "PERSON":
            rephrased_sentence.append("someone")
        else:
            rephrased_sentence.append(token.text)
    return " ".join(rephrased_sentence).replace('there are',"there's").replace('problem','issue').replace('  ',' ').replace("do n't",'do not').replace('standard','traditional').replace('alternative','unusual').replace("There 's",'There is').replace('magically','somehow').replace(' ,',',')


# In[39]:


def process_message(input_message):
    sentences = input_message.split('.')
    rephrased_sentences = [rephrase_sentence(sentence) for sentence in sentences]
    new_message = '.'.join(rephrased_sentences)

    return new_message


# In[70]:


def questions_in_terminal(df,model,tokenizer):
    while True:
        clear_output(wait=True)
        question = str(input("Ask a question (type 'exit' to quit): "))
        if(question.lower() in ['exit','done','none','goodbye','bye','finished']):
            break
        #bert = answer_bert_question(question,df,model,tokenizer)
        generated_answers,similarities,perplexity,sorted_indices = generate_answers(question, df, model, tokenizer, n=5)
        for i, answer in enumerate(generated_answers):
            #print(f"{bert}\n\n")
            print(f"Answer {i + 1}: {answer.replace('  ',' ')}\nSimilarity: {similarities.tolist()[0][sorted_indices[i]]}\nPerplexity: {perplexity}\n")

def get_word_cloud(column):
    global df,stop_list
    tokens = None
    comment_words = ''
    for index in range(len(df)):
        val = str(df[column][index])
        if(index%10000==0):
            print(index)
        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)
    word_cloud = WordCloud(width=800,height=800,
                    background_color='white',
                    min_font_size=10).generate(comment_words)
    plt.figure(figsize=(8, 8),facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()
    return comment_words

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"


# In[12]:


data = pd.read_csv('Good_Merged_QA.csv', sep=',', encoding='latin-1', engine='python')
tags = pd.read_csv('Tags.csv', sep=',', encoding='latin-1', engine='python')


# In[13]:


df = data[(data['Score_Q'] >= 0) & (data['Score_A'] >=0)]
df = df.reset_index(drop=True)


# In[14]:


df['Body_Q'] = df['Body_Q'].apply(remove_gt)


# In[15]:


df['Body_A'] = df['Body_A'].apply(remove_gt)


# In[16]:


stop_list = set(stopwords.words('english'))


corpus_train = df['Body_Q']
tokenized_corpus_train = [sentence.lower().split() for sentence in corpus_train]

word2vec_model = models.Word2Vec(tokenized_corpus_train, vector_size=250, window=5, min_count=1, sg=0)
word2vec_model.build_vocab(tokenized_corpus_train,progress_per=100)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

st.header("BERT Question and Answer")
n = st.slider(label="Number of Answers",min_value=1,max_value=10)
st.write(f"Ask any question and I will give you {n} possible answers.")
user_question = st.text_input("Input question here.")

#for i, answer in enumerate(generated_answers):
#    print(f"Answer {i + 1}: {answer.replace('  ',' ')}\nSimilarity: {similarities.tolist()[0][i]}\nPerplexity: {perplexity}\n")
generated_answers,similarities,perplexity,sorted_indices = generate_answers(user_question, df, model, tokenizer, n)
st.table(generated_answers)
st.write(similarities[0][sorted_indices[:n]])
st.write(perplexity)
