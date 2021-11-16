import pandas as pd 
import numpy as np
from gensim.models import LdaMulticore, ldamodel
import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import re


#Topic Modeling 

#load the cleaned and preprocessed dataset 
ArNews_df_update = pd.read_csv("/Users/AlaAlBinSaleh/Desktop/Desktop/Capstone/ArNews_df_Cleaned.csv", index_col=[0])

#tokenize the text  
corpus=[]
word=[]

for i in range(len(ArNews_df_update['Detokenize'])):
        word =ArNews_df_update['Detokenize'].iloc[i]
        corpus.append(word)
texts = [[word for word in str(document).lower().split()]for document in corpus]

#create a dictionary 
from gensim import corpora
dictionary = corpora.Dictionary(texts)
print(dictionary)

#bag of words 
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

#create model
lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=10)

#print topics 
for idx, topic in lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#evaluate the model 

coherence_model_lda = CoherenceModel(model=lda, corpus=corpus,texts=texts ,dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

#find optimal results 
def coherence_score(dictionary, corpus, texts, limit, start, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        
        coherencemodel = CoherenceModel(model=model, corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = coherence_score(dictionary=dictionary, corpus=corpus, texts=texts, start=2, limit=50, step=5)
x = range(2, 50, 5)
for num, cv in zip(x, coherence_values):
    print("topic number:", num, " , coherence score", cv)

#plot to find the highest coherence score 
plt.plot(x, coherence_values)
plt.xlabel("Topics number")
plt.ylabel("Coherence Ecore")
plt.title("the Optimal Topic number Based on Coherence Score")
plt.annotate("Max Score", xy=(37, 0.43725),
            arrowprops=dict(facecolor='#F3F085', shrink=1))
plt.axvline(x=37, color='#EA4141', linestyle='--')

