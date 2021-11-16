import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from plotnine import *
from glob import glob
import numpy as np 
import arabicstopwords.arabicstopwords as stp
import missingno as msno
import re
#from farasa.stemmer import FarasaStemmer
import nltk 



#Download Arabic stopwords from NLTK 
#nltk.download()
nltk.download("stopwords")
arabic_stopwords_list = set(nltk.corpus.stopwords.words("arabic"))
#update the stopword lisy 
arabic_stopwords_list.update(["فى","في","كل","لم","لن","له""من","هو","هي","قوة","كما","لها",'منذ',"وقد",'ولا',
"نفسه","لقاء","مقابل","هناك","وقال","وكان","نهاية","وقالت","وكانت","للامم","فيه",
"كلم","لكن","وفي","وقف","ولم","ومن","وهو","وهي","يوم","فيها","منها","مليار",
"لوكالة","يكون","يمكن","مليون","حيث","اكد","الا","اما","امس","السابق","التى","التي",
"اكثر","ايار","ايضا","ثلاثة","الذاتي","الاخيرة","الثاني","الثانية","الذى","الذي","الان",
"امام","ايام","خلال","حوالى","الذين","الاول","الاولى","بين","ذلك","دون","حول","حين","الف",
"الى","انه","اول","ضمن","انها","جميع","الماضي","الوقت","المقبل","اليوم","ـ","ف",
"و","و6","قد","لا","ما","مع","مساء","هذا","واحد","واضاف","واضافت","فان","قبل","قال","كان",
"لدى","نحو","هذه","وان","واكد","كانت","واوضح","مايو","ب","ا","أ","،","عشر","عدد","عدة","عشرة",
"عدم","عام","عاما","عن","عند","عندما","على","عليه","عليها","زيارة","سنة","سنوات","تم","ضد","بعد",
"بعض","اعادة","اعلنت","بسبب","حتى","اذا","احد","اثر","برس","باسم","غدا","شخصا","صباح","اطار",
"اربعة","اخرى","بان","اجل","غير","بشكل","حاليا","بن","به","ثم","اف","ان","او","اي","بها","صفر"
"أنه","بأن"])

#obtain Dataset 
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#Step 1: convert to jason files into csv 
files_list = glob('news_*')

dfs = [] 
for file in files_list:
    data = pd.read_json(file) 
    dfs.append(data) 

ArNews_df = pd.concat(dfs, ignore_index=True)
#save the dataset 
ArNews_df.to_csv("/Users/AlaAlBinSaleh/Desktop/Capstone/data/ArNews_df.csv")

#Step 2:Data cleaning 
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#handling missing data 
# closer look into the structure of the dataset 
ArNews_df.info()
ArNews_df.shape
ArNews_df.head()

# check for the missing data 
ArNews_df.isnull().values.any() # why?????????????
#len(ArNews_df.author.unique()) has a lot of missing values that are not showing 

#replacing blanks " " with nan: 
ArNews_df = ArNews_df.replace(r'^\s*$', np.nan, regex=True)

# dealing with missing values: 
ArNews_df.isnull().sum()

#missing data plot  
msno.matrix(ArNews_df)

# drop all NaN  
ArNews_df = ArNews_df[ArNews_df.content.isnull() == False]
ArNews_df = ArNews_df[ArNews_df.author.isnull() == False]
ArNews_df = ArNews_df[ArNews_df.title.isnull() == False]
#reset index 
ArNews_df = ArNews_df.reset_index(drop=True)

ArNews_df.content.isnull().any() #false 

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Removing unwanted characters and stop words  
#remove punctuation 
p = re.compile(r'[^\w\s]+')
ArNews_df['content'] = [p.sub('', x) for x in ArNews_df['content'].tolist()]

#remove \n from the content 
ArNews_df['content'] = ArNews_df.content.str.replace("\xa0","") 
ArNews_df['content'] = ArNews_df.content.str.replace("\n"," ")
ArNews_df['content'] = ArNews_df.content.str.replace("_ ","")
ArNews_df['content'] = ArNews_df.content.str.replace("- ","")
ArNews_df['content'] = ArNews_df.content.str.replace(":","")
ArNews_df['content'] = ArNews_df.content.str.replace("»","")
ArNews_df['content'] = ArNews_df.content.str.replace("«","")

#remove strings connected to symbols    
ArNews_df['content'] = ArNews_df.content.str.replace("سبق","")
ArNews_df['content'] = ArNews_df.content.str.replace("سبق-","")
ArNews_df['content'] = ArNews_df.content.str.replace("- -","")

#remove digits/numbers  
ArNews_df['content'] = ArNews_df['content'].str.replace('\d+', '')

# remove parentheses () and data inside them 
ArNews_df["content"] = ArNews_df["content"].str.replace(r" \(.*\)","")

#fix the source/news paper names just because 
ArNews_df['source'] = ArNews_df.source.str.replace("aawsat","Al Sharq Al Awsat")
ArNews_df['source'] = ArNews_df.source.str.replace("aleqtisadiya","Al Eqtisadiya")
ArNews_df['source'] = ArNews_df.source.str.replace("aljazirah","Al Jazirah")
ArNews_df['source'] = ArNews_df.source.str.replace("almadina","Al Madina")
ArNews_df['source'] = ArNews_df.source.str.replace("alriyadh","Al Riyadh")
ArNews_df['source'] = ArNews_df.source.str.replace("alwatan","Al Watan")
ArNews_df['source'] = ArNews_df.source.str.replace("alweeam","Al Weeam")
ArNews_df['source'] = ArNews_df.source.str.replace("alyaum","Al Yaum")
ArNews_df['source'] = ArNews_df.source.str.replace("okaz","Okaz")
ArNews_df['source'] = ArNews_df.source.str.replace("sabq","Sabq")
ArNews_df['source'] = ArNews_df.source.str.replace("arreyadi","Arreyadi")
ArNews_df['source'] = ArNews_df.source.str.replace("arriyadiyah","Arriyadiyah")

#author column cleaning 
ArNews_df['author'] = [p.sub('', x) for x in ArNews_df['author'].tolist()]
ArNews_df['author'] = ArNews_df.author.str.replace("سبق","")
ArNews_df['author'] = ArNews_df.author.str.replace("واس","")
ArNews_df['author'] = ArNews_df.author.str.replace("\xa0","") 

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#preprocessing 

#tokenization: 
ArNews_df['words_before_stop'] = ArNews_df["content"].apply(nltk.word_tokenize)

#remove stop words 
ArNews_df['words_after_stop']= ArNews_df['words_before_stop'].apply(lambda x: [item for item in x if item not in arabic_stopwords_list])

#stemming 

#feature Engineering 

#number of words before and after stop words removal 
ArNews_df['word_count_stop'] = ArNews_df['words_before_stop'].str.len() #before 
ArNews_df['word_count_wo_stop'] = ArNews_df['words_after_stop'].str.len() #after 


#drop articles that didnt change after stop words  
ArNews_df = ArNews_df[ArNews_df.word_count_stop != ArNews_df.word_count_wo_stop]


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#EDA 
#----------------------------------------------------------------------------
# News Source vs Content 
Source_Content = Counter(ArNews_df["source"])
Source_Content_df = pd.DataFrame(
    {'source': Source_Content.keys(),
     'articles_count': Source_Content.values(),
    })
Source_Content_df = Source_Content_df.reset_index()
Source_Content_df.sort_values(by=["articles_count"])


# plot the 
sns.barplot(x="source", 
            y="articles_count",
            data= Source_Content_df,
            order=Source_Content_df.sort_values('articles_count',ascending = False).source,
            palette=sns.color_palette("viridis", 12)
            )
plt.xticks(rotation=90)
plt.rcParams.update({'font.size': 12})
sns.set(rc = {'figure.figsize':(15,8)})
plt.title('Number of Articles Based on Their Source', fontsize=20)
plt.xlabel('Source', fontsize=16)
plt.ylabel('Number_of_Articles', fontsize=16)

"""
Insight: 
Al Riyadh has the most articles in this data are. since, it is one of the leading news platform in KSA 
While the Arriyadi has the least number of aricles . 

Check what topics does it cover based keywords 
"""

#----------------------------------------------------------------------------
#Before and after word removal plot (Change)
plt.plot(ArNews_df.word_count_stop, color='#472365')
plt.plot(ArNews_df.word_count_wo_stop, color='#C7D353')
plt.title('Words Count Distribution Before And After StopWords Removal', fontsize=20)
plt.xlabel('Articles', fontsize=16)
plt.ylabel('Length', fontsize=16)

"""
Insight: 
most articles had drop in words count except for 4 articles when are already small in size  
"""
#------------------------------------------------------------------
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from collections import Counter
detokenizer = Detok()

#detokenize 
ArNews_df['Detokenize'] = ArNews_df["words_after_stop"].apply(detokenizer.detokenize)

#find the most frequent words
#top 15 words after stemming 
Top_words = Counter(" ".join(ArNews_df["Detokenize"]).split()).most_common(15)
words = [t for t,n in Top_words]
frequency = [n for t,n in Top_words]
Top_words_df = pd.DataFrame({'words': words,'frequency': frequency})
Top_words_df.sort_values(by=["frequency"])

#plot  
sns.barplot(x="frequency", 
            y="words", 
            data=Top_words_df, 
            palette=sns.color_palette("viridis", 15)
            )
sns.set(rc = {'figure.figsize':(15,8)})
plt.title('Top 100 Frequent Words', fontsize=20)
plt.xlabel('Articles', fontsize=16)
plt.ylabel('Length', fontsize=16)

#wordcloud: 
from ar_wordcloud import ArabicWordCloud

texts = [' '.join(Top_words_df['words']) for comment in Top_words_df['words']]
texts = ' '.join(texts)
awc = ArabicWordCloud(background_color="black").generate(texts)
wc = awc.from_text(texts)
plt.imshow(wc)
plt.axis("off")
plt.rcParams["figure.figsize"] = (15,8)
plt.title('WordCloud: Top 100 Frequent Words', fontsize=20)

#------------------------------
#save updated Dataset
ArNews_df.to_csv("/Users/AlaAlBinSaleh/Desktop/Capstone/data/ArNews_df_update.csv")


from farasa.ner import FarasaNamedEntityRecognizer
ner = FarasaNamedEntityRecognizer()