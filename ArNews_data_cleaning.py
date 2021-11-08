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
from camel_tools.sentiment import SentimentAnalyzer
import nltk 
from nltk.stem.snowball import SnowballStemmer
import unicodedata
from textblob_ar import TextBlob as tb


#Download Arabic stopwords from NLTK 
#nltk.download()
nltk.download("stopwords")
arabic_stopwords_list = set(nltk.corpus.stopwords.words("arabic"))
stemmer = SnowballStemmer("arabic")
sentiment = SentimentAnalyzer.pretrained()

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

# remove cities name from the author name 
Saudi_City = ["الرياض","مكة المكرمة","جدة","المدينة المنورة","الأحساء","الدمام","الطائف","بريدة","تبوك","القطيف","خميس مشيط",
"الخبر","حفر الباطن","الجبيل","الخرج","أبها","حائل","نجران","ينبع","صبيا","الدوادمي","بيشة","أبو عريش","القنفذة","محايل","سكاكا",
"عرعر","عنيزة","القريات","صامطة","جازان","المجمعة","القويعية","الرس","وادي الدواسر","بحرة","الباحة","الجموم","رابغ","أحد رفيدة",
"شرورة","الليث","رفحاء","عفيف","العرضيات","العارضة","الخفجي","بلقرن","الدرعية","ضمد","طبرجل","بيش","الزلفي","الدرب","الافلاج","سراة عبيدة",
"رجال المع","بلجرشي","الحائط","ميسان","بدر","أملج","رأس تنورة","المهد","الدائر","البكيرية","البدائع","خليص","العلا","الحناكية","الطوال",
"النماص","المجاردة","بقيق","تثليث","المخواة","النعيرية","الوجه","ضباء","بارق","طريف","خيبر","أضم","النبهانية","رنية","دومة الجندل",
"المذنب","تربة","الخرمة","قلوة","شقراء","المويه","المزاحمية","الأسياح","بقعاء","السليل","تيماء"]

ArNews_df['author_words'] = ArNews_df["author"].apply(nltk.word_tokenize)
ArNews_df['author_words'] = ArNews_df['author_words'].apply(lambda x: [item for item in x if item not in Saudi_City])
ArNews_df['author_words'].isnull().sum()

others = ["واشنطن", "بغداد", "طهران", "نيويورك", "الشرق", "الأوسط", "كليفلاند", "الولايات", "المتحدة", "برلين", "موسكو","أتلانتا","رويترز", "القاهرة", "تقرير", ""]

#ArNews_df = ArNews_df.drop('author_words', 1)

len(ArNews_df['author'].unique())
ArNews_df['author_words'].loc[662]
26108
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#preprocessing 

#tokenization: 
ArNews_df['words_before_stop'] = ArNews_df["content"].apply(nltk.word_tokenize)

#remove stop words 
ArNews_df['words_after_stop']= ArNews_df['words_before_stop'].apply(lambda x: [item for item in x if item not in arabic_stopwords_list])

#stemming 
#ArNews_df['words_before_stop'] = ArNews_df['Words_before_stop'].apply(lambda x: [stemmer.stem(y) for y in x])
ArNews_df['words_stem'] = ArNews_df['words_before_stop'].apply(lambda x: [stemmer.stem(y) for y in x])
#ArNews_df.apply(lambda row: nltk.word_tokenize(row["verbatim"]), axis=1)

#remove stop words 
#ArNews_df['words_after_stop']= ArNews_df['words_before_stop'].apply(lambda x: [item for item in x if item not in arabic_stopwords_list])

#feature Engineering 

#number of words before and after stop words removal 
ArNews_df['word_count_stop'] = ArNews_df['words_before_stop'].str.len() #before 
ArNews_df['word_count_wo_stop'] = ArNews_df['words_after_stop'].str.len() #after 

#sentemint 
type(ArNews_df['content'].loc[23446])

#drop articles that are less than 20 word long 
ArNews_df = ArNews_df[ArNews_df.word_count_stop > 20]

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#EDA 
#----------------------------------------------------------------------------
# News Source vs Content 
Source_Content = ArNews_df.groupby(['source'])["content"].count().sort_values(ascending=True)
pd.DataFrame(Source_Content)

# plot the 
sns.countplot(x="source", 
              data=ArNews_df,
              palette =sns.color_palette("hls", 12)
            )
plt.xticks(rotation=45)
plt.rcParams.update({'font.size': 10})
plt.title("Number of Articles Based on Their Source")

"""
Insight: 
Al Riyadh has the most articles in this data are. since, it is one of the leading news platform in KSA 
While the Arriyadi has the least number of aricles . 

Check what topics does it cover based keywords 
"""

#----------------------------------------------------------------------------
#Before and after word removal plot (Change)
ArNews_df['words_differance'] = (ArNews_df['word_count_stop']) - (ArNews_df['word_count_wo_stop'])

len(ArNews_df[ArNews_df.words_differance == 0])

plt.plot(ArNews_df.word_count_stop, color='#2D80B2')
plt.plot(ArNews_df.word_count_wo_stop, color='#A6D59E')
plt.title("Words count distribution before and after stop words removal")

"""
Insight: 
most articles had drop in words count except for 4 articles when are already small in size  
"""

#----------------------------------------------------------------------------
#sentiment anaysis 
import os
java_path = "C:/Program Files/Java/jdk1.8.0_131/bin/java.exe"
/Library/Java/JavaVirtualMachines/jdk1.8.0_144/Contents/Home/bin/java
os.environ['JAVAHOME'] = java_path

sentiment = SentimentAnalyzer.pretrained()

def Sentiment_analysis(text):
  try:
    return 
  except:
    return None

a = ArNews_df["content"].apply(lambda x: [sentiment.predict(y) for y in x])

from collections import Counter
Counter(" ".join(ArNews_df["content"]).split()).most_common(100)
pd.Series(ArNews_df["content"])
blob = TextBlob(.tolist())

ArNews_df.to_csv("/Users/AlaAlBinSaleh/Desktop/Capstone/data/ArNews_df_update.csv")

