import pandas as pd
from camel_tools.sentiment import SentimentAnalyzer
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

#load Prep and clean data 
ArNews_df_update = pd.read_csv("/Users/AlaAlBinSaleh/Desktop/Desktop/Capstone/ArNews_df_Cleaned.csv", index_col=[0])

#define the Sentiment Analyzer  
sentiment = SentimentAnalyzer.pretrained()

ArNews_df_update["sentiment"] = ArNews_df_update['content'].apply(Sentiment_Analysis)


def Sentiment_Analysis(text):
  try:
    return sentiment.predict(text)
  except:
    return None

#sentiment analysis on part of the dataset 
partial_df = ArNews_df_update[:1000]

#apply on unclean content 
partial_df["sentiment"] = partial_df['content'].apply(Sentiment_Analysis)

#convert the data into a string to plot 
partial_df['sentiment'] = partial_df.sentiment.str.replace("[","")
partial_df['sentiment'] = partial_df.sentiment.str.replace("]","")
partial_df['sentiment'] = partial_df.sentiment.str.replace("'","")

#plot 
sns.countplot(x = "sentiment", 
               data = partial_df,
                palette=sns.color_palette("viridis", 3)
            )
sns.set(rc = {'figure.figsize':(15,8)})
plt.title('Sentiment for the first 1000 articles', fontsize=20)
plt.xlabel('Sentiment', fontsize=16)
plt.ylabel('count', fontsize=16)

#apply sentiment on clean content 
partial_df["sentiment_Detokenize"] = partial_df['Detokenize'].apply(Sentiment_Analysis)
partial_df['sentiment_2']= partial_df['sentiment_Detokenize'].astype(str).str.replace("]", "")
partial_df['sentiment_2']= partial_df['sentiment_2'].astype(str).str.replace("[", "")

#plot 
sns.countplot(x = "sentiment_2", 
               data = partial_df.sort_values("sentiment_2"),
                palette=sns.color_palette("viridis", 3), 
            )
sns.set(rc = {'figure.figsize':(15,8)})
plt.title('Sentiment for the first 1000 articles', fontsize=20)
plt.xlabel('Sentiment', fontsize=16)
plt.ylabel('count', fontsize=16)

