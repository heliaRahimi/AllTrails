import glob
import os
import pandas as pd
import ast
from GimmeAllTheTrails.utils.Contraction_map import process_text, subjectivity,polarity,sent_Analysis
from textblob import TextBlob
# import plotly.express as px
# import matplotlib.pyplot as plt
# import pyLDAvis
# import pyLDAvis.sklearn
# import pyLDAvis.gensim_models as gensimvis
#
# from GimmeAllTheTrails.utils.LDA import optimal_lda_model


class Sentiment_Analysis(object):

     """
    load written reviews dataset in this class and clean them
    """
     def __init__(self, file_dir):
        self.file_dir = file_dir
        self.dataset = pd.read_csv(self.file_dir)
        self.cleaned_data = self.cleaning_trails_reviews()

    #Create a function to get the subjectivity
     def subjectivity_2(self):
         return TextBlob(self.dataset["processed_reviews"]).sentiment.subjectivity

    #Create a function to get the polarity
     def polarity_2(self):
        return TextBlob(self.dataset["processed_reviews"]).sentiment.polarity

     def sent_Analysis_2(score):
         if score <0:
             return "Negative"
         elif score == 0:
             return "Neutral"
         else:
             return "Positive"


     def cleaning_trails_reviews(self) -> pd.DataFrame:
        """
        get the lat lon pairs and associated name pairs for each hiking trail location
        :return:
        """
        # get coords and trail id
        trail_reviews = self.dataset["reviews"]
        print(self.dataset.shape)
        # Removing missing values if any
        nan_value = float("NaN")
        self.dataset.replace(" ",nan_value,inplace= True)
        self.dataset = self.dataset.dropna()
        self.dataset['text_reviews'] = self.dataset["reviews"].apply(lambda x:x.strip(",").replace('\n\n\n','').replace('"',"").replace("'","").replace('•',',').lstrip(', '))
        print(self.dataset['text_reviews'])
        # converting the list to string
        self.dataset['processed_reviews'] = self.dataset['text_reviews'].apply(lambda x: x[1:-1])
        self.dataset['clean_reviews'] =  self.dataset['processed_reviews'].apply(process_text)
        print(self.dataset['clean_reviews'])
        self.dataset['clean_reviews'] = self.dataset['clean_reviews'].apply(lambda x : ' '.join(x))
        print(self.dataset['clean_reviews'])
        self.dataset['Subjectivity'] = self.dataset['processed_reviews'].apply(subjectivity)
        self.dataset['Polarity'] = self.dataset['processed_reviews'].apply(polarity)
        self.dataset['Analysis'] = self.dataset['Polarity'].apply(sent_Analysis)
        self.dataset.to_csv(r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv\sentiment.csv")
        print(self.dataset.head(10))
#################################################################################


        # #Positive tweets
        # positive_words = ' '.join([text for text in self.dataset['clean_reviews'][self.dataset['Analysis'] == "Positive"]])
        # #wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(positive_words)
        # #plt.figure(figsize=(10, 7))
        # #plt.imshow(wordcloud, interpolation="bilinear")
        # #plt.axis('off')
        # #plt.figtext(.5,.8,title,fontsize = 20, ha='center')
        # #plt.show()
        #
        # negative_words = ' '.join([text for text in self.dataset['clean_reviews'][self.dataset['Analysis'] == "Negative"]])
        # # Create wordcloud
        #
        # print(self.dataset.loc[self.dataset['Analysis'] == "Negative"])
        # negative = []
        #  # build negative review list
        # for i in range(len(self.dataset.loc[self.dataset['Analysis'] == "Negative"])):
        #     negative.append(self.dataset['clean_reviews'][i])
        #
        # neg_reviews = pd.DataFrame(negative,columns = ['negative_reviews'])
        # best_lda_model, dtm_tfidf, tfidf_vectorizer = optimal_lda_model(neg_reviews, 'negative_reviews')
        #
        # # Topic Modelling Visualization for the Negative Reviews
        # #vis_data = gensimvis.prepare(best_lda_model, dtm_tfidf, tfidf_vectorizer)
        # vis_data = pyLDAvis.sklearn.prepare(best_lda_model, dtm_tfidf, tfidf_vectorizer)
        # pyLDAvis.show(vis_data)


if __name__ == "__main__":
    data = Sentiment_Analysis(r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv\written_reviews.csv")
    print(data)
    """
    # plot the polarity and subjectivity
        fig = px.scatter(self.dataset[2:10],
                 x='Polarity',
                 y='Subjectivity',
                 color = 'Analysis',
                 size='Subjectivity')

#add a vertical line at x=0 for Netural Reviews
        fig.update_layout(title='Sentiment Analysis',
                  shapes=[dict(type= 'line',
                               yref= 'paper', y0= 0, y1= 1,
                               xref= 'x', x0= 0, x1= 0)])
        fig.show()"""

