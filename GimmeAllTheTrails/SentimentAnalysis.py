import pandas as pd
from textblob import TextBlob

from GimmeAllTheTrails.utils.Contraction_map import process_text, subjectivity,polarity,sent_Analysis



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

     @staticmethod
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
        print(self.dataset.head(10))


def make_sentiment_csv(written_csv_dir, outfile):
    """
    applies sentiment algo to data,
    :param written_csv_dir: location of written_reviews.csv file
    :param outfile: location to output resultant csv file
    :return: None
    """
    sentiment = Sentiment_Analysis(written_csv_dir)
    sentiment.dataset.to_csv(outfile)
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
    make_sentiment_csv("data/csv/written_reviews.csv",
                     "data/csv/sentiment.csv")



