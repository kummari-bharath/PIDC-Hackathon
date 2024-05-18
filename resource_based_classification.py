import csv
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

# # Function to calculate sentiment score for a word
def calculate_word_sentiment_score(word, pos_tag):
    synsets = list(swn.senti_synsets(word, pos_tag))
    if synsets:
        # Use the average of all available synsets
        score = sum([synset.pos_score() - synset.neg_score() for synset in synsets]) / len(synsets)
        return score
    else:
        return 0

# # Function to perform sentiment analysis on a sentence
def analyze_sentiment(review):
    tokens = word_tokenize(review.lower())
    pos_tags = nltk.pos_tag(tokens)
    sentiment_score = 0
    
    for word, pos_tag in pos_tags:
        if pos_tag.startswith('J'):  # Adjectives
            pos_tag = 'a'
        elif pos_tag.startswith('V'):  # Verbs
            pos_tag = 'v'
        elif pos_tag.startswith('N'):  # Nouns
            pos_tag = 'n'
        elif pos_tag.startswith('R'):  # Adverbs
            pos_tag = 'r'
        else:
            pos_tag = None
        
        if pos_tag:
            score = calculate_word_sentiment_score(word, pos_tag)
            sentiment_score += score
    
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# # Example sentences
# sentences = [
#     "I love this movie, it's fantastic!",
#     "This restaurant has terrible service.",
#     "The weather today is neither good nor bad.",
#     "He runs very fast."
# ]

# reading the dat from the datset and sending the reviews stament into senti scorer

f=open("C:/Users/Guest User/OneDrive/Desktop/sentiment_analysis/tripadvisor_hotel_reviews.csv","r",encoding='utf-8')
rows= csv.reader(f)

reviews=[]
with open("C:/Users/Guest User/OneDrive/Desktop/sentiment_analysis/tripadvisor_hotel_reviews.csv", 'r', newline='', encoding='utf-8') as csvfile:
    rows = csv.reader(csvfile)
   
    for row in rows:
        # Assuming the column you want is the first one (index 0)
        reviews.append(row[0])
reviews_updated=reviews[1:]


#writing into the new csv file with 
q=open("results_of_sentiment.csv","w",newline='',encoding='utf-8')
fields=["Review","Sentiment"]
writer =csv.DictWriter(q,fieldnames=fields)
writer.writeheader()


#Perform sentiment analysis on each sentence
for review in reviews_updated:
    sentiment = analyze_sentiment(review)
    # print(f"review: {review}")
    # print(f"Sentiment: {sentiment}")
    writer.writerow({"Review":review,"Sentiment":sentiment})
    