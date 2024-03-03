from src import subreddit
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
import pyLDAvis.gensim_models
import gensim

"""
1. Get the search subreddit data for each query topic
2. Using each of their titles (string) create a list
3. Use topic modelling to find the most common topics
"""

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation and not char.isdigit()])
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]

def main():
    PRIVACY_TOPICS = [
        "security",
        "privacy"
    ]

    # Get the search subreddit data for each query topic to list of strings
    documents = []

    for query in PRIVACY_TOPICS:
        data = subreddit.search_subreddit_data_max(query)
        documents.extend([submission_result.title for submission_result in data])

    # Preprocess the titles
    processed_titles = [preprocess(title) for title in documents]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_titles)

    # Convert dictionary to a bag of words corpus
    corpus = [dictionary.doc2bow(text) for text in processed_titles]

    # Apply LDA
    lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    # Print the topics identified by LDA
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        print(topic)

    # Assign the topics to the titles
    submission_topics = [max(
        lda_model[dictionary.doc2bow(preprocess(title))],
        key=lambda x: x[1]
    )[0] for title in documents]

    # Output the data
    # for title, topic in zip(documents, submission_topics):
    #     print(f'Title: "{title}" has been categorized under Topic: {topic}')

    # Prepare the visualization data
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, mds="mmds")

    pyLDAvis.save_html(vis_data, "./output/lda_visualization.html")

if __name__ == "__main__":
    main()
