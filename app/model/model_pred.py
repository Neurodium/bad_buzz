import tensorflow as tf
import re
import pickle
import spacy
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# API version
__version__ = "0.1.0"

# load TextVectorizer
vect_path = "model/tv_layer.pkl"

# load LSTM model
model = tf.keras.models.load_model('model/model.h5')


def doc_rewording(docs):
    # replace html tags
    docs = docs.lower().replace("&quot;", "")\
                    .replace("&amp;", "")\
                    .replace("&gt;", "")\
                    .replace("&lt;", "")
    return docs


def tweet_normalization(tweet):
    # Delete the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Delete URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Just keep letters and important punctuation
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Remove additional spaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet


def get_vectorizer(vect_path):
    # load textvectorizaton data
    from_disk = pickle.load(open(vect_path, "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])

    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])
    return new_v


def vectorize_tweet(tweet):
    # check if spacy dict is downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    # load vectorizer
    vectorizer = get_vectorizer(vect_path)
    # clean text
    tweet = tweet_normalization(doc_rewording(tweet))
    token_list = []
    # tokenize text
    for token in nlp(tweet):
        token_list.append(token.lemma_.lower().strip())  # lemmatize word before adding to list
    token_clean_list = list(filter(None, token_list))
    tweet = (" ").join(token_clean_list)
    vect_tweet = vectorizer(tweet)
    vect_tweet = vect_tweet.numpy().reshape(1, -1)
    return vect_tweet

def predict_pipeline(tweet):
    # predict if tweet sentiment is positive or negative
    sentiment = ''
    pred = model.predict(vectorize_tweet(tweet))
    if pred[0][0] > 0.5:
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    return sentiment

