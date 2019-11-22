import numpy as np
import pandas as pd
import pyprind, os, re, nltk, pickle,sqlite3
from tqdm import tqdm

pbar = pyprind.ProgBar(50000)  #50000 number of documents to be read in 
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()

#cleaning text data
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) #remove all html markup
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) #keep emoticon character
    text = re.sub('[\W]+', ' ', text.lower())+' '.join(emoticons).replace('-', '') #remove non-word character, convert to lower case 
    return text

#apply preprocessor function to the movie dataset
tqdm.pandas()
df = pd.read_csv(r'movie_data.csv')
df['review'] = df['review'].progress_apply(preprocessor)

#Process document into tokens
def tokenizer(text):
    return text.split()

#word stemming; transfomring word to its root form
#Note: In order to install the NLTK, you can simply execute pip install nltk
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#for example
tokenizer_porter('runners like running and thus they run')

#stop word removal e.g. is, has etc.(there are about 127 stopwords in NLTK)
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#example
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

#25000 training and 25000 test sets
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

#clean and tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower())+ ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


#Next we define a generator function, stream_docs, that reads in and returns one document at a time:
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

#verify if the stream_docs function work 
output_path = r'movie_data.csv'
next(stream_docs(path=output_path))

#take a document stream from the stream_docs function and return a particular 
##number of documents specified by the size parameter
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

#using hashing vectorizer for text processing
#Note CountVectorizer is not used because it requires holding all vocabulary in memory
#TfidfVectorizer is not used because it requires keeping all feature vectors of the
#training data in memory to calculate inverse document frequencies.

#We will use Hashingvectorizer, this is data independent
#uisng SGD classifier algorithm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None,tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1)
doc_stream = stream_docs(path=output_path)

#incremental training of 45000 documents
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

#evaluate performance of the model with 5000 documents left
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

#update the model with the prediction 
clf = clf.partial_fit(X_test, y_test)

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
    pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'),'wb'), protocol=4)
    pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'),protocol=4)

#quick check if the pickled files and vectorizer.py are working
from movieclassifier.vectorizer import vect
clf = pickle.load(open(os.path.join('./movieclassifier/pkl_objects','classifier.pkl'), 'rb'))

label = {0:'negative', 1:'positive'}
example = ['the movie is fair']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))