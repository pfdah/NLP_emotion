import nltk
import pickle
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import LabelEncoder 



def preprocess(dataframe):
    label_enc = LabelEncoder()
    tokenizer = nltk.NLTKWordTokenizer()  
    stemmer =  LancasterStemmer()

    dataframe['emotion'] = label_enc.fit_transform(dataframe['emotion'])
    
    with open('label_enc.pkl', 'wb') as handle:
        pickle.dump(label_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    dataframe['text_token'] = ''
    for i in range(len(dataframe['text'])):
        dataframe['text_token'][i] = tokenizer.tokenize(dataframe['text'][i])
        dataframe['text_token'][i] = [stemmer.stem(j) for j in dataframe['text_token'][i]]
    return dataframe