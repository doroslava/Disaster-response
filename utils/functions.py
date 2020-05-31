from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def tokenize(text):
    ''' 
    Tokenize text data.
        
        Parameters:
            text (string): Text to tokenize
        
        Returns:
            clean_text (list): List of words from tokenized text
    '''
    clean_tokens = []
        
    # Remove non word/number characters.
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Tokenize text.
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer.
    lemmatizer = WordNetLemmatizer()
    
    # Lematize words, convert to lower case, and remove stop words.
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens