import string 
from nltk.stem import PorterStemmer

class preprocessing:
    def __init__(self,chunks,corpus):
        self.chunks = chunks
        self.corpus = corpus

    def tokenized_corpus(self):
        chunk_texts = [c['text'] for c in self.chunks]
        tokenize_corpus = []
        stemmer = PorterStemmer()

        for chunk in chunk_texts:
            if not isinstance(chunk,str):
                chunk = str(chunk)
            text = chunk.lower()
            for char in string.punctuation:
                text = text.replace(char,' ')
            words = text.split()
            stemmed_words = [stemmer.stem(w) for w in words]
            tokenize_corpus.append(stemmed_words)

        return tokenize_corpus
    

    def vocabulary_construction(self,corpus):
        vocab = {}
        unique_words = set()
        for word in corpus:
            unique_words.update(word)
        for i,w in enumerate(unique_words):
            vocab[w] = i

        return vocab
