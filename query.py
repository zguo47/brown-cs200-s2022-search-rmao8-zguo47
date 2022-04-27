import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from file_io import read_docs_file, read_title_file, read_words_file

class Query:
    def __init__(self, title_path: str, docs_path: str, words_path: str):
        self.title_path = title_path
        self.docs_path = docs_path
        self.words_path = words_path
        self.dictionary = {}
        self.id_to_pagerank = {}
        self.word_doc_relevance = {}
        read_title_file(self.title_path, self.dictionary)
        read_docs_file(self.docs_path, self.id_to_pagerank)
        read_words_file(self.words_path, self.word_doc_relevance)

if __name__ == "__main__":
        if len(sys.argv) - 1 == 3:
            query = Query(sys.argv[1], sys.argv[2], sys.argv[3])
            query.id_to_pagerank = {x: 1 for x in query.id_to_pagerank}
        elif len(sys.argv) - 1 == 4:
            if sys.argv[1] == "--pagerank":
                query = Query(sys.argv[2], sys.argv[3], sys.argv[4])
            else:
                raise TypeError('illegal argument')
        else:
            raise TypeError('Wrong number of arguments!')
        i = ""
        while i != "quit":
            i = input("please input a query, input 'quit' to quit: ")
            if i == "quit":
                break
            ws = i.split(" ")
            word = []
            STOP_WORDS = set(stopwords.words('english'))
            for w in ws:
                if w not in STOP_WORDS:
                    nltk_test = PorterStemmer()
                    word.append(nltk_test.stem(w)) 
            relevance = {id: sum([query.word_doc_relevance[w][id]for w in word if w in query.word_doc_relevance.keys()]) * query.id_to_pagerank[id]for id in query.dictionary.keys()}
            print([query.dictionary[sorted(relevance, key=relevance.get)[x]] for x in range(len(relevance) - 1, len(relevance) - 11, -1)])
            
                    
            print(word)
            print(i.upper())

        
