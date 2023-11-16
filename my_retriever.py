from math import sqrt
import pickle


def cos_sim(vq: list | tuple, vd: list | tuple) -> float:
    """
    Calcualtes the cosine similarity between a query vector and a document vector.
    For our purposes the query vector will never change and therefore can be omitted from the denominator in the
    interest of efficiency.
    :param vq: query vector
    :param vd: document vector
    :return: the cosine similarity between the document and query vectors
    """

    # Calculate the dot product using zip(), then divide by the magnitude of document vector
    return sum(x * y for x, y in zip(vq, vd)) / sqrt(sum(y ** 2 for y in vd))


class Retrieve:
    # Create new Retrieve object storing index and term weighting 
    # scheme. ​(You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.doc_ids = set()
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
    def compute_number_of_documents(self):
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    def idf(self, word: str) -> float:
        return self.compute_number_of_documents() / len(index[word])

    def relevant_docs(self, query: list[str]) -> dict[str, dict[int, int]]:
        """
        Returns a dictionary mapping the words in a query to documents which contain them
        :param query: the query to be processed
        :return: a dictionary of words, each mapped to a dictionary of document IDs and the number of times that word
        occurs within them
        """
        d_dict = dict()
        for word in query:
            if word in self.index:
                d_dict.update({word: self.index[word]})

        return d_dict

    def vectorise_query(self, query: list[str]) -> list[int]:
        """
        Turns a list of query terms into an array of how many times each term appears in that query
        :param query: a tuple consisting of the query id and a list of query terms
        :return: a list of integers representing the frequency of each query term within the query
        """
        q_dict = dict()

        for word in query:
            if word in q_dict:
                q_dict[word] += 1
            else:
                q_dict.update({word: 1})

        return list(q_dict.values())

    # Method performing retrieval for a single query (which is
    # represented as a list of preprocessed terms). ​Returns list
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        match self.term_weighting:
            case "binary":
                relevant = self.relevant_docs(query)
                hits = set()
                for doc in relevant.values():
                    hits.update(doc.keys())
                return list(hits)
            case "tf":
                relevant = self.relevant_docs(query)
                sorted_docids = None
                for _, docs in relevant.items():
                    sorted_docids = dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)).keys()
                return list(sorted_docids)
            case "tfidf":
                docs = self.relevant_docs(query)
                return list(range(1, 11))
            case _:
                # Due to the command line input validation in IR_engine.py, this case should theoretically never
                # actually run, but is here just in case
                raise Exception("Invalid Weighting - use either 'binary', 'tf' or 'tfidf'")


QUERY = ['what', 'articles', 'exist', 'which', 'deal', 'with', 'tss', 'time', 'sharing', 'system', 'an', 'operating', 'system', 'for', 'ibm', 'computers']

# with open('IR_data.pickle', 'rb') as data_in:
#     all_data = pickle.load(data_in)
# choice = 'index_stoplist_no_stemming_no'
# index = all_data[choice]
#
# choice = 'queries_stoplist_no_stemming_no'
# queries = all_data[choice]
#
# print(Retrieve(index, "tf"))
