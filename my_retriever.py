"""
COM3110 Text Processing Information Retrieval Assignment
November 2023
Kiran Da Costa
"""

from math import sqrt, log


def cos_sim(vq: list | tuple, vd: list | tuple) -> float:
    """
    Calcualtes the cosine similarity between a query vector and a document vector.
    For our purposes the query vector will never change and therefore can be omitted from the denominator in the
    interest of efficiency.
    :param vq: query vector, as a list of either intgers or floats
    :param vd: document vector, as a list of either integers or floats
    :return: a single float representing the cosine similarity between the document and query vectors
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
        """
        Calculates the inverse document frequency of a word within a document
        :param word: the term being considered
        :return: the IDF of that term for the corpus
        """
        if word in self.index:
            # No. of documents in the collection divided by the no. of occurances of a word in that collection
            return log(self.compute_number_of_documents() / len(self.index[word]))
        else:
            # If the word is not in the collection, return zero
            return 0

    def relevant_docs(self, query: list[str]) -> dict[str, dict[int, int]]:
        """
        Returns a dictionary mapping the words in a query to documents which contain them
        :param query: a list of strings, each being a query term
        :return: a dictionary of terms, each mapped to a dictionary of document IDs and the number of times that term
        occurs within them
        """
        d_dict = dict()
        for word in query:
            if word in self.index:
                d_dict.update({word: self.index[word]})

        return d_dict

    def vectorise_query(self, query: list[str], weighting: str) -> list[float]:
        """
        Takes a query and turns it into a vector according to the term weighting used
        :param query: a list of individual query terms
        :param weighting: either 'binary', 'tf', or 'tfidf'
        :return: the query vector to be used in similarity calculations
        """
        q_dict = dict()

        match weighting:
            case "binary":
                for term in query:
                    if term in q_dict:
                        continue
                    else:
                        q_dict.update({term: 1})
            case "tf":
                for term in query:
                    if term in q_dict:
                        q_dict[term] += 1
                    else:
                        q_dict.update({term: 1})
            case "tfidf":
                for term in query:
                    if term in q_dict:
                        q_dict[term] += 1
                    else:
                        q_dict.update({term: 1})

                for term, tf in q_dict.items():
                    q_dict[term] = tf * self.idf(term)

        return list(q_dict.values())

    def vectorise_docs(self, index: dict[str, dict[int, int]], weighting: str) -> dict[int, list[float]]:
        """
        Formats the documents relevant to a query into a set of workable vectors that can be used in cosine
        similarity calculations
        :param index: a dictionary of all documents relevant to the query
        :param weighting: weighting system to be used represented as a string (one of 'binary', 'tf', or 'tfidf')
        :return: a dictionary of document IDs mapped to their document vectors
        """

        # Compute IDF for the query-relevant documents in one pass
        idf_dict = dict()

        for term in index.keys():
            idf_dict.update({term: self.idf(term)})

        # Initliase document vector dictionary
        d_dict = dict()

        # Populate dictionary with document IDs of relevant documents
        for docs in index.values():
            for docid in docs.keys():
                d_dict.update({docid: []})

        # Ensure to allow for each weighting system
        match weighting:
            case "binary":
                for docs in index.values():
                    for docid in d_dict.keys():
                        if docid in docs:
                            d_dict[docid].append(1)
                        else:
                            d_dict[docid].append(0)
            case "tf":
                for term, docs in index.items():
                    for docid, count in d_dict.items():
                        if docid in docs:
                            d_dict[docid].append(docs[docid])
                        else:
                            d_dict[docid].append(0)
            case "tfidf":
                for term, docs in index.items():
                    for docid, count in d_dict.items():
                        if docid in docs:
                            d_dict[docid].append(docs[docid])
                        else:
                            d_dict[docid].append(0)

                for docid, counts in d_dict.items():
                    d_dict[docid] = [tf*idf for tf, idf in zip(counts, list(idf_dict.values()))]

        return d_dict

    # Method performing retrieval for a single query (which is
    # represented as a list of preprocessed terms). ​Returns list
    # of doc ids for relevant docs (in rank order).

    def for_query(self, query):
        # Retrieve relevant documents for this query
        relevant = self.relevant_docs(query)
        # Turn the query into a workable vector
        q_vec = self.vectorise_query(query, self.term_weighting)
        # Turn the relevant documents into a set of workable vectors
        d_vecs = self.vectorise_docs(relevant, self.term_weighting)

        # Initialise dictionary of matches
        matches = dict()
        # Compute the cosine similarity between the query and each relevant document
        for docid, d_vec in d_vecs.items():
            matches[docid] = cos_sim(q_vec, d_vec)

        # Rank the matches from most to least relevant
        ranked_matches = dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))

        return list(ranked_matches.keys())
