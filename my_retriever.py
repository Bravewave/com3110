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
        if word in self.index:
            return log(self.compute_number_of_documents() / len(self.index[word]))
        else:
            return 0

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

    def vectorise_query(self, query: list[str], weighting: str) -> list[float]:
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
        d_dict = dict()
        idf_dict = dict()

        for term in index.keys():
            idf_dict.update({term: self.idf(term)})

        for docs in index.values():
            for docid in docs.keys():
                d_dict.update({docid: []})

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
        relevant = self.relevant_docs(query)
        q_vec = self.vectorise_query(query, self.term_weighting)
        d_vecs = self.vectorise_docs(relevant, self.term_weighting)

        matches = dict()
        for docid, d_vec in d_vecs.items():
            matches[docid] = cos_sim(q_vec, d_vec)

        ranked_matches = dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))

        return list(ranked_matches.keys())
