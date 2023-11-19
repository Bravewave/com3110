"""
COM3110 Text Processing Information Retrieval Assignment
November 2023
Kiran Da Costa
"""

from math import sqrt, log


def cos_sim(vq: list[int | float], vd: list[int | float]) -> float:
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

        # Initialise document dictionary
        d_dict = dict()
        # Search the index for each query term and if present, add to the dictionary of relevant documents
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

        # Initalise query vector
        q_dict = dict()

        # Account for all weighting options
        match weighting:
            case "binary":
                for term in query:
                    # Since this is binary, if the term is already in the dictionary, move onto the next iteration
                    if term in q_dict:
                        continue
                    else:
                        # Otherwise, add it to dictionary with a count of 1
                        q_dict.update({term: 1})
            case "tf":
                for term in query:
                    # If the term is already in the query dictionary add 1 to its term count
                    if term in q_dict:
                        q_dict[term] += 1
                    else:
                        # Otherwise, add it to the dictionary with a count of 1
                        q_dict.update({term: 1})
            case "tfidf":
                # Code repeated from above to compute term frequency
                for term in query:
                    if term in q_dict:
                        q_dict[term] += 1
                    else:
                        q_dict.update({term: 1})

                # Multiply every term frequency by the inverse document frequency of the collection, resulting in a
                # dictionary of terms and their tfidf value
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
                # For every relevant document, search for that document in the document vector
                # No need to iterate through the keys of 'index' as we will not be using the terms themselves
                for docs in index.values():
                    for docid in d_dict.keys():
                        # Since this is binary, we simply append 1 if the document is found (i.e. the term is in the
                        # document) and 0 if it is not
                        if docid in docs:
                            d_dict[docid].append(1)
                        else:
                            d_dict[docid].append(0)
            case "tf":
                # For TF we append the term count rather than just a 1, but if the term isn't present, we do the same
                # as above
                for docs in index.values():
                    for docid in d_dict.keys():
                        if docid in docs:
                            d_dict[docid].append(docs[docid])  # 'docs[docid]' references the term count
                        else:
                            d_dict[docid].append(0)
            case "tfidf":
                # Repeat the same as above to compute the term frequency
                for docs in index.values():
                    for docid in d_dict.keys():
                        if docid in docs:
                            d_dict[docid].append(docs[docid])
                        else:
                            d_dict[docid].append(0)

                # Multiply every TF value in the vector by the collection IDF using list comprehension
                for docid, counts in d_dict.items():
                    d_dict[docid] = [tf * idf for tf, idf in zip(counts, list(idf_dict.values()))]

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
        # The 'sorted()' method sorts elements into ascending order by default and therefore must be reversed
        ranked_matches = dict(sorted(matches.items(), key=lambda match: match[1], reverse=True))

        return list(ranked_matches.keys())
