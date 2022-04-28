"""
The indexer of Search.
"""
from collections import defaultdict
import math
import re
import copy
from typing import final
from xml.dom.minidom import Element
import xml.etree.ElementTree as et
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from numpy import tile

from file_io import write_docs_file, write_title_file, write_words_file


class Indexer:
    def __init__(self, xml_path: str, title_path: str, docs_path: str, words_path: str):
        """
        doc string goes here!
        """
        if len(sys.argv) - 1 != 4:
            raise TypeError('Wrong number of arguments!')

        self.xml_path = xml_path
        self.title_path = title_path
        self.docs_path = docs_path
        self.words_path = words_path
        self.word_corpus = set()
        self.dictionary = {}
        self.title_to_id = defaultdict(lambda: -1)
        self.id_to_link = {}
        self.word_doc_count = defaultdict(lambda: defaultdict(lambda: 0))
        self.word_doc_relevance = {}
        self.id_to_pagerank = {}

        root: Element = et.parse(self.xml_path).getroot()
        all_pages: et.ElementTree = root.findall("page")

        for page in all_pages:
            title: str = page.find('title').text.strip()
            id: int = int(page.find('id').text.strip())
            self.dictionary[id] = title
            self.title_to_id[title] = id
            text: str = page.find('text').text.strip()

            words = self.token_stop_stem(title, id, text)
            self.word_corpus.update(words)
            self.fill_word_doc_count(words, id)

        self.fill_word_doc_relevance()
        # self.refill_id_to_link()
        self.fill_id_to_pagerank()

        write_title_file(self.title_path, self.dictionary)
        write_docs_file(self.docs_path, self.id_to_pagerank)
        write_words_file(self.words_path, self.word_doc_relevance)

    def token_stop_stem(self, title, id, text):

        STOP_WORDS = set(stopwords.words('english'))
        n_regex = '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        str = re.findall(n_regex, title + ' ' + text)
        final = []

        for word in str:
            brackets = "[["
            if brackets in word:
                word = word.replace('[[', '').replace(']]', '')
                words = word.split("|")
                if id not in self.id_to_link:
                    if self.title_to_id[words[0]] != id:
                        self.id_to_link[id] = [words[0]]
                else:
                    if self.title_to_id[words[0]] != id and words[0] not in self.id_to_link[id]:
                        self.id_to_link[id].append(words[0])
                if len(words) > 1:
                    new_words = re.findall(n_regex, ''.join(words[1:]))
                    str.extend(new_words)
                else:
                    new_words = re.findall(n_regex, words[0])
                    str.extend(new_words)
            else:
                if word not in STOP_WORDS:
                    nltk_test = PorterStemmer()
                    final.append(nltk_test.stem(word))
        return final

    def fill_word_doc_count(self, words, id):

        for word in words:

            self.word_doc_count[word][id] += 1

    def fill_word_doc_relevance(self):

        max_occur = defaultdict(lambda: 0)
        word_number = defaultdict(lambda: 0)

        self.word_doc_relevance = copy.deepcopy(self.word_doc_count)

        for word in self.word_corpus:
            for id in self.dictionary.keys():
                if self.word_doc_count[word][id] > max_occur[id]:
                    max_occur[id] = self.word_doc_count[word][id]

                if self.word_doc_count[word][id] != 0:
                    word_number[word] += 1

        for word in self.word_corpus:
            for id in self.dictionary.keys():
                self.word_doc_relevance[word][id] = self.word_doc_relevance[word][id] / max_occur[id]

                idf = math.log(len(self.dictionary) / word_number[word])

                self.word_doc_relevance[word][id] = self.word_doc_relevance[word][id] * idf

    # def refill_id_to_link(self):

    #     for id in self.id_to_link.keys():
    #         new_list = []
    #         for title in self.id_to_link[id]:
    #             if title in self.title_to_id:
    #                 t = self.title_to_id[title]
    #                 if id != t and t not in new_list:
    #                     new_list.append(t)
    #         self.id_to_link[id] = copy.deepcopy(new_list)

    def fill_id_to_pagerank(self):

        l = len(self.dictionary.keys())

        r = defaultdict(lambda: 0)

        r_n = defaultdict(lambda: 1 / l)

        distance = 1

        while distance > 0.001:
            r = copy.deepcopy(r_n)
            for id2 in self.dictionary.keys():
                r_n[id2] = 0
                for id1 in self.dictionary.keys():
                    if id1 not in self.id_to_link:
                        self.id_to_link[id1] = list(
                            self.dictionary.values())
                        self.id_to_link[id1].remove(self.dictionary[id1])

                    if self.dictionary[id2] not in self.id_to_link[id1]:
                        r_n[id2] = r_n[id2] + r[id1] * 0.15 / l

                    else:
                        r_n[id2] = r_n[id2] + r[id1] * (0.15 / l + (
                            1 - 0.15) * 1 / len([x for x in self.id_to_link[id1]
                                                 if x in self.dictionary.values()]))

            distance = math.sqrt(
                sum([(r_n[x] - r[x]) ** 2 for x in r_n.keys()]))

        self.id_to_pagerank = copy.deepcopy(r_n)


if __name__ == "__main__":
    i = Indexer(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
