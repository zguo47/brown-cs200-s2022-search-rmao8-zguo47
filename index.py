"""
The indexer of Search.
"""

import re
from typing import final
from xml.dom.minidom import Element
import xml.etree.ElementTree as et
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from numpy import tile


class Indexer:
    def __init__(self, xml_path: str, title_path: str, docs_path: str, words_path: str):
        """
        doc string goes here!
        """
        self.xml_path = xml_path
        self.title_path = title_path
        self.docs_path = docs_path
        self.words_path = words_path
        self.dictionary = {}
        self.id_to_link = {}

        root: Element = et.parse(self.xml_path).getroot()
        all_pages: et.ElementTree = root.findall("page")

        for page in all_pages:
            title: str = page.find('title').text
            id: int = int(page.find('id').text.strip())
            self.dictionary[id] = title
            text: str = page.find('text').text.strip()

            self.token_stop_stem(title, id, text)

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
                self.id_to_link[id] = words[0]
                if len(words) > 1:
                    new_words = re.findall(n_regex, ''.join(words[1:]))
                    str.extend(new_words)
                else:
                    new_words = re.findall(n_regex, words[0])
                    str.extend(new_words)
            else:
                word = word.lower()
                if word not in STOP_WORDS:
                    nltk_test = PorterStemmer()
                    final.append(nltk_test.stem(word))
        print(final)


i = Indexer("SmallWiki.xml", "1", "2", "3")
