#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import time
from multiprocessing import cpu_count

import wget
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DIRECTORY_PATH = 'word2vec/'
WIKI_PATH = DIRECTORY_PATH + 'latest-pages-articles.xml.bz2'
WIKI_TEXT_FILENAME = DIRECTORY_PATH + 'latest-text.txt'
WIKI_TEXT_TOKENS_FILENAME = DIRECTORY_PATH + 'latest-text-tokens.txt'
VECTORS_MODEL_FILENAME = DIRECTORY_PATH + 'gensim.{}d.data.model'.format(50)
VECTORS_TEXT_FILENAME = DIRECTORY_PATH + 'gensim.{}d.data.txt'.format(50)
WIKI_LATEST_URL = 'https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2'


def process_wiki_to_text(input_filename, output_text_filename):
    if os.path.isfile(output_text_filename):
        logging.info('Skipping process_wiki_to_text(). File already exist: {}'.format(output_text_filename))
        return

    with open(output_text_filename, 'w', encoding="UTF-8") as out:
        # Open the Wiki Dump with gensim
        wiki = WikiCorpus(input_filename, dictionary={}, processes=cpu_count())
        wiki.metadata = True
        texts = wiki.get_texts()

        for i, article in enumerate(texts):
            # article[1] refers to the name of the article.
            text_list = article[0]
            sentences = text_list

            # Write each page in one line
            text = ' '.join(sentences) + '\n'
            out.write(text)


def tokenize_text(input_filename, output_filename):
    if os.path.isfile(output_filename):
        logging.info('Skipping tokenize_text(). File already exists: {}'.format(output_filename))
        return

    with open(output_filename, 'w', encoding="UTF-8") as out:
        with open(input_filename, 'r', encoding="UTF-8") as inp:
            for i, text in enumerate(inp.readlines()):
                tokenized_text = ' '.join(get_words(text))

                out.write(tokenized_text + '\n')


def get_words(text):
    import MeCab
    mt = MeCab.Tagger('-d C:/Users/User/anaconda3/Lib/site-packages/unidic_lite/dicdir')

    mt.parse('')

    parsed = mt.parseToNode(text)
    components = []

    while parsed:
        components.append(parsed.surface)
        parsed = parsed.next

    return components


def generate_vectors(input_filename, output_filename, output_filename_2):
    if os.path.isfile(output_filename):
        logging.info('Skipping generate_vectors(). File already exists: {}'.format(output_filename))
        return Word2Vec.load(output_filename)

    start = time.time()

    model = Word2Vec(LineSentence(input_filename),
                     vector_size=50,
                     window=5,
                     min_count=5,
                     workers=4,
                     epochs=5)

    model.save(output_filename)
    model.wv.save_word2vec_format(output_filename_2, binary=False)

    logging.info('Finished generate_vectors(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))

    return model


def get_word2vec_model():
    if not os.path.isfile(WIKI_PATH):
        wget.download(WIKI_LATEST_URL)

    process_wiki_to_text(WIKI_PATH, WIKI_TEXT_FILENAME)
    tokenize_text(WIKI_TEXT_FILENAME, WIKI_TEXT_TOKENS_FILENAME)
    return generate_vectors(WIKI_TEXT_TOKENS_FILENAME, VECTORS_MODEL_FILENAME, VECTORS_TEXT_FILENAME)
