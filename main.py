import json
import logging
import os

import numpy as np
import pandas as pd

from word2vec.generate_vectors import get_word2vec_model

# Tidy data
TIDY_DATA_DIRECTORY = 'data/tidy/'
JLPT_DF_FILENAME = TIDY_DATA_DIRECTORY + 'jlpt-df.csv'
KANJI_LEVELS_DF_FILENAME = TIDY_DATA_DIRECTORY + 'kanji-levels-df.csv'
KANA_CHAR_COUNT_DF_FILENAME = TIDY_DATA_DIRECTORY + 'kana-char-count-df.csv'
FREQ_LISTS_DF_FILENAME = TIDY_DATA_DIRECTORY + 'freq-list-df.csv'
JLPT_WITH_FREQ_DF_FILENAME = TIDY_DATA_DIRECTORY + 'jlpt-with-freq-df.csv'

# JLPT data
JLPT_DIRECTORY = 'data/jlpt/'
JLPT_FILENAMES_AND_BIN_NUMBERS = [(JLPT_DIRECTORY + jlpt_file_name, bin_number)
                                  for jlpt_file_name, bin_number in
                                  [('jlpt-voc-4-extra.utf', '4'),
                                   ('jlpt-voc-3-extra.utf', '3'),
                                   ('jlpt-voc-2-extra.utf', '2'),
                                   ('jlpt-voc-1-extra.utf', '1')]]

# Kanji levels data
KANJI_LEVELS_FILENAME = 'data/levels/' + 'kanji-levels.json'

# Frequency list data
FREQ_LISTS_DIRECTORY = 'data/freq_lists/'
FREQ_LISTS_FILENAMES = [FREQ_LISTS_DIRECTORY + freq_list_filename for freq_list_filename in
                        ['netflix_unidic_3011_no_names_word_freq_report.txt',
                         'word_freq_report.txt']]


def load_jlpt_df():
    """
    Loads JLPT kanji and kana and sorts them into levels 1-4. Returns a dataframe. Loads it if already saved.
    """

    if os.path.isfile(JLPT_DF_FILENAME):
        logging.info('Skipping load_jlpt_df(). File already exist: {}'.format(JLPT_DF_FILENAME))
        return pd.read_csv(JLPT_DF_FILENAME, encoding='utf8', keep_default_na=False)

    jlpt_vocab_dict = {'jlpt': [], 'kanji': [], 'kana': []}
    kanji_list = set()

    for file_name, jlpt in JLPT_FILENAMES_AND_BIN_NUMBERS:
        with open(file_name, 'r', encoding='utf8') as f:
            for line in f.readlines():
                # skip if line is a comment or empty
                if line.startswith("#") or line.startswith("\n"):
                    continue

                word_list = line[:-1].split(" ")

                # skip words that have tilde or appear in lower difficulty bins
                if '~' in word_list[0] or word_list[0] in kanji_list:
                    continue

                # add JLPT difficulty
                jlpt_vocab_dict['jlpt'].append(int(jlpt))

                # add both kanji and kana if it has it
                if len(word_list) > 1 and word_list[1].startswith('[') and '（' not in word_list[1]:
                    jlpt_vocab_dict['kanji'].append(word_list[0])
                    jlpt_vocab_dict['kana'].append(word_list[1][1:-1])
                    # add to list of used kanji
                    kanji_list.add(word_list[0])
                # add just kana as it has no kanji
                else:
                    jlpt_vocab_dict['kanji'].append('')
                    jlpt_vocab_dict['kana'].append(word_list[0])

    df = pd.DataFrame(jlpt_vocab_dict)
    df.to_csv(JLPT_DF_FILENAME, index=False, encoding='utf8')
    return df


def load_kanji_levels_df():
    """
    Loads the kanji levels JSON and makes a dataframe out of it. Normalizes the levels between 0 and 1.
    Assumes levels are from 1 to 10
    """

    if os.path.isfile(KANJI_LEVELS_DF_FILENAME):
        logging.info('Skipping load_kanji_levels_df(). File already exist: {}'.format(KANJI_LEVELS_DF_FILENAME))
        return pd.read_csv(KANJI_LEVELS_DF_FILENAME, encoding='utf8', keep_default_na=False)

    kanji_levels_dict = {'kanji': [], 'level': []}

    with open(KANJI_LEVELS_FILENAME, 'r', encoding='utf8') as f:
        for kanji, level in json.load(f).items():
            kanji_levels_dict['kanji'].append(kanji)
            kanji_levels_dict['level'].append((int(level) - 1) / 9)

    df = pd.DataFrame(kanji_levels_dict)
    df.to_csv(KANJI_LEVELS_DF_FILENAME, index=False, encoding='utf8')
    return df


def generate_kana_char_count_vectors(kana):
    """
    Generates the character count vectors for each kana
    """

    if os.path.isfile(KANA_CHAR_COUNT_DF_FILENAME):
        logging.info(
            'Skipping generate_kana_char_count_matrix(). File already exist: {}'.format(KANA_CHAR_COUNT_DF_FILENAME))
        return pd.read_csv(KANA_CHAR_COUNT_DF_FILENAME, encoding='utf8', keep_default_na=False)

    kana_char_count_vectors = {'kana': kana}
    char_columns = set()

    for row, word in enumerate(kana):
        # set the count of each char to 0 for current word
        for char in char_columns:
            kana_char_count_vectors[char].append(0)

        for char in word:
            # char already has a column
            if char in char_columns:
                kana_char_count_vectors[char][-1] += 1
            # there is no char column, make one
            else:
                kana_char_count_vectors[char] = [0] * (row + 1)
                kana_char_count_vectors[char][-1] = 1
                char_columns.add(char)

    df = pd.DataFrame(kana_char_count_vectors)
    df.to_csv(KANA_CHAR_COUNT_DF_FILENAME, index=False, encoding='utf8')
    return df


def load_freq_list_df(file_name):
    """
    Takes two types of column schema and returns the word/kanji/kana and percentage.

    Primary column schema:
    eg. file_name = '/data/freq_lists/word_freq_report.txt'
    from https://github.com/chriskempson/japanese-subtitles-word-frequency-list
    col_names = ['occurrences', 'word', 'freq_group', 'freq_rank', 'percentage', 'cum_percentage', 'part_of_speech'?]

    Secondary column schema:
    eg. file_name = '/data/freq_lists/netflix_unidic_3011_no_names_word_freq_report.txt'
    col_names = ['occurrences', 'kanji', 'hiragana', 'kana', 'part_of_speech',
                 'anotherpos', 'freq_group', 'freq_rank', 'percentage', 'cum_percentage']
    """

    is_schema_one = None
    freq_list_dict = {'kanji': [], 'kana': [], 'percentage': []}
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f.readlines():
            row = line[:-1].split("\t")

            # not enough columns
            if len(row) < 6:
                continue

            # set schema
            if is_schema_one is None:
                is_schema_one = len(row) < 8

            # exclude particles
            if is_schema_one and len(row) == 7 and 'prt' in row[5]:
                continue

            if is_schema_one:
                freq_list_dict['kanji'].append(row[1])
                freq_list_dict['kana'].append('')
                freq_list_dict['percentage'].append(float(row[4]))
            else:
                # skip if latin characters are present
                if row[1].upper().isupper():
                    continue
                # only add kanji if it's not the same as kana
                freq_list_dict['kanji'].append('' if row[1] == row[2] else row[1])
                freq_list_dict['kana'].append(row[2])
                freq_list_dict['percentage'].append(float(row[8]))

    return pd.DataFrame(data=freq_list_dict)


def load_merged_freq_list():
    """
    Load frequency lists into one dataframe
    """

    if os.path.isfile(FREQ_LISTS_DF_FILENAME):
        return pd.read_csv(FREQ_LISTS_DF_FILENAME, encoding='utf8', keep_default_na=False)

    # load and merge frequency lists
    freq_list_df = pd.concat([load_freq_list_df(file_name) for file_name in FREQ_LISTS_FILENAMES])
    # merge the same kanji, kana into one row
    freq_list_df.groupby(['kanji', 'kana'])['percentage'].transform('sum')
    # normalize percentage
    total_freq = freq_list_df['percentage'].sum()
    freq_list_df['percentage'].transform(lambda x: x / total_freq)

    freq_list_df.to_csv(FREQ_LISTS_DF_FILENAME, index=False, encoding='utf8')
    return freq_list_df


def add_jlpt_to_freq_list(jlpt, freq_list):
    """
    Adds a JLPT level column to the frequency list dataframe
    1-4 if it has a corresponding JLPT level
    0 if it does not
    """

    if os.path.isfile(JLPT_WITH_FREQ_DF_FILENAME):
        logging.info(
            'Skipping add_jlpt_to_freq_list(). File already exist: {}'.format(JLPT_WITH_FREQ_DF_FILENAME))
        return pd.read_csv(JLPT_WITH_FREQ_DF_FILENAME, encoding='utf8', keep_default_na=False)

    # get the references to the numpy arrays
    kanji_list = jlpt['kanji'].values
    kana_list = jlpt['kana'].values
    level_list = jlpt['jlpt'].values

    # find JLPT levels for the kanji and/or kana
    jlpt_level = [level_list[np.where(kanji_list == kanji)][0] if kanji in kanji_list else
                  level_list[np.where(kana_list == kana)][0] if kana in kana_list else 0
                  for kanji, kana in zip(freq_list['kanji'], freq_list['kana'])]
    # add levels to frequency list dataframe
    freq_list['jlpt_level'] = np.array(jlpt_level).astype(int)

    freq_list.to_csv(JLPT_WITH_FREQ_DF_FILENAME, index=False, encoding='utf8')
    return freq_list


def add_freq_to_jlpt(freq_list, jlpt):
    """
    Adds frequency percentage to JLPT dataframe
    """

    if os.path.isfile(JLPT_WITH_FREQ_DF_FILENAME):
        logging.info(
            'Skipping add_freq_to_jlpt(). File already exist: {}'.format(JLPT_WITH_FREQ_DF_FILENAME))
        return pd.read_csv(JLPT_WITH_FREQ_DF_FILENAME, encoding='utf8', keep_default_na=False)

    # get the references to the numpy arrays
    kanji_list = freq_list['kanji'].values
    kana_list = freq_list['kana'].values
    percentage_list = freq_list['percentage'].values

    # find percentage for the kanji and/or kana
    percentage = [percentage_list[np.where(kanji_list == kanji)][0] if kanji in kanji_list else
                  percentage_list[np.where(kana_list == kana)][0] if kana in kana_list else 0
                  for kanji, kana in zip(jlpt['kanji'], jlpt['kana'])]
    # add percentage to JLPT dataframe
    jlpt['percentage'] = np.array(percentage).astype(float)

    jlpt.to_csv(JLPT_WITH_FREQ_DF_FILENAME, index=False, encoding='utf8')
    return jlpt


if __name__ == "__main__":
    # load JLPT dataframe
    jlpt_df = load_jlpt_df()

    # load kanji levels dataframe
    kanji_levels_df = load_kanji_levels_df()
    # merge kanji levels dataframe
    jlpt_kanji_df = pd.merge(jlpt_df, kanji_levels_df, on='kanji', how='left').fillna(-1)

    # generate the kana character count vectors
    kana_char_count_vectors = generate_kana_char_count_vectors(jlpt_df['kana'].to_list())
    # merge JLPT and kana vectors
    jlpt_kanji_kana_df = pd.merge(jlpt_kanji_df, kana_char_count_vectors,
                                  left_index=True, right_index=True, suffixes=('', '_drop')).drop('kana_drop', axis=1)

    # load frequency lists into one dataframe
    freq_list_df = load_merged_freq_list()

    # add JLPT levels to frequency list dataframe
    jlpt_freq_df = add_freq_to_jlpt(freq_list_df, jlpt_kanji_kana_df)

    # get the word2vec model
    model = get_word2vec_model()
