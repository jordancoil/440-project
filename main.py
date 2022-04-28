import numpy as np
import pandas as pd
from word2vec.generate_vectors import get_word2vec_model


# load JLPT vocab and sort into levels 1-4
def get_jlpt_vocab_matrix():
    file_names_and_bin_numbers = [
        ('data/jlpt/jlpt-voc-4-extra.utf', '4'),
        ('data/jlpt/jlpt-voc-3-extra.utf', '3'),
        ('data/jlpt/jlpt-voc-2-extra.utf', '2'),
        ('data/jlpt/jlpt-voc-1-extra.utf', '1')]

    #               jlpt,   kanji,  kana
    vocab_levels = [[], [], []]
    kanji_list = set()

    for file_name, jlpt in file_names_and_bin_numbers:
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
                vocab_levels[0].append(int(jlpt))

                # add both kanji and kana if it has it
                if len(word_list) > 1 and word_list[1].startswith('[') and '（' not in word_list[1]:
                    vocab_levels[1].append(word_list[0])
                    vocab_levels[2].append(word_list[1][1:-1])
                    # add to list of used kanji
                    kanji_list.add(word_list[0])
                # add just kana as it has no kanji
                else:
                    vocab_levels[1].append('')
                    vocab_levels[2].append(word_list[0])

    return vocab_levels


def generate_kana_char_count_matrix(kana):
    kana_vectors = [kana]
    char_to_column_index = dict()

    for word in kana:
        # set the count of each char to 0 for current word
        for index in char_to_column_index.values():
            kana_vectors[index].append(0)

        for char in word:
            # char already has a column
            if char in char_to_column_index.keys():
                kana_vectors[char_to_column_index[char]][-1] += 1
            # there is no char column, make one
            else:
                char_to_column_index[char] = len(kana_vectors)
                kana_vectors.append([0] * len(kana_vectors[0]))
                kana_vectors[char_to_column_index[char]][-1] = 1

    return kana_vectors


if __name__ == "__main__":
    vocab_matrix = get_jlpt_vocab_matrix()
    kana_char_count_matrix = generate_kana_char_count_matrix(vocab_matrix[2])
    model = get_word2vec_model()
    print(model.wv.most_similar(vocab_matrix[1][1]))


# load JLPT vocab and sort into levels 1-4
def load_jlpt():
    jlpt = dict()

    file_names_and_bin_numbers = [
        ('data/jlpt/jlpt-voc-1.utf', '1'),
        ('data/jlpt/jlpt-voc-2.utf', '2'),
        ('data/jlpt/jlpt-voc-3.utf', '3'),
        ('data/jlpt/jlpt-voc-4.utf', '4')]

    for file_name, bin_number in file_names_and_bin_numbers:
        with open(file_name, 'r', encoding='utf8') as f:
            for line in f.readlines():
                if line.startswith("#") or line.startswith("\n"):
                    continue

                word_list = line[:-1].split(" ")

                for word in word_list:
                    # checks for invalid words or if the word is already in a lower difficulty bin
                    if '~' in word or word.startswith('（'):
                        continue
                    jlpt[word] = bin_number

    return jlpt


def load_freq_list(file_name):
    """
    eg. file_name = '/data/freq_lists/word_freq_report.txt'
    https://github.com/chriskempson/japanese-subtitles-word-frequency-list

    Field 1: Number of times word was encountered         < yes
    Field 2: Word
    Field 3: Frequency Group
    Field 4: Frequency Rank                               < yes
    Field 5: Percentage (Field 1 / Total number of words) < yes
    Field 6: Cumulative percentage
    Field 7: Part-of-speech

    Alternatives:
    file_name = '/data/freq_lists/netflix_unidic_3011_no_names_word_freq_report.txt'
    col_names = ['occurrences', 'kanji', 'hiragana', 'kana', 'part_of_speech',
                 'anotherpos', 'freq_group', 'freq_rank', 'percentage', 'cum_percentage']
    """

    with open(file_name, 'r', encoding='utf8') as f:
        keys = []
        data = []

        for line in f.readlines():
            row = line[:-1].split("\t")

            # not enough columns
            if len(row) < 5:
                continue

            # exclude particles
            if 'prt' in row[6]:
                continue

            keys.append(row[1])
            data.append([row[0], row[3], row[4]])

    return pd.DataFrame(data=np.array(data, dtype=float), index=keys,
                        columns=['occurrences', 'freq_rank', 'percentage'])


def add_jlpt_to_freq_list(freq_list, jlpt):
    """
    Adds a JLPT level column to the frequency list dataframe
    1-4 if it has a corresponding JLPT level
    0 if it does not
    """

    jlpt_level = [jlpt[word] if word in jlpt else 0 for word in freq_list.index.values]
    freq_list['jlpt_level'] = np.array(jlpt_level).astype(int)

    return freq_list


def split_dataframe_by_jlpt(df):
    grouped = df.groupby(df.jlpt)
    df_0 = grouped.get_group(0)
    df_jlpt = pd.concat([grouped.get_group(group) for group in grouped.groups if not group == 0])

    return df_0, df_jlpt


def get_jlpt_freq(vocab_bins, freq_list):
    jlpt_freq = {
        '4': [],
        '3': [],
        '2': [],
        '1': []
    }

    for k in jlpt_freq.keys():
        for vk in vocab_bins[k].keys():
            s = freq_list.get(vk)
            if s:
                jlpt_freq[k].append(int(s))

    return jlpt_freq
