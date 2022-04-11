import re
import pandas as pd

def LoadVocabBins():
    # load JLPT vocab and sort into levels 1-4

    p = r'\n'

    vocab_bins = {
        '1': {},
        '2': {},
        '3': {},
        '4': {}
    }
    files = [
        ('/data/jlpt/jlpt-voc-1.utf', '1'), 
        ('/data/jlpt/jlpt-voc-2.utf', '2'), 
        ('/data/jlpt/jlpt-voc-3.utf', '3'), 
        ('/data/jlpt/jlpt-voc-4.utf', '4')]

    for file, bin in files:
    with open(file) as f:
        for l in f.readlines():
            if not l.startswith("#") and not l.startswith("\n"):
                l = re.sub(p,'',l)
                w = l.split(" ")

                if '~' in w[0]:
                    # dont add these
                    continue

                if w[0] in vocab_bins[bin].keys():
                    # print("Duplicate found in bin ", bin, " : ", w[0])
                    pass
                else:
                    # can't really remember what this does...
                    if len(w) > 1 and not w[1].startswith('（'):
                        vocab_bins[bin][w[0]] = w[1]
                    else:
                        vocab_bins[bin][w[0]] = 1

    # remove duplicate keys from higher bins
    for k4 in vocab_bins['4'].keys():
        vocab_bins['3'].pop(k4, None)
        vocab_bins['2'].pop(k4, None)
        vocab_bins['1'].pop(k4, None)

    for k3 in vocab_bins['3'].keys():
        vocab_bins['2'].pop(k4, None)
        vocab_bins['1'].pop(k4, None)

    for k2 in vocab_bins['2'].keys():
        vocab_bins['1'].pop(k2, None)

    return vocab_bins


def LoadFreqList(file_name):
    '''
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
    col_names = ['occurances', 'kanji', 'hiragana', 'kana', 'part_of_speech', 
                 'anotherpos', 'freq_group', 'freq_rank', 'percentage', 'cum_percentage']
    '''
    p1 = r'\ufeff'
    p2 = r'\n'

    with open(file_name) as f:
        num_feats = 3
        keys = []
        data = []

        lines = f.readlines()
        for i, l in enumerate(lines):
            lc = l
            lc = re.sub(p1,'',lc)
            lc = re.sub(p2,'',lc)
            r = lc.split("\t")

            try:
                # don't include particles
                if 'prt' in r[6]:
                    continue
                else:
                    keys.append(r[1])
                    data.append([r[0], r[3], r[4]])
            except:
                print("Error converting line {} with array: {} to float.".format(l, [r[0], r[3], r[4]]))
    
    npdata = np.array(data).astype(float)
    df = pd.DataFrame(data=npdata, index=keys)
    df.columns = ['occurances', 'freq_rank', 'percentage']

    return df

# old version
# def LoadFreqList(file_name):
#     freq_list = {}
#     p1 = r'\ufeff'
#     p2 = r'\n'
#     with open(file_name) as f:
#         for l in f.readlines():
#             l = re.sub(p1,'',l)
#             l = re.sub(p2,'',l)
#             r = l.split("\t")
#             if 'prt' in r[6]:
#                 continue
#             if r[1] not in freq_list.keys():
#                 freq_list[r[1]] = r[2]
#     return freq_list


def AddJLPTtoFreqList(freq_list_df):
    '''
    Adds a JLPT column to the frequency list pandas dataframe
    1-4 if has corresponding JLPT level
    0 if it is not found
    '''
    num_rows = freq_list_df.shape[0]
    jlpt = []

    found = 0
    not_found = 0

    for index, row in freq_list_df.iterrows():
        success = False
        for k in ['1','2','3','4']:
            try:
                if vocab_bins[k][index]:
                    success = True
                    found += 1
                    jlpt.append(k)
                    break
            except:
                continue
  
        if not success:
            not_found += 1
            jlpt.append('0')

        success = False

    print("found {}, not_found {}, num_rows {}".format(found, not_found, num_rows))
    assert (found + not_found) == num_rows, "found + not_found != num rows"
    assert len(jlpt) == num_rows, "len(jlpt) != num_rows"
    print("Assertions passed")

    jlptdf = pd.DataFrame(data=np.array(jlpt).astype(int))
    freq_list_df['jlpt'] = np.array(jlpt).astype(int)

    return freq_list_df


def SplitDFByJLPT(df):
    grouped = df.groupby(df.jlpt)
    df_0 = grouped.get_group(0)
    df_jlpt = pd.concat( [ grouped.get_group(group) for group in grouped.groups if not group == 0 ])

    return df_0, df_jlpt


def GetJLPTFreq(vocab_bins, freq_list):
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

