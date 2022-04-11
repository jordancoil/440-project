import matplotlib.pyplot as plt
import numpy as np

def PrintBinSize(vocab_bins):
    print("1: ", len(vocab_bins['1'].keys()))
    print("2: ", len(vocab_bins['2'].keys()))
    print("3: ", len(vocab_bins['3'].keys()))
    print("4: ", len(vocab_bins['4'].keys()))


def BoxPlotJLPTFreq(jlpt_freq):
    # dunno if this works as is, just copied from jupytr notebook
    data = [jlpt_freq['1'],jlpt_freq['2'],jlpt_freq['3'],jlpt_freq['4']]

    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)
    plt.show()


def ScatterPlotOccurancesVsJLPT(df):
    df.plot.scatter(x='occurances', y='jlpt', title= "occurances vs jlpt level");