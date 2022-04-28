import matplotlib.pyplot as plt


def print_bin_size(jlpt):
    print("1: ", sum([1 if jlpt[word] == 1 else 0 for word in jlpt]))
    print("2: ", sum([1 if jlpt[word] == 2 else 0 for word in jlpt]))
    print("3: ", sum([1 if jlpt[word] == 3 else 0 for word in jlpt]))
    print("4: ", sum([1 if jlpt[word] == 4 else 0 for word in jlpt]))


def box_plot_jlpt_freq(jlpt_freq):
    # dunno if this works as is, just copied from jupytr notebook
    data = [jlpt_freq['1'], jlpt_freq['2'], jlpt_freq['3'], jlpt_freq['4']]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(data)
    plt.show()


def scatter_plot_occurances_vs_jlpt(df):
    df.plot.scatter(x='occurrences', y='jlpt', title="occurrences vs jlpt level");
