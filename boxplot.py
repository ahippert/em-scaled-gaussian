import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style="ticks",font_scale=1.25)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


def plot_boxplot(showfig, savefig):
    """Box plots using seaborn
    """

    # Load distances
    y_temp = np.load("boxplot.npy")

    # Indexes corresponding to the estimators to plot
    ind = [2, 4, 0, 3, 5, 1, 6, 7]
    y = y_temp[:,ind]

    # Graphics
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(data=y, width=.3, palette="vlag")
    sns.stripplot(data=y, size=1, color=".5", linewidth=0, alpha=.7)
    plt.ylabel('Estimation error', size=14)
    plt.xticks(np.arange(8), ('Tyl-clair', 'Tyl-obs', 'EM-Tyl', 'SCM-clair', 'SCM-obs', 'EM-SCM', 'Mean-Tyl', 'RMI', ), rotation=30)
    plt.ylim(np.min(y)-.05, np.max(y)+.05)
    ax.yaxis.grid(True)
    sns.despine(trim=True, bottom=True)
    fig.tight_layout()
    if showfig:
        plt.show()
    if savefig:
        fig.savefig("boxplots.pdf", bbox_inches='tight')


