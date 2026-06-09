"""Paper figure style — single source of truth for fonts/sizes.

Call set_paper_style() at the top of any notebook or script before plotting.
The numbers are tuned for the KCC two-column layout where figures get
column-width scaled, so axis labels need to be visibly larger than the
matplotlib defaults.
"""
import matplotlib.pyplot as plt


def set_paper_style():
    plt.rcParams.update({
        'font.size':         14,
        'axes.titlesize':    16,
        'axes.labelsize':    15,
        'xtick.labelsize':   13,
        'ytick.labelsize':   13,
        'legend.fontsize':   12,
        'figure.titlesize':  17,
        'figure.titleweight':'bold',
        'axes.titleweight':  'bold',
        'axes.labelweight':  'bold',
    })
