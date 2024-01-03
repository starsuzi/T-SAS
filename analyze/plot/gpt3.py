import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import matplotlib.font_manager

#matplotlib.font_manager._rebuild()

# font = {'family' : 'Times New Roman',
#        'size'   : 14}
# font = {'family' : 'Times New Roman',
#        'size'   : 14}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


matplotlib.rc("font", size=14)

plt.figure(figsize = (4.5,2.5))

ax = plt.subplot(1, 1, 1)

width = 0.65

def create_x(t, w, n, d):
    return np.array([t*x + w*n for x in range(d)])

######################################################

value_naive = [66.39, 81.17, 83.00]
value_gpt3 = [70.57, 81.57, 75.16]
value_ours = [75.29, 84.01, 84.72]


value_a_x = create_x(3, width, 1, 3)
value_b_x = create_x(3, width, 2, 3) + 0.040
value_c_x = create_x(3, width, 3, 3) + 0.080


rects1 = ax.bar(value_a_x, value_naive, width, color='red', edgecolor='black', zorder=1.0, alpha=0.5)
rects2 = ax.bar(value_b_x, value_gpt3, width, color='orange', edgecolor='black', zorder=1.0, alpha=0.5)
rects3 = ax.bar(value_c_x, value_ours, width, color='green', edgecolor='black', zorder=1.0, alpha=0.5)


middle_x = [(a+j)/2 for (a,j) in zip(value_a_x, value_c_x)]



ax.set_xticks(middle_x)
ax.set_xticklabels([ 'NQ', 'TQA', 'SQD'], fontsize=16)

ax.margins(x=0.1, y=None)

plt.xticks() #fontsize=20
plt.yticks() #fontsize=20

plt.ylabel("F1", fontsize=16)

ax.set(ylim=[60, 85])
ax.yaxis.set_ticks([60, 65, 70, 75, 80, 85])
ax.legend(
    (rects1[0], rects2[0], rects3[0]), #
    ('Naïve LM','GPT-3', 'T-SAS (Ours)'), 
    prop={'size': 11}, loc='lower right', handletextpad=0.1, 
    labelspacing=0.01, framealpha=0.5, borderpad=0.01, ncol=1
    #bbox_to_anchor=(-0.01, 1.02)
)
#plt.suptitle("Effectiveness of Confidence & Uncertainty", fontsize=18)

######################################################

plt.savefig("/data/soyeong/prompt_test/analyze/plot/plot_outputs/gpt3.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
plt.savefig("/data/soyeong/prompt_test/analyze/plot/plot_outputs/gpt3.png", dpi=300, bbox_inches="tight", pad_inches=0)
