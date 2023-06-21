import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# font = {'family' : 'Times New Roman',
#        'size'   : 14}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

matplotlib.rc("font", size=12)
fig = plt.figure(figsize=(4, 2.6))

width = 0.05

def create_x(t, w, n, d):
    return np.array([t*x + w*n for x in range(d)])

######################################################
value_a = [62.139686297721]    # baseline # 22.727817557058
value_b = [77.617638354]    # Ours # 1.5720420128647

value_a_x = create_x(5, width, 1, 1)
value_b_x = create_x(5, width, 2, 1) + 0.005

# tab:red, orange, green, blue
ax = plt.subplot(1, 2, 1)
rects1 = ax.bar(value_a_x, value_a, width, color='red', edgecolor='black', zorder=1.0, alpha=0.5)
rects2 = ax.bar(value_b_x, value_b, width, color='orange', edgecolor='black', zorder=1.0, alpha=0.5)

middle_x = [(a+d)/2 for (a,d) in zip(value_a_x, value_b_x)]

ax.set_xticks(middle_x)
ax.set_xticklabels(['Labeled'], fontsize=14)

ax.margins(x=0.1, y=None)

ax.set(ylim=[43, 48])
ax.yaxis.set_ticks([43, 44, 45, 46, 47, 48])
ax.legend(
    (rects1[0], rects2[0]), 
    ('Naïve LM', 'T-SAS (Ours)'), 
    prop={'size': 11}, loc='upper left', handletextpad=0.1, 
    labelspacing=0.01, framealpha=0.5, borderpad=0.01,
    bbox_to_anchor=(-0.03, 1.02)
)
plt.suptitle("MRR on Natural Question", fontsize=18)

######################################################
value_a = [35.24]    # DPR (only_inbatch_32)
value_b = [35.68]    # QA (gen_only_inbatch_32)

# tab:red, orange, green, blue
ax = plt.subplot(1, 2, 2)
rects1 = ax.bar(value_a_x, value_a, width, color='red', edgecolor='black', zorder=1.0, alpha=0.5)
rects2 = ax.bar(value_b_x, value_b, width, color='orange', edgecolor='black', zorder=1.0, alpha=0.5)

middle_x = [(a+d)/2 for (a,d) in zip(value_a_x, value_b_x)]

ax.set_xticks(middle_x)
ax.set_xticklabels(['Unlabeled'], fontsize=14)

ax.margins(x=0.1, y=None)

ax.set(ylim=[34, 39])
ax.yaxis.set_ticks([34, 35, 36, 37, 38, 39])
# ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('DPR', 'DPR w/ QA', 'DPR w/ DA', 'DPR w/ AR', 'DAR (Ours)'), prop={'size': 10}, loc='upper left', handletextpad=0.1, labelspacing=0.05, framealpha=0.5, borderpad=0.1)

plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/prompt_robustness.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/prompt_robustness.png", dpi=300, bbox_inches="tight", pad_inches=0)
