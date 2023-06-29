import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# font = {'family' : 'Times New Roman',
#        'size'   : 14}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

matplotlib.rc("font", size=14)
fig = plt.figure(figsize=(3, 2.5))

width = 0.05

def create_x(t, w, n, d):
    return np.array([t*x + w*n for x in range(d)])


######################################################
value_a = [70.6455236]    # Naïve LM
value_b = [79.40957256]    # Self-Adaptive w/ Greedy
value_c = [76.98611707]    # Self-Adaptive w/ Soft
value_d = [76.22771637]    # Self-Adaptive w/ LMSI
value_e = [82.84365951]    # T-SAS (Ours)

# value_a_err = [19.26619718]    # Naïve LM
# value_b_err = [6.593382348]    # Self-Adaptive w/ Greedy
# value_c_err = [10.52756287]    # Self-Adaptive w/ Soft
# value_d_err = [14.09877162]    # Self-Adaptive w/ LMSI
# value_e_err = [2.017651645]    # T-SAS (Ours)


value_a_err = ['$σ= $19']    # Naïve LM
value_b_err = ['$σ= $7']    # Self-Adaptive w/ Greedy
value_c_err = ['$σ= $11']    # Self-Adaptive w/ Soft
value_d_err = ['$σ= $14']    # Self-Adaptive w/ LMSI
value_e_err = ['$σ= $2']    # T-SAS (Ours)

value_a_x = create_x(7, width, 1, 1)
value_b_x = create_x(7, width, 2, 1) + 0.025
value_c_x = create_x(7, width, 3, 1) + 0.050
value_d_x = create_x(7, width, 4, 1) + 0.075
value_e_x = create_x(7, width, 5, 1) + 0.100

# # tab:red, orange, green, blue
# ax = plt.subplot(1, 1, 1)
# rects1 = ax.bar(value_a_x, value_a, width, yerr=value_a_err, color='red', edgecolor='black', zorder=1.0, alpha=0.5)
# rects2 = ax.bar(value_b_x, value_b, width, yerr=value_b_err, color='orange', edgecolor='black', zorder=1.0, alpha=0.5)
# rects3 = ax.bar(value_c_x, value_c, width, yerr=value_c_err, color='green', edgecolor='black', zorder=1.0, alpha=0.5)
# rects4 = ax.bar(value_d_x, value_d, width, yerr=value_d_err, color='blue', edgecolor='black', zorder=1.0, alpha=0.5)
# rects5 = ax.bar(value_e_x, value_e, width, yerr=value_e_err, color='purple', edgecolor='black', zorder=1.0, alpha=0.5)

# tab:red, orange, green, blue
ax = plt.subplot(1, 1, 1)
rects1 = ax.bar(value_a_x, value_a, width, color='red', edgecolor='black', zorder=1.0, alpha=0.5)
rects2 = ax.bar(value_b_x, value_b, width, color='orange', edgecolor='black', zorder=1.0, alpha=0.5)
rects3 = ax.bar(value_c_x, value_c, width, color='green', edgecolor='black', zorder=1.0, alpha=0.5)
rects4 = ax.bar(value_d_x, value_d, width, color='blue', edgecolor='black', zorder=1.0, alpha=0.5)
rects5 = ax.bar(value_e_x, value_e, width, color='purple', edgecolor='black', zorder=1.0, alpha=0.5)

# Label with given captions, custom padding and annotate options
# ax.bar_label(rects1, labels = value_a_err, fmt='{:,.0f}', rotation = 10)
# ax.bar_label(rects2, labels = value_b_err, fmt='{:,.0f}', rotation = 10)
# ax.bar_label(rects3, labels = value_c_err, fmt='{:,.0f}', rotation = 10)
# ax.bar_label(rects4, labels = value_d_err, fmt='{:,.0f}', rotation = 10)
# ax.bar_label(rects5, labels = value_e_err, fmt='{:,.0f}', rotation = 10)

ax.bar_label(rects1, labels = value_a_err, fmt='{:,.0f}', fontsize = 12, rotation = 10)
ax.bar_label(rects2, labels = value_b_err, fmt='{:,.0f}', fontsize = 12, rotation = 10)
ax.bar_label(rects3, labels = value_c_err, fmt='{:,.0f}', fontsize = 12, rotation = 10)
ax.bar_label(rects4, labels = value_d_err, fmt='{:,.0f}', fontsize = 12, rotation = 10)
ax.bar_label(rects5, labels = value_e_err, fmt='{:,.0f}', fontsize = 12, rotation = 10)



middle_x = [(a+i)/2 for (a,i) in zip(value_a_x, value_e_x)]

ax.set_xticks(middle_x)
ax.set_xticklabels(['Self-Adaptive LMs'], fontsize=18)

ax.margins(x=0.1, y=None)

plt.ylabel("Average F1", fontsize=18)

ax.set(ylim=[65, 85])
ax.yaxis.set_ticks([65, 70, 75, 80, 85])
ax.legend(
    (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), 
    ('Naïve LM', 'w/ Greedy', 'w/ Soft', 'w/ LMSI', 'T-SAS (Ours)'), 
    prop={'size': 9}, loc='lower right', handletextpad=0.1, 
    labelspacing=0.01, framealpha=0.5, borderpad=0.01, ncol=1,fontsize=16
    #bbox_to_anchor=(-0.01, 1.02), loc='upper right'
)

plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/prompt_robustness.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/prompt_robustness.png", dpi=300, bbox_inches="tight", pad_inches=0)
