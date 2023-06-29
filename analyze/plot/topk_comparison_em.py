import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['hatch.linewidth'] = 1.5


matplotlib.rc("font", size=12)

plt.figure(figsize = (8,3))

mc = [
  57.5625,
  59.015625,
  58.234375,
  60.046875,
  60.046875,
  62.828125,
  62.78125,
  63.484375,
  63.265625,
  #63.3593749999999

]

topk = [
  56.796875,
  56.671875,
  57.078125,
  57.0625,
  57.828125,
  58.90625,
  59.28125,
  60.75,
  61.6875,
  #62.59375
]

x1 = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] #, 0.9


# # libraries

# Data
#df=pd.DataFrame({'x_values': x1,'No Pred.': [55.44, 55.44, 55.44, 55.44, 55.44], 'All Pred.': [55.76, 55.76, 55.76, 55.76, 55.76], 'AS-ConvQA$_{uncer}$': f1})
df=pd.DataFrame({'x_values': x1, 'T-SAS w/ MC.': mc, 'T-SAS w/ Top-$K$':topk})

# multiple line plots
plt.plot( 'x_values', 'T-SAS w/ MC.', "bo--",  linewidth=2, markersize=9, data=df)
plt.plot( 'x_values', 'T-SAS w/ Top-$K$', "ro--", linewidth=2, markersize=9, data=df)
#plt.plot( 'x_values', 'All Pred.', "orange", marker='', alpha=0.6, linewidth=3, markersize=10, data=df)

#plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")


plt.xticks()
plt.yticks()

plt.xlabel("Filter Threshold", fontsize=20)
plt.ylabel("EM", fontsize=20)
plt.legend(loc="lower right", fontsize=16, prop={'size': 9},)


plt.ylim([55, 65])
plt.gca().yaxis.set_ticks([55, 57, 59, 61, 63, 65])
plt.gca().xaxis.set_ticks(x1)


plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/topk_comparison_em.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/topk_comparison_em.png", dpi=300, bbox_inches="tight", pad_inches=0)
