import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['hatch.linewidth'] = 1.5


matplotlib.rc("font", size=14)

plt.figure(figsize = (7,3))

mc = [
  68.25116813,
  69.20580789,
  68.72515846,
  70.47700794,
  70.74022508,
  73.07392904,
  73.14189404,
  74.20378838,
  74.44797678,
  #74.2810668682079

]

topk = [
  68.08517365,
  67.84082948,
  68.26380305,
  68.3136087,
  69.25000498,
  69.98755189,
  70.60909802,
  71.67706794,
  72.52982006,
  #73.4630285238118
]

x1 = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]#, 0.9]


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
plt.ylabel("F1", fontsize=20)
plt.legend(loc="lower right", fontsize=16, prop={'size': 9},)


plt.ylim([67, 75])
plt.gca().yaxis.set_ticks([67, 69, 71, 73, 75])
plt.gca().xaxis.set_ticks(x1)


plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/topk_comparison_f1.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/topk_comparison_f1.png", dpi=300, bbox_inches="tight", pad_inches=0)
