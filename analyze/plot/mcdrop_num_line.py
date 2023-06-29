import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['hatch.linewidth'] = 1.5


matplotlib.rc("font", size=14)

plt.figure(figsize = (7,3))

thres5 = [
    71.8932422982547,
    74.712014475239,
    74.66301083,
    74.9662834,
    75.2498748,
    75.14141377

]


x1 = [
  1,
  3,
  5,
  10,
  15,
  30
    ]


# # libraries

# Data
#df=pd.DataFrame({'x_values': x1,'No Pred.': [55.44, 55.44, 55.44, 55.44, 55.44], 'All Pred.': [55.76, 55.76, 55.76, 55.76, 55.76], 'AS-ConvQA$_{uncer}$': f1})
df=pd.DataFrame({'x_values': x1, 'T-SAS': thres5, 'T-SAS w/o Stochastic':[74.0995440289053] * len(thres5)})

# multiple line plots
plt.plot( 'x_values', 'T-SAS', "bo--", linewidth=2, markersize=9, data=df)
plt.plot( 'x_values', 'T-SAS w/o Stochastic', 'ro-', linewidth=2, markersize=10, data=df)
#plt.plot( 'x_values', 'All Pred.', "orange", marker='', alpha=0.6, linewidth=3, markersize=10, data=df)

#plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")


plt.xticks()
plt.yticks()

plt.xlabel("Number of Dropout masks", fontsize=20)
plt.ylabel("F1", fontsize=20)
plt.legend(loc="lower right", fontsize=16, prop={'size': 9},)


plt.ylim([71, 76])
plt.gca().yaxis.set_ticks([71,72, 73, 74, 75, 76])
plt.gca().xaxis.set_ticks(x1)


plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/mcdrop_num_line.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
plt.savefig("/data/syjeong/prompt_test/analyze/plot/plot_outputs/mcdrop_num_line.png", dpi=300, bbox_inches="tight", pad_inches=0)
