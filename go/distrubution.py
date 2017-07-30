from matplotlib import pyplot as plt
import seaborn as sns
import data_helper
import numpy as np

"""
    Show the data distribution of pokemen GO
"""
df = data_helper.load()
df = data_helper.mergeMultipleTypes(df)
df = data_helper.reverseMapping(df)
sns.lmplot(x="cp", y='hp', data=df, hue='value', fit_reg=False)
plt.show()