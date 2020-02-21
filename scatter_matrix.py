from bokeh.sampledata.iris import flowers as df
import pandas as pd
import matplotlib.pyplot as plt

pd.plotting.scatter_matrix(df, figsize=(30,20), s=300)
plt.savefig('scatter_matrix.png')