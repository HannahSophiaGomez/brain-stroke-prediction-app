
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def gender_stroke_plot(data):
    f,ax=plt.subplots(1,2,figsize=(10,7))
    data[['gender','stroke']].groupby(['gender']).mean().plot.bar(ax=ax[0])
    ax[0].set_title('Had a stroke vs Gender')
    sns.countplot(x='gender',hue='stroke',data=data,ax=ax[1])
    ax[1].set_title('Gender:Had a Stroke vs Not had a Stroke')
    return f