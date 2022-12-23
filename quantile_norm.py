import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




def quantile_normalize(df):
    """
    input: dataframe with numerical columns
    output: dataframe with quantile_minmax normalized values
    """
    df_sorted = pd.DataFrame(np.sort(df.values,
                                     axis=0),
                             index=df.index,
                             columns=df.columns)
    print("sorted:",df_sorted)
    df_mean = df_sorted.mean(axis=1)
    print("mean:",df_mean)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn =df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return(df_qn)

dir1 = 'quantile'
norm="new"

qn_feature=pd.read_csv('../data/data_mapping/14F/newquantile.csv')
sns.boxplot(data=qn_feature)
sns.stripplot(data = qn_feature,size=1)

# set x-axis label
plt.xlabel("Features", size=18)
# set y-axis label
plt.ylabel("Measurement", size=18)
plt.title("Boxplot of raw data after Quantile Normalization")
plt.savefig('../data/data_mapping/14F/'+dir1+'/'+norm+'/Boxplot_after_Quantile_Normalization.png',dpi=150)



