import numpy as np
import pandas as pd

def find_period(df,mini=1,maxi=31):
    '''
    input: pd.Series object containing historical temperature data
    output: the most likely period 
    '''
    r=[]
    for i in range(1,32):
        old=df.copy(deep=True)
        new=df.copy(deep=True)
        old=old.drop(old.index[list(range(df.shape[0]-96*i,df.shape[0]))])
        new=new.drop(new.index[list(range(0,96*i))])
        c=pd.concat((old,new),axis=1)
        r.append(pd.Series(df).autocorr(lag=i*96))
    period=7
    for i in range(31):
        if r[i]>0.7: 
            period=i+1
            break
    return period