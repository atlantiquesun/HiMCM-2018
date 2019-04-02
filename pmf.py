import numpy as np
import pandas as pd

def pmf(df,n_days,period):
    '''
    df: pd.Series object
    period: period in days
    '''
    duration=[] 
    mi=[]
    q=n_days//period
    r=n_days%period
    df=df[r*96:]
    df=np.asarray(df).reshape((q,period*96))
    for i in range(96*period):
        dur_dic={} #frequency dictionary
        for j in range(q):
            if df[j,i-1]!=1 and df[j,i]==1:
                d=dur(df,(j,i),period)
                if d in dur_dic.keys():
                    dur_dic[d]+=1
                else:
                    dur_dic[d]=1
        if len(dur_dic)!=0 and sum(dur_dic.values())>2:
            m=max(dur_dic.values())/sum(dur_dic.values()) 
            spread=max(dur_dic.keys())-min(dur_dic.keys()) 
            duration.append((dur_dic,m,spread)) 
            mi.append(m)
    return (duration,mi)

def dur(df,pos,period):
    '''
    df: 2-D matrix, each row is a period of temperature data
    pos:(period number, time block numer), 0-indexed
    '''
    count=0
    for i in range(pos[1]+1,period*96):
        if df[pos[0],i]==1:
            count+=1
    return count