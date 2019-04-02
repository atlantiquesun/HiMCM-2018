import numpy as np
import pandas as pd

def recent(r2,r1):
    return (r2-r1).sum()/r1.shape[0]
    
def weight(ts,history=4):
    val=ts[history]
    ts=ts[:history]
    diff=abs(val-ts)
    m=max(diff)
    w=(m+0.001)-diff
    for i in range(history-1):
        w[i]=w[i+1] 
    s=sum(w)
    w=w/s
    loss=(min(diff))/(sum(ts))
    if val<min(ts):
        w=w-loss
    if val>max(ts):
        w=w+loss
    return w

def batch_predict(ts,position,period=7,reverse=False,history=4):   
    if position!=period:
        y=ts[-96*(period-position+1):-96*(period-position)] 
    else:
        y=ts[-96:]
    print(ts)
    x=np.asarray(ts[-96*(history+2)*period:]).reshape((history+2,period*96)) 
    if position!=period:
        x=x[:-1,-96*(period-position+1):-96*(period-position)] 
    else:       
        x=x[:-1,-96:] #get the last day's historical data
    w=np.zeros((history,1))
    for i in range(96):
        w=np.concatenate((w,weight(x[:,i],history=history).reshape(history,1)),axis=1)
    w=w[:,1:]
    if position!=period:
        s1=-96*(period-position+1)-96*period
        e1=-96*(period-position+1)
        s2=-96*(period-position+1)-2*96*period
        e2=-96*(period-position+1)-96*period
        result=np.multiply(w,x[1:,:]).sum(axis=0)+recent(ts[s1:e1],ts[s2:e2])
    else:
        result=np.multiply(w,x[1:,:]).sum(axis=0)+recent(ts[-96*2:-96],ts[-96:])
    r=list(result)
    return (result,y,r)

def batch(ts,period=7,history=4,n=3):
    mean=[] #mean actual temperature
    mean_pred=[] #mean predicted temperature
    mean_abs_error=[] #mean absolute error
    max_abs_error=[] #max absolute error
    count=0 
    for j in range(n):  
        for i in range(1,period+1):
            (result,y,_)=batch_predict(ts,i,period=period,history=history)
            m=result.sum()/96
            ae=np.absolute(result-np.asarray(y)).sum()/96
            me=max(list(np.absolute(result-np.asarray(y))))
            if ae>1.5: count+=1
            mean.append(y.sum()/96)
            mean_pred.append(m)
            mean_abs_error.append(ae)
            max_abs_error.append(me)
        ts=ts[:-96*period]    
    return(mean,mean_pred,mean_abs_error,max_abs_error,count)
