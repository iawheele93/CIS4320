#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:07:59 2017

@author: 
"""
import numpy as np
from matplotlib import pyplot as plt
import json

def kmeanerr(X,ylabel,knum):
    errarray=np.zeros(knum)
    kcenters=np.zeros((knum,X.shape[1]))
    
    for i in range(knum):
        agroup=X[ylabel==i,:]
        acenter=np.mean(agroup,axis=0)
        rawdist=agroup-acenter
        errarray[i]=np.sum(np.sum(rawdist**2,axis=1))
        kcenters[i,:]=acenter
                
    return(errarray,kcenters)    
        

def kmeanfun(X,knum,errlimit,looplimit):
    """
        each row is a data point
    """

    distarray=np.zeros((X.shape[0],knum))
    kcenters=np.random.uniform(X.min(),X.max(),(knum,X.shape[1]))
    oldlabel=np.random.randint(0,knum,X.shape[0])
    
    loopcnt=0
    while True:
        for i in range(knum):
            rawdist=X-kcenters[i,:]
            distarray[:,i]=np.sum(rawdist**2,axis=1)
        newlabel=np.argsort(distarray,axis=1)[:,0]
    
        oldinfo=kmeanerr(X,oldlabel,knum)
        newinfo=kmeanerr(X,newlabel,knum)
    
        errdiff=np.abs(np.sum(oldinfo[0])-np.sum(newinfo[0]))
        print('Loopnum:{0}, err={1}'.format(loopcnt,errdiff))
        if errdiff<errlimit:
            break
        elif loopcnt>looplimit:
            break
        else:
            oldlabel=newlabel
            kcenters=newinfo[1]
            loopcnt+=1
    
    return(newlabel,newinfo[1])

def main():
    infilename='kmean_data.txt'
    with open(infilename,'r') as infile:
        indata=json.load(infile)
    
    X=np.array(indata['X'])
    ylabel=np.array(indata['ylabel'])

    (klabel,kcenters)=kmeanfun(X,3,100,100)
    plt.scatter(X[:, 0], X[:, 1], c=klabel)
    plt.title("K-mean result")
    plt.show()

if __name__=="__main__":
    main()

