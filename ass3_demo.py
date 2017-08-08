# -*- coding: utf-8 -*-
"""
CIS 4320: Assignment 3
@author: 
"""

"""
The main objective of this assignment is to implement the RANSAC algorithm
for linear regression
"""

import numpy as np
from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
import json

def RANSAC(observed_data,threshold):
    """
    observed_data: the set of obverserved data for fitting a linear regression 
        model.
    threshold: the threshold is used to differentiate inliers from outliers by
        testing the error deviation. You can use a predefined percentage as the
        threshold as we show in the class.
    """
    max_loop=100
    maxdiff=0.1*threshold
    
    X,y=observed_data
    newX=X
    newy=y
    
    lmodel=linear_model.LinearRegression()
	
	#randomly generate an index array for outliers
    idxsold=np.random.randint(0,len(X),(int(len(X)*threshold),1))
    
    for i in range(max_loop):
    
        lmodel.fit(newX,newy)
        yhat=lmodel.predict(X)
         
	#element-wise vector operation
        ydiff=(yhat-y)**2
		
	#apply sorting and catch the index
        idxs=np.argsort(ydiff,axis=0)
		
	#identify indices of outliers
        idxsout=idxs[-int(len(idxs)*threshold):len(idxs)]
	#identify indices of inliers
        idxsin=idxs[0:len(idxs)-int(len(idxs)*threshold)]
        
	#compare with the previous loop, identify the difference 
	#of membership of outliers
        totaldiff=len(np.setdiff1d(idxsout,idxsold))
        
	#Alternative break
        if totaldiff<int(maxdiff*len(X)):
            print('early break on loop#{0}'.format(i))
            break
        
        newX=X[idxsin.flatten()]
        newy=y[idxsin.flatten()]
        idxsold=idxsout
       
    #Keep the following code unchanged
    return (newX,newy)


"""
Use the following code to test your RANSAC implementation. 
DON'T change this part.
"""

def main():
	infilename='ransac_data.txt'
	#infilename='ass3input.txt'
	with open(infilename,'r') as infile:
		indata=json.load(infile)

	X=np.array(indata['X']) 
	y=np.array(indata['y'])   

	#change threshold to 0.1 and 0.01, compare the difference
	threshold=0.3
	newX,newy=RANSAC((X,y),threshold)

	model = linear_model.LinearRegression()
	model.fit(newX, newy)
	yhat=model.predict(newX)

	m_coef=model.coef_
	m_intercept=model.intercept_

	outdata={'coef':m_coef.tolist(),'intercept':m_intercept.tolist()}
	with open('ass3output.txt','w') as outfile:
		json.dump(outdata,outfile)

	plt.scatter(X, y, color='yellowgreen', marker='.')
	plt.plot(newX, yhat, color='navy', linestyle='-', linewidth=2,
			 label='Linear regressor')

	plt.show()

if __name__ == "__main__": 
	main()	
