# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:18:34 2020

@author: Magnus Bengtsson
"""
import numpy as np
#ANN proof of concept
#Definitions
#function ANN_v2()
input=([0.05, 0.1, 0.3, 0.1,])
b1=0.35
b2=0.6
input_w=np.asarray([[.15, .25, 0.1],[.2, .3, 0.1],[0.1,0.1, 0.1],[0.1, 0.1, 0.1]]) # %första kolumnen är vikter till första gömda neuronen osv..
hidden_w1=([.40, .5,0.1],[.45, .55, 0.1],[.15, .2, 0.1]) #;% kolumn avgör nästa lagers antal noder
hidden_w=np.asarray(hidden_w1)
number_input=3
number_hidden=2 # %3

def forward_pass(weight1,input,b1):
     net1=np.dot(input,weight1)+b1*1 # %sumering varje rad
     out1=np.divide(1,1+np.exp(np.multiply(-1,net1))) # 1./(1+np.exp(-net1))   #### elementvis division
     return net1, out1

def backward_init(target,out,hout,hidden_w):
    dEtot_dout_o=-(target-out)# Derivative of Etot WRT out corr.            dC_da
    dout_o_dnet=np.multiply(out,(np.subtract(1,out)))  #out.*(1-out) corr.  da_dz
    dnet_dw=hout#                                                           dz_dw
    diff =[[0 for x in range(len(hout))] for y in range(len(out))]#alt np.asarray() Gener ate empty mat
    for i in range(0,len(out)):
        for j in range(0,len(hout)):
            diff[j][i]=dEtot_dout_o[i]*dout_o_dnet[i]*dnet_dw[j]  #alt diff[i,j]
    hidden_w=hidden_w-0.5*np.asarray(diff) #uppdaterar vikten för noden
    return hidden_w

def backward_hidden(target,out,hout,input_w,hidden_w,input):
    dout_o_dnet=np.multiply(out,(np.subtract(1,out))) #out.*(1-out)    da_dz
    dEtot_dout_o=-(target-out)
    
    dhout_o_dnet=np.multiply(hout,(np.subtract(1,hout))) #hout.*(1-hout) da_dz
    dEtot_dnet=np.multiply(dEtot_dout_o,dout_o_dnet)
    dnet_dhout=hidden_w
    dnet_dwi=input
    ii=0
    diff=np.zeros(len(dhout_o_dnet)*len(dnet_dwi))
    for j in range(0,len(dhout_o_dnet)):
        for k in range(0,len(dnet_dwi)):
            diff[ii]=np.sum(np.multiply(dEtot_dnet,dnet_dhout[j][:])*dhout_o_dnet[j]*dnet_dwi[k])
            ii=ii+1
    ii=0
    input_w2=[[0 for x in range(len(input_w[0,:]))] for y in range( len(input_w[:,0]))]
    for j in range(0,len(input_w[0,:])): #%djupled i viktmatris
        for k in range(0,len(input_w[:,0])): #%sidled viktmatris
            input_w2[k][j]=input_w[k][j]-0.5*diff[ii] # %gradient descent
            ii=ii+1
    return np.asarray(input_w2)

##################main loop#########################################
for i in range(0,10000):
    [net1a,hout]=forward_pass(input_w,input,b1)
    [net1b,out]=forward_pass(hidden_w,hout,b2)
########%=====
    target=([0.01, 0.99, 0.1])
    Etot=np.sum(0.5*np.power((target-out),2))
###Etot=sum(E);
##%Backward pass
    hidden_wo=hidden_w
    hidden_w=backward_init(target,out,hout,hidden_w)
##%Hidden layer
    input_w=backward_hidden(target,out,hout,input_w,hidden_wo,input)

    print(Etot)
    ############################################################