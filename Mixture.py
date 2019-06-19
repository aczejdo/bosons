#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:10:03 2019

@author: aleksczejdo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import matplotlib.cm as cm
from scipy.special import eval_hermite as eh

#------------------get our wavefunctions----------------------------------------
def getPsi(L,K):
    Psi=np.zeros((K,L))
    global ix,x,dx
    for i in range(0,K):
        A=0
        Psi[i,ix]=eh(i,x[ix])*np.exp(-0.5*x[ix]**2)
        A=integ(Psi[i,:]**2)
        Psi[i,:]=Psi[i,:]/np.sqrt(A)
    return(Psi)

#---------------------simple trapezoidal integral-------------------------------
def integ(X):
    return(np.sum(X[:])*dx - (X[0]+X[-1])*dx/2)

#--------------------get all the wave function overlaps-------------------------
def getU(K):
    U=np.zeros((K,K,K,K))
    for i in range(0,K):
            for j in range(0,K):
                for k in range(0,K):
                    for l in range(0,K):
                        if ((i+j+k+l)%2==0):
                            U[i,j,k,l]=integ(Psi[i]*Psi[j]*Psi[k]*Psi[l])
                        else:
                            continue
    return(U)

#----------------------Generate binomial table for FAST lookup------------------
def getBTable(K,N):
    BinomialTable=np.zeros((K,N+1),dtype=int)
    for i in range(0,K):
        for j in range(0,N+1):
            #BinomialTable[i,j]=int(comb(K-1-i + N - (1+ j),K-1-i))
            BinomialTable[i,j]=int(comb(K-1-i + N - (1+ j),K-1-i))
    return(BinomialTable)

#-------------Get indices of operated state-------------------------------------
#very sexy algorithm for finding the exact index of a state
def searchBasis(state,whichbasis):
    #global BinomialTable
    ind=0
    for i in range(0,Karr[whichbasis]):
        #ind+=int(comb(K-1-i + N - (1+ np.sum(state[0:i+1])),K-1-i))
        ind+=BinomialTable[whichbasis][i,int(np.sum(state[0:i+1]))]
    return(ind)

#--------------------Energy for each N -----------------------------------------
def getEarr(K,other):
    Earr=np.arange(0,K)+0.5
    return(Earr)
#--------------------Build the Hamiltonian--------------------------------------
def createH(g):
    #global U
    #print(g)
    #Earr=np.arange(0,K)+0.5
    #global bi
    diag=np.zeros(Nshape,dtype='object')
    Earr=np.zeros(Nshape,dtype='object')
    for i in range(0,Nshape[0]):
        diag[i]=np.arange(0,Fsizearr[i],dtype=int)
        Earr[i]=getEarr(Karr[i],0) #change 0 to other parameter vector like mass
    H=np.zeros((Fsize,Fsize),dtype=np.float64)
    for ib in range(0,Nshape[0]): #interate over boson species
        add=np.sum(Fsizearr[:ib])
        H[diag[ib]+add,diag[ib]+add]=np.sum(basis[ib]*Earr[ib],axis=1) #filling in noninteracting part
        if G[ib,ib]==0: #see if self interacting
            continue
        else: #Fill in self interactions
            g=G[ib,ib]
            for ii in range(0,Fsizearr[ib]):
                statei=basis[ib][ii]
                for k in range(0,Karr[ib]):
                    if statei[k]==0:
                        continue
                    else:
                        for l in range(0,Karr[ib]):
                            if statei[l]==0  or (l==k and statei[l]==1):
                                continue
                            else:
                                state=statei+0 #copy state
                                A=np.sqrt(state[k]) #annihilate
                                state[k]-=1
                                A*=np.sqrt(state[l]) #annihilate
                                state[l]-=1
                                for i in range(0,Karr[ib]):
                                    for j in range(0,Karr[ib]):
                                        if ((i+j+k+l)%2!=0):
                                            continue
                                        else:
                                           stateout=state+0 #copy state
                                           B=A*np.sqrt(stateout[i]+1) #create
                                           stateout[i]+=1
                                           B*=np.sqrt(stateout[j]+1) #create
                                           stateout[j]+=1


                                           #print(stateout)
                                           index=searchBasis(stateout,ib) #find state
                                           #print(state,ii,'in')
                                           #print(stateout,index,'\n')
                                           #print((g/2)*A*U[i,j,k,l])
                                           if index>ii:
                                               H[ii+add,index+add]+=(g/2)*B*U[i,j,k,l]
                                               H[index+add,ii+add]+=(g/2)*B*U[i,j,k,l]
                                           if index==ii:
                                               H[ii+add,index+add]+=(g/2)*B*U[i,j,k,l]
        for jb in range(ib,Narr[ib]):
            if G[ib,jb]==0:
                continue
            else:
                add1=add
                add2=np.sum(Fsizearr[:jb])
                for ii in range(0,Fsizearr[ib]):
                    state1i=basis[ib][ii]
                    for jj in range(0,Fsizearr[jb]):
                        state2i=basis[jb][ii]
                        for k in range(0,Karr[jb]): #SPECIES 2
                            if state2i[k]==0:
                                continue
                            else:
                                state2=state2i+0 #copy SPECIES 2
                                for l in range(0,Karr[ib]): #SPECIES 1
                                    if state1i[l]==0:
                                        continue
                                    else:
                                        state1=state1i+0 #copy SPECIES 1
                                        A=np.sqrt(state2[k]) #annihilate SPECIES 2
                                        state2[k]-=1
                                        A*=np.sqrt(state1[l]) #annihilate SPECIES 1
                                        statei[l]-=1
                                        for i in range(0,Karr[ib]): #it SPECIES 1
                                            for j in range(0,Karr[jb]): #it SPECIES 2
                                                if ((i+j)%2!=0 and (k+l)%2!=0):
                                                    continue
                                                else:
                                                   stateout1=state1+0 #copy state SPECIES 1
                                                   stateout2=state2+0 #copy state SPECIES 2
                                                   B=A*np.sqrt(stateout1[i]+1) #create SPECIES 1
                                                   stateout1[i]+=1
                                                   B*=np.sqrt(stateout2[j]+1) #create SPECIES 2
                                                   stateout2[j]+=1
        
        
                                                   #print(stateout)
                                                   index1=searchBasis(stateout1,ib) #find state SPECIES1
                                                   index2=searchBasis(stateout2,jb) # find state SPECIES2
                                                   #print(state,ii,'in')
                                                   #print(stateout,index,'\n')
                                                   #print((g/2)*A*U[i,j,k,l])
                                                   #NEED TO MODIFY CORRECT HAMILTONIAN AT ind1 + add1, ind2+add2
                                                   if index>ii:
                                                       H[ii+add,index+add]+=(g/2)*B*U[i,j,k,l]
                                                       H[index+add,ii+add]+=(g/2)*B*U[i,j,k,l]
                                                   if index==ii:
                                                       H[ii+add,index+add]+=(g/2)*B*U[i,j,k,l]
                



    return(H)



def generate(N,nb):
        states = np.zeros((int(comb(nb+N-1, nb)), N), dtype=int)
        states[0, 0]=nb
        ni = 0  # init
        for i in range(1, states.shape[0]):
            states[i,:N-1] = states[i-1, :N-1]
            states[i,ni] -= 1
            states[i,ni+1] += 1+states[i-1, N-1]
            if ni >= N-2:
                if np.any(states[i, :N-1]):
                    ni = np.nonzero(states[i, :N-1])[0][-1]
            else:
                ni += 1
        return (np.array(states.tolist()))

#USER INPUT  ---=========================================================
Karr=np.array([4,3]) #states
Narr=np.array([2,2]) #particles bosons
Nshape=Narr.shape
G=np.array([[1,1],[1,1]])
L=int(2**10)
xmax=int(7)
nd=4
#USER INPUT  ---=========================================================



#Calculating Useful Values
Fsizearr=np.zeros(Nshape,dtype=int) #Size of each fock basis
for i in range(0,Nshape[0]):
    Fsizearr[i]=int(comb(Karr[i]-1+Narr[i],Karr[i]-1)) #size of each Fock Basis
Fsize=int(np.sum(Fsizearr)) #total size

xmin=int(-xmax) #positions
ix=np.arange(0,L) #indexing for fast access
x=np.linspace(xmin,xmax,L)
xnd=np.linspace(xmin,xmax,L//nd) #reduces position for fast vis
dx=x[1]-x[0]


#Make Fock Bases
basis=np.zeros(Nshape,dtype='object')
BinomialTable=np.zeros((Nshape),dtype='object')
for i in range(0,Nshape[0]):
    basis[i] = generate(Karr[i],Narr[i])
    BinomialTable[i]=getBTable(Karr[i],Narr[i])

#Get all necessary wavefunctions and interactions
Psi=getPsi(L,np.amax(Karr)) #get K wavefunctions (just need to calculate the max since they are the same)
U=getU(int(np.amax(Karr)))     #get the wave function overlaps just need max for same reason



H=createH(1)
plt.imshow(H)
