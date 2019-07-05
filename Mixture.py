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
from functools import partial
from multiprocessing import Pool
from timeit import default_timer as timer
import scipy.sparse.linalg as spl
import  pycuda.autoinit                        # initialize  pycuda
import  pycuda.gpuarray  as  gpuarray            # import  gpuarray
from  skcuda.cublas  import *
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

def getEarr(K,other):
    m=other
    Earr=m*(np.arange(0,K)+0.5)
    return(Earr)







def createH(G,marr):
    H=np.zeros((Fsize,Fsize))
    mult=np.zeros(Nshape,dtype=int)
    for i in range(0,Nshape[0]):
        mult[i]=np.product(Fsizearr[:i]) #multiplier based on size of other components

    Eset=np.zeros(Nshape,dtype='object') #Shro eq sol energies
    for i in range(0,Nshape[0]):
        Eset[i]=getEarr(Karr[i],marr[i]) #
    
    #get H0 values (from schrodinger Eq sol) THIS IS FINE
    for ii in range(0,Fsize):
       H[ii,ii]+=np.sum(np.concatenate(BASIS[ii]*Eset))

    #MAKE Hab Also probably fine
    
    for a in range(0,Nshape[0]):
        for b in range(a,Nshape[0]):
            if (G[a,b]==0): #if interaction
                continue
            elif (a==b): #self interacting
                         
                for ii in range(0,Fsize):
                    statei=BASIS[ii][a]
                    index0=searchBasis(statei,a) #get initial index in basis a 
                    for k in range(0,Karr[a]):
                        if statei[k]==0:
                            continue
                        else:
                            for l in range(0,Karr[a]):
                                if statei[l]==0  or (l==k and statei[l]==1):
                                    continue
                                else:
                                    state=statei+0
                                    A=np.sqrt(state[k])
                                    state[k]-=1
                                    A*=np.sqrt(state[l])
                                    state[l]-=1
                                    for i in range(0,Karr[a]):
                                        for j in range(0,Karr[a]):
                                            if U[i,j,k,l]==0:
                                                continue
                                            else:
                                               stateout=state+0
                                               B=A*np.sqrt(stateout[i]+1)
                                               stateout[i]+=1
                                               B*=np.sqrt(stateout[j]+1)
                                               stateout[j]+=1

                                               indexa=searchBasis(stateout,a)
                                               index=(indexa-index0)*mult[a] #really delta index

                                               H[ii ,ii+index]+=(G[a,a]/2)*B*U[i,j,k,l] 
                               
                
                
                

            else: #inter component interactions
                for ii in range(0,Fsize): 
                    stateAinit=BASIS[ii][a]
                    stateBinit=BASIS[ii][b]
                    index0A=searchBasis(stateAinit,a)
                    index0B=searchBasis(stateBinit,b)
                    for k in range(0,Karr[b]): #SPECIES 2
                        if stateBinit[k]==0:
                            continue
                        else:
                            
                            for l in range(0,Karr[a]): #SPECIES 1
                                if stateAinit[l]==0:
                                    continue
                                else:
                                    stateA=stateAinit+0 #copy SPECIES 1
                                    stateB=stateBinit+0 #copy SPECIES 2
                                    A=np.sqrt(stateB[k]) #annihilate SPECIES 2
                                    stateB[k]-=1
                                    A*=np.sqrt(stateA[l]) #annihilate SPECIES 1
                                    stateA[l]-=1
                                    for i in range(0,Karr[a]): #it SPECIES 1
                                        for j in range(0,Karr[b]): #it SPECIES 2
                                            if U[i,j,k,l]==0:
                                                continue
                                            else:
                                               stateoutA=stateA+0 #copy state SPECIES 1
                                               stateoutB=stateB+0 #copy state SPECIES 2
                                               B=A*np.sqrt(stateoutA[i]+1) #create SPECIES 1
                                               stateoutA[i]+=1
                                               B*=np.sqrt(stateoutB[j]+1) #create SPECIES 2
                                               stateoutB[j]+=1


                                               #print(stateout)
                                               indexa=searchBasis(stateoutA,a) #find state SPECIES1
                                               indexb=searchBasis(stateoutB,b) # find state SPECIES2
                                               index=int((indexa-index0A)*mult[a] + (indexb-index0B)*mult[b]) #also really delta index
                                               
                                               #if index!=0:
                                               H[ii,ii+index]+=(G[a,b])*B*U[i,j,k,l]
                                                  # H[ii+index,ii]+=(G[a,b])*B*U[i,j,k,l]
                                               #if index==0:
                                                #   H[ii,ii]+=(G[a,b])*B*U[i,j,k,l]
                                                   
                                                       
    

    
    #print(Fsizearr)
    #print(np.arange(0,np.product(Fsizearr[1:])*mult[1],dtype=int))
    return(H)


def getT(ba):
    T=np.zeros((Fsize,Fsize))
    pvec=np.zeros((Nshape[0]),dtype='object')
    A=0
    B=Fsize-1
    for i in range(0,Nshape[0]):
        pvec[i]=np.arange(0,Karr[i])
    for i in range (0,Fsize):
        P=0
        for j in range(0,Nshape[0]):
            P+=(pvec[j]@BASIS[i][j])
        if P%2==0:
            T[A,i]=1
            A+=1
        else:
            T[B,i]=1
            #T[B,i]=1
            B-=1
    return(T)

def readbasis(b):
    pvec=np.zeros((Nshape[0]),dtype='object')
    for i in range(0,Nshape[0]):
        pvec[i]=np.arange(0,Karr[i])
    for i in range (0,Fsize):
        P=0
        for j in range(0,Nshape[0]):
            P+=(pvec[j]@b[i][j])
        if P%2==0:
            print('EVEN')
        else:
            print('ODD')
    print('\n')
    return()

def plotSpectrum(G,Neig,Save):
    gplot=np.sum(np.sum(G,axis=1),axis=1)
    print(gplot.shape, E.shape)
    plt.figure()
    for i in range(0,Neig):
        plt.plot(gplot,E[:,i],label='En_%i'%i,color=cmap(i/Neig))
    #plt.plot(0,Narr[:]/2,'mx')
    #if Save==True:
    #    plt.savefig('/home/aleksczejdo/Dokumenty/FewBody/Plots/%iParticles/%iN_%iK__SpectrumBoson.png'%(N,N,K))
    plt.show()
    return()



#-------------------------------2 particle density matrix-------------------------
def getp2parr(gs,a,b):
    if a==b:
        fSLICE=makePaa
    else:
        fSLICE=makePab
    global p2,multa,multb
    multa=np.product(Fsizearr[:a]) 
    multb=np.product(Fsizearr[:b]) 
    p2=np.zeros((Karr[a]*Karr[b],Karr[a]*Karr[b])) 
    
    processes=16
    #func = partial(somefunc)
    func = partial(fSLICE,a,b,gs)
    iterable=np.arange(0,Karr[a]*Karr[b])
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=processes)
    result_p2 = pool.map_async(func,iterable)
    pool.close()
    pool.join()
    p2=np.array(result_p2.get(timeout=120)).T
    
    return(p2)
    
def makePaa(a,b,gs,kl):
    p2slicey=np.zeros((Karr[a]**2))
    l=kl%Karr[a]
    k=kl//Karr[a]
    
    K=Karr[a]

    for ii in range(0,Fsize):
        #gsii=gs[ii]
        state_n=BASIS[ii][a]
        #ind0=searchBasis(state_n,a)
        if state_n[k] ==0: #state n 1 is not 0
            continue
        elif (l==k and state_n[k]==1) or (state_n[l] ==0): #state n 2 is not 0 and also if l==k that its>1
            continue                    
        else:
            gsii=gs[ii]
            ind0=searchBasis(state_n,a)
            state_n1=state_n*1 #copy state for _,_,k,l anhillation 
            A=np.sqrt(state_n1[l]) 
            state_n1[l]-=1
            A*=np.sqrt(state_n1[k])
            state_n1[k]-=1
            for i in range(0,K):
                for j in range(0,K):
                    new=state_n1+0
                    new[j]+=1
                    B=A*np.sqrt(new[j])
                    new[i]+=1
                    B*=np.sqrt(new[i])
                    
                    inda=searchBasis(new,a)
                    ind=int((inda-ind0)*multa)
                    
                    p2slicey[int(i*K+j)]+=np.conjugate(gs[ii+ind])*gsii*B  
    return(p2slicey)
def makePab(a,b,gs,kl):  
    p2slicey=np.zeros((Karr[a]*Karr[b]))
    l=kl%Karr[a]
    k=kl//Karr[a]
    for ii in range(0,Fsize):
        #gsii=gs[ii]
        stateAinit=BASIS[ii][a]
        stateBinit=BASIS[ii][b]
        #index0A=searchBasis(stateAinit,a)
        #index0B=searchBasis(stateBinit,b)
    #for k in range(0,Karr[b]): #SPECIES 2
        if stateBinit[k]==0:
            continue
        elif stateAinit[l]==0:
            #for l in range(0,Karr[a]): #SPECIES 1
            #if stateAinit[l]==0:
            continue
        else:
            gsii=gs[ii]
            index0A=searchBasis(stateAinit,a)
            index0B=searchBasis(stateBinit,b)
            stateA=stateAinit+0 #copy SPECIES 1
            stateB=stateBinit+0 #copy SPECIES 2
            A=np.sqrt(stateB[k]) #annihilate SPECIES 2
            stateB[k]-=1
            A*=np.sqrt(stateA[l]) #annihilate SPECIES 1
            stateA[l]-=1
            for i in range(0,Karr[a]): #it SPECIES 1
                for j in range(0,Karr[b]): #it SPECIES 2

                   stateoutA=stateA+0 #copy state SPECIES 1
                   stateoutB=stateB+0 #copy state SPECIES 2
                   B=A*np.sqrt(stateoutA[i]+1) #create SPECIES 1
                   stateoutA[i]+=1
                   B*=np.sqrt(stateoutB[j]+1) #create SPECIES 2
                   stateoutB[j]+=1


                   #print(stateout)
                   indexa=searchBasis(stateoutA,a) #find state SPECIES1
                   indexb=searchBasis(stateoutB,b) # find state SPECIES2
                   index=int((indexa-index0A)*multa + (indexb-index0B)*multb)
                   #print(BASIS[ii+index])
                   #print(stateoutA,stateoutB, '\n')
                   p2slicey[int(i*Karr[b]+j)]+=np.conjugate(gs[ii+index])*gsii*B
    return(p2slicey)


#Parallel    
def getPxxyyParr(p2,a,b):

    #nd=2 #simplifies calculation by using fewer visualizing points
    M=np.zeros((L//nd, L//nd))
    itind=np.arange(0 ,L//nd)[::-1]

    psismaller=Psi[:,np.arange(0,L,nd)]
    func = partial(correlate2,p2,a,b,psismaller)
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=16)
    result_Corr= pool.map_async(func,itind)
    pool.close()
    pool.join()
    result=result_Corr.get(timeout=120)
    M += np.array(result)
    return(M)

#Kernel for doing correlations, point*all other points ->vector    
def correlate2(p2,a,b,psi,it):
    out=np.zeros(psi.shape[1])
    for i in range(0,Karr[a]):
        for j in range(0,Karr[b]):
            for k in range(0,Karr[b]):
                for l in range(0,Karr[a]):
                    out+=p2[i*Karr[b]+j,k*Karr[a]+l]*np.conjugate(psi[i,it])*np.conjugate(psi[j,:])*psi[k,it]*psi[l,:]
    return(out)



def count(b):
    e=0
    o=0
    pvec=np.zeros((Nshape[0]),dtype='object')
    for i in range(0,Nshape[0]):
        pvec[i]=np.arange(0,Karr[i])
    for i in range (0,Fsize):
        P=0
        for j in range(0,Nshape[0]):
            P+=(pvec[j]@b[i][j])
        if P%2==0:
            e+=1
        else:
            o+=1
    return(e,o)
def createHpar(Gg,marr):
    #H=np.zeros((Fsize,Fsize))
    global Gii,mult
    Gii=Gg+0
    mult=np.zeros(Nshape,dtype=int)
    for i in range(0,Nshape[0]):
        mult[i]=np.product(Fsizearr[:i]) #multiplier based on size of other components
    
    
    processes=16
    #func = partial(somefunc)
    func = partial(makeSlice)
    iterable=np.arange(0,Fsize)
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=processes)
    result_Ham = pool.map_async(func,iterable)
    pool.close()
    pool.join()
    H=np.array(result_Ham.get(timeout=120))
    print(H.shape)
    
    #H=np.zeros((Fsize,Fsize))
    #for ii in range(0,Fsize):
    #    H[ii]=makeSlice(ii)        
        
    
    
    

    #MAKE Hab Also probably fine

                                                       
    
    Eset=np.zeros(Nshape,dtype='object') #Shro eq sol energies
    for i in range(0,Nshape[0]):
        Eset[i]=getEarr(Karr[i],marr[i]) #
    
    #get H0 values (from schrodinger Eq sol) THIS IS FINE
    for ii in range(0,Fsize):
        H[ii,ii]+=np.sum(np.concatenate(BASIS[ii]*Eset))
    
    #print(Fsizearr)
    #print(np.arange(0,np.product(Fsizearr[1:])*mult[1],dtype=int))
    return(H)

def makeSlice(ii):
    global Gii,mult,U,BASIS,BinomialTable
    BASIS_ii=BASIS[ii]
    Hslice=np.zeros((Fsize))
    for a in range(0,Nshape[0]):
        for b in range(a,Nshape[0]):
            if (Gii[a,b]==0): #if interaction
                continue
            elif (a==b): #self interacting
                statei=BASIS_ii[a]
                index0=searchBasis(statei,a) #get initial index in basis a 
                for k in range(0,Karr[a]):
                    if statei[k]==0:
                        continue
                    else:
                        for l in range(0,Karr[a]):
                            if statei[l]==0  or (l==k and statei[l]==1):
                                continue
                            else:
                                state=statei+0
                                A=np.sqrt(state[k])
                                state[k]-=1
                                A*=np.sqrt(state[l])
                                state[l]-=1
                                for i in range(0,Karr[a]):
                                    for j in range(0,Karr[a]):
                                        if U[i,j,k,l]==0:
                                            continue
                                        else:
                                            stateout=state+0
                                            B=A*np.sqrt(stateout[i]+1)
                                            stateout[i]+=1
                                            B*=np.sqrt(stateout[j]+1)
                                            stateout[j]+=1
                                            
                                            indexa=searchBasis(stateout,a)
                                            index=(indexa-index0)*mult[a] #really delta index
                                    
                                            Hslice[ii+index]+=(Gii[a,a]/2)*B*U[i,j,k,l] 
   
            else: #inter component interactions
                stateAinit=BASIS_ii[a]
                stateBinit=BASIS_ii[b]
                index0A=searchBasis(stateAinit,a)
                index0B=searchBasis(stateBinit,b)
                for k in range(0,Karr[b]): #SPECIES 2
                    if stateBinit[k]==0:
                        continue
                    else:
                        
                        for l in range(0,Karr[a]): #SPECIES 1
                            if stateAinit[l]==0:
                                continue
                            else:
                                stateA=stateAinit+0 #copy SPECIES 1
                                stateB=stateBinit+0 #copy SPECIES 2
                                A=np.sqrt(stateB[k]) #annihilate SPECIES 2
                                stateB[k]-=1
                                A*=np.sqrt(stateA[l]) #annihilate SPECIES 1
                                stateA[l]-=1
                                for i in range(0,Karr[a]): #it SPECIES 1
                                    for j in range(0,Karr[b]): #it SPECIES 2
                                        if U[i,j,k,l]==0:
                                            continue
                                        else:
                                           stateoutA=stateA+0 #copy state SPECIES 1
                                           stateoutB=stateB+0 #copy state SPECIES 2
                                           B=A*np.sqrt(stateoutA[i]+1) #create SPECIES 1
                                           stateoutA[i]+=1
                                           B*=np.sqrt(stateoutB[j]+1) #create SPECIES 2
                                           stateoutB[j]+=1


                                           #print(stateout)
                                           indexa=searchBasis(stateoutA,a) #find state SPECIES1
                                           indexb=searchBasis(stateoutB,b) # find state SPECIES2
                                           index=int((indexa-index0A)*mult[a] + (indexb-index0B)*mult[b]) #also really delta index
                                           
                                           #if index!=0:
                                           Hslice[ii+index]+=(Gii[a,b])*B*U[i,j,k,l]
                                              # H[ii+index,ii]+=(G[a,b])*B*U[i,j,k,l]
                                           #if index==0:
                                            #   H[ii,ii]+=(G[a,b])*B*U[i,j,k,l]
    #print(Hslice.shape,ii, 'shape')
    return(Hslice)   
    
def getD(pxy):
    processes=16
    #func = partial(somefunc)
    func = partial(correlate,pxy)
    iterable=np.arange(0,(L//nd))
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=processes)
    result_integ= pool.map_async(func,iterable)
    pool.close()
    pool.join()
    d=np.sum(np.array(result_integ.get(timeout=120)))*(xnd[1]-xnd[0])**2
    
    
    return(d)    

def correlate(pxy,ind):
    a=0
    for i in range(0,L//nd):
        a+=pxy[ind,i]*(xnd[ind]-xnd[i])**2
    return(a)
    
def getp(gs,a):
    K=Karr[a]
    multa=np.product(Fsizearr[:a])
    p=np.zeros((K,K))
    for ii in range(0,Fsize):
        gsk=gs[ii]
        state=BASIS[ii][a]
        index0=searchBasis(state,a) #get initial index in basis a 
        for j in range(0,K):
            if state[j]==0:
                continue
            else:
                new=(state+0.0)
                A=np.sqrt(new[j])
                new[j]-=1
                for i in range(0,K):
                    out=new+0
                    out[i]+=1
                    B=A*np.sqrt(out[i])
                    
                    indexa=searchBasis(out,a)
                    ind=(indexa-index0)*multa
                    p[i,j]+=gs[ii+ind]*gsk*B

                
    return(p)    
    
def getpparr(gs,a):

    p1=np.zeros((Karr[a],Karr[a])) 
    
    processes=16
    #func = partial(somefunc)
    func = partial(makePijSlice,a,gs)
    iterable=np.arange(0,Karr[a])
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=processes)
    result_p1 = pool.map_async(func,iterable)
    pool.close()
    pool.join()
    p1=np.array(result_p1.get(timeout=120)).T
    
    return(p1)
    
def makePijSlice(a,gs,j):
    K=Karr[a]
    pslice=np.zeros((K))
    for ii in range(0,Fsize):
        gsk=gs[ii]
        state=BASIS[ii][a]
        if state[j]==0:
            continue
        else:
            new=(state+0.0)
            A=np.sqrt(new[j])
            new[j]-=1
            for i in range(0,K):
                out=new+0
                out[i]+=1
                B=A*np.sqrt(out[i])
                
                ind=searchBasis(out,a)
                
                pslice[i]+=gs[ind]*gsk*B

                
    return(pslice)
   
def getPxyParr(p,a):

    #nd=2
    M=np.zeros((L//nd,L//nd))
    itind=np.arange(0,L//nd)
    
    psismaller=Psi[:,np.arange(0,L,nd)]
    
    func = partial(correlatep1,a,p,psismaller)
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=16)
    result_Corr= pool.map_async(func,itind)
    pool.close()
    pool.join()
    result=result_Corr.get(timeout=120)
    M += np.array(result)  
    return(M)

#Kernel for doing correlations, point*all other points ->vector    
def correlatep1(a,p,psi,it):
    out=np.zeros((psi.shape[1]))
    for i in range(0,Karr[a]):
        for j in range(0,Karr[a]):
            out+=p[i,j]*np.conjugate(psi[i,it])*psi[j,:]
    return(out)        
    
# =========================================================   
#=========================================================    
# =========================================================   
# =========================================================   
#USER INPUT  ---=========================================================
TOTALTIME=timer()
Karr=np.array([8,4,4]) #states
Narr=np.array([5,1,1]) #particles bosons
Nsum=np.sum(Narr)
Nshape=Narr.shape

L=int(2**10)
xmax=int(7)
nd=8
NG=9
gmax=8
gmin=-2
gconst=0
G=np.zeros((NG,Nshape[0],Nshape[0]))
G1=np.linspace(0,gmax,NG)
G2=np.linspace(gmin,gmax,NG)
G3=np.ones(NG)*gconst
G[:,0,0]= 0.001#GAA
G[:,1,1]= 1.001 #GBB
G[:,2,2]= 1.002#GAB
G[:,0,1]=(G2+0.0001)
G[:,0,2]=G2+0.0001
Neig=5
marr=np.array([1,1,1])

#=========================================================
#=========================================================


useGPU=True
SaveTxt=False
SaveDistTxt=True

PlotInteraction=True
PlotTemperature=False
PlotSpectrum=True
PlotEntropy=False

#SavingPlots
SaveSpectrum=True
SaveEntropy=False
SaveTemperature=False
SaveInteraction=True
Nb=1
betamin=0.2
betamax=5
#Tplots=False
nd=4 #factor which we cut down for visualizing L*L corrs
#The scaling for each 2body corr is about 2^(exponent - 6) seconds
#where the exponent is the power in L-log2(nd)
immax=2
immin=-2 #harmonic oscillator x

#USER INPUT  ---=========================================================
#=========================================================
#=========================================================



#Calculating Useful Values
Fsizearr=np.zeros(Nshape,dtype=int) #Size of each fock basis
for i in range(0,Nshape[0]):
    Fsizearr[i]=int(comb(Karr[i]-1+Narr[i],Karr[i]-1)) #size of each Fock Basis
Fsize=int(np.product(Fsizearr)) #total size

xmin=int(-xmax) #positions
ix=np.arange(0,L) #indexing for fast access
x=np.linspace(xmin,xmax,L)
xnd=np.linspace(xmin,xmax,L//nd) #reduces position for fast vis
dx=x[1]-x[0]


#marr=np.random.rand(Nshape[0])
#Make Fock Bases
basis=np.zeros(Nshape,dtype='object')
BinomialTable=np.zeros((Nshape),dtype='object')
for i in range(0,Nshape[0]):
    basis[i] = generate(Karr[i],Narr[i])
    BinomialTable[i]=getBTable(Karr[i],Narr[i])

#Get all necessary wavefunctions and interactions
Psi=getPsi(L,np.amax(Karr)) #get K wavefunctions (just need to calculate the max since they are the same)
U=getU(int(np.amax(Karr)))     #get the wave function overlaps just need max for same reason

index=0
BASIS=np.zeros((Fsize,Nshape[0]),dtype='object')
if Nshape[0]==1:
    for i in range(0,Fsizearr[0]):
        BASIS[index][0]=basis[0][i]
        index+=1
if Nshape[0]==2:
    for j in range(0,Fsizearr[1]):
        for i in range(0,Fsizearr[0]):
            BASIS[index]=[basis[0][i],basis[1][j]]
            index+=1
if Nshape[0]==3:
    for k in range(0,Fsizearr[2]):
        for j in range(0,Fsizearr[1]):
            for i in range(0,Fsizearr[0]):
                BASIS[index][0]=basis[0][i]
                BASIS[index][1]=basis[1][j]
                BASIS[index][2]=basis[2][k]
                index+=1
if Nshape[0]==4:
    for l in range(0,Fsizearr[3]):
        for k in range(0,Fsizearr[2]):
            for j in range(0,Fsizearr[1]):
                for i in range(0,Fsizearr[0]):
                    BASIS[index]=np.array([basis[0][i],basis[1][j],basis[2][k]])
                    index+=1


global cmap
cmap = cm.get_cmap('viridis')

global E,EV
Entropy=np.zeros((NG))
E=np.zeros((NG,Neig))
EV=np.zeros((NG),dtype='object')
#G=np.linspace(gmin,gmax,NG)

print(Fsize, "Fock States")
print("PlotInteraction",PlotInteraction,'S?',SaveInteraction)
print("PlotTemperature",PlotTemperature,'S?',SaveTemperature)
print("PlotSpectrum",PlotSpectrum,'S?',SaveSpectrum)
print("PlotEntropy",PlotEntropy,'S?',SaveEntropy)


for iii in range(0,NG):
    print(iii, 'of ', NG-1 )
    print('g=',G[iii])
    print('Building H')
    S=timer()
    
    H=createHpar(G[iii],marr)
    print('     %.5f H build time'%(timer()-S))
    if iii==0:
        neven,nodd=count(BASIS)
        S=timer()
        T=getT(BASIS)
        print('     %.5f BD-Basis build Time'%(timer()-S))
        #plt.imshow(T)
        if useGPU==True:
            a=np.array(T,dtype=np.float32)
            a_gpu = gpuarray.to_gpu(a.copy ())   # mxk  matr.on the  device
        
            alpha = np.float32(1.0)                             # scalar  alpha
            beta = np.float32(0.0)                               # scalar  beta
        #print('Transforming Basis')
        #S=timer()
        #NewBASIS=np.dot(T,BASIS)
        #print(timer()-S,'BD basis tranform time')
        #print(neven,nodd)
    
    """plt.imshow(H,cmap=cmap)
    plt.show()
    print('Hamiltonian')"""
    #S=timer()
    #H2=T@H@T.T
    
    #print('     %.5f BD-transform time'%(timer()-S))

    print('Transforming H')
    S=timer()
    if useGPU==True:
        print('on GPU')
        b=np.array(H,dtype=np.float32).T
        c=np.zeros((Fsize,Fsize),dtype=np.float32).T
        a_gpu = gpuarray.to_gpu(a.T.copy ())   # mxk  matr.on the  device
        b_gpu = gpuarray.to_gpu(b.T.copy ())   # kxn  matr.on the  device
        c_gpu = gpuarray.to_gpu(c.T.copy ())   # mxn  matr.on the  device
        h = cublasCreate()                   # initialize  cublas  context
        # matrix -matrix  multiplication:    c=alpha*a*b+beta*c
        # syntax:
        # cublasDgemm(handle , transa , transb , m, n, k, alpha , A, lda ,
        #                         B, ldb , beta , C, ldc)
        cublasSgemm(h, 'n', 'n', a.shape[0], b.shape[1], a.shape[1],
                    alpha, a_gpu.gpudata,a.shape[0], b_gpu.gpudata,
                    b.shape[0], beta, c_gpu.gpudata, c.shape[0])
        c1=c_gpu.get ().T
        c1_gpu = gpuarray.to_gpu(c1.T.copy ()) 
        b_gpu=None
        a_gpu=gpuarray.to_gpu(a.copy())
        cublasSgemm(h, 'n', 'n', a.shape[0], b.shape[1], a.shape[1],
                    alpha, c1_gpu.gpudata,a.shape[0], a_gpu.gpudata,
                    b.shape[0], beta, c_gpu.gpudata, c.shape[0])
        H2=c_gpu.get ().T
        cublasDestroy(h) 
   
        print(timer()-S,'gpubd transform')
    else:
        print('using @')
        S=timer()
        H2=T@H@T.T
        print(timer()-S,'cpubd transform')
    #print(np.sum(H3-H2))



    #plt.imshow(H2)
    #plt.show
    print('Calculating Eigenvalues')
    S=timer()
    Etop,EVtop=spl.eigsh(H2[0:neven,0:neven],k=Neig,which='SA')
    Ebot,EVbot=spl.eigsh(H2[neven:,neven:],k=Neig,which='SA')
    print('     %.5f eigtime \n '%(timer()-S))
    
    print('finessing eigenvectors')
    Etot=np.append(Etop,Ebot)
    indmin=Etot.argsort()[:Neig]
    E[iii]=np.zeros((Neig))
    EV[iii]=np.zeros((Neig,Fsize))
    
    for i in range(0,Neig):
        if indmin[i]<Neig:
            E[iii][i]=Etop[indmin[i]]
            EV[iii][i,:neven]=EVtop.T[indmin[i]]
        else:
            E[iii][i]=Ebot[indmin[i]-Neig]
            EV[iii][i,neven:]=(EVbot.T[indmin[i]-Neig])
    print('Transforming Eigenvectors back')
    S=timer()
    for i in range(0,Neig):
        EV[iii][i]=T.T@EV[iii][i]
    print(timer()-S,'U.T@(EV) time')
    
    
    if SaveTxt==True:
        print('saving files')
        karrnarr=np.array([Karr,Narr])
        np.savetxt('/home/aleksczejdo/Dokumenty/Text/K_N.txt',karrnarr,delimiter=',')
        nev='/home/aleksczejdo/Dokumenty/Text/EV_%i.txt'%(iii)
        ne='/home/aleksczejdo/Dokumenty/Text/E%i.txt'%(iii)
        ng='/home/aleksczejdo/Dokumenty/Text/G%i.txt'%(iii)
        np.savetxt(nev,EV[iii],delimiter=',')
        np.savetxt(ne,E[iii],delimiter=',')
        np.savetxt(ng,G[iii],delimiter=',')
        print('files saved')
        #E[iii],EV[iii]=spl.eigsh(H,k=Neig,sigma=-1000,maxiter=10000,tol=1E-7)
        #Echeck,EVcheck=spl.eigsh(H,k=Neig,which='SM')
    
    #EVcheck=EVcheck.T
    
if PlotSpectrum==True:
    plotSpectrum(G,Neig,SaveSpectrum)   

nc=Nshape[0]
nplot=0
while nc>0: 
    nplot+=nc
    nc-=1
#plt.show()
#T=getT(BASIS)
#plt.imshow(T)
#print(np.sum(np.eye(Fsize)-T.T@T),'I check')
#NewBASIS=np.dot(T,BASIS)
#readbasis(BASIS)
#readbasis(NewBASIS)
ndist=nplot-len(Narr[Narr<2]) #get rid off the K=1 2 body matrix (cant do that)
#d=np.zeros((NG,nplot))
d=np.zeros((NG,ndist))
names=np.zeros((nplot,2))
if PlotInteraction==True:
    print('plotting things')
    pltind=0
    for j in range(0,Nshape[0]):
        for k in range(j,Nshape[0]):
            if j==k and Narr[k]==1:
                continue
            else:
                names[pltind,0]=j
                names[pltind,1]=k
                pltind+=1
    for i in range(0,NG):
       
        print(i/NG, 'plots')
        plt.figure(figsize=(3*Nshape[0]+2,2*Nshape[0]+2))
        pltind=0
        ndplotind=0
        for j in range(0,Nshape[0]):
            for k in range(j,Nshape[0]):
                
                if j==k and Narr[j]==1:
                    S=timer()
                    p1=getp(EV[i][0],k)
                    #print(np.trace(n2),'traceP2')
                    print(timer()-S,'1-pmatrix time')
                    S=timer()
            
                    Pxy=getPxyParr(p1,k)

                    Spxy=timer()-S
                    pltind+=1
                    diag=np.diag(Pxy)
             
                    
                    #ax2 = plt.subplot(2,nplot,pltind+nplot)
                    ax2 = plt.subplot(2,nplot,pltind)
                    ax2.imshow(Pxy,cmap=cmap, extent=[immin,immax,immin,immax])    
                    plt.title('P1 (%i) %.3f'%(j,G[i,j,k]))
                    print(Spxy, 'corr1 time',i,j,k)
                else:    
                    #2 particle dm
                    S=timer()
                    p2=getp2parr(EV[i][0],j,k)
                    #print(np.trace(n2),'traceP2')
                    print(timer()-S,'2-pmatrix time')
                    #2particle desnity corr contracting so x'=x y'=y
                    S=timer()
                    Pxxyy=getPxxyyParr(p2,j,k)
                    #print(timer()-S,'Pxxyy time')
                    Spxxyy=timer()-S
                    pltind+=1
                    diag=np.diag(Pxxyy)
                    ##ax1= plt.subplot(2,nplot,pltind+nplot)
                    #ax1=plt.plot(xnd,diag)
                    #ax1.imshow(p2,cmap=cmap, extent=[immin,immax,immin,immax])
                    #plt.title('P2 %i-%i %.3f'%(j,k,G[i,j,k]))
                    
                    #ax2 = plt.subplot(2,nplot,pltind+nplot)
                    ax2 = plt.subplot(2,nplot,pltind)
                    ax2.imshow(Pxxyy,cmap=cmap, extent=[immin,immax,immin,immax])    
                    plt.title('P2(%i : %i) %.3f'%(j,k,G[i,j,k]))
                    print(Spxxyy, 'corr2 time',i,j,k)
                    dnorm=np.sum(Pxxyy)
                    d[i,ndplotind]=getD(Pxxyy)/dnorm
                    ndplotind+=1
        if SaveInteraction==True:
            print('saving figure')
            plt.savefig('/home/aleksczejdo/Dokumenty/FewBody/Plots/Enviro/%i_Boson.png'%(i))
        plt.show()            
        print(d[i],'distances')

    if NG>1:
        plotmax=NG
        for i in range(0,NG):
            if np.amax(d[i])>4:
               plotmax=i
               break
        plt.figure()
        for i in range(0,ndist):
            #ax=plt.subplot(1,nplot,i+1)
            plt.plot(np.arange(0,plotmax),d[:plotmax,i],color=cmap(i/Neig),label=names[i])
        plt.legend(loc='best')
        
        plt.savefig('/home/aleksczejdo/Dokumenty/FewBody/Plots/Enviro/DISTANCE.png')
        np.savetxt('/home/aleksczejdo/Dokumenty/FewBody/Plots/Enviro/distance%i%i%i%i%i%i.txt'%(Narr[0],Narr[1],Narr[2],Karr[0],Karr[1],Karr[2]),d)
        plt.show()
print(timer()-TOTALTIME,'total time')    