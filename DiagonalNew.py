#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:44:22 2019

@author: aleksczejdo
"""

import numpy as np
from scipy.special import comb
from timeit import default_timer as timer
from multiprocessing import Pool
from scipy.special import eval_hermite as eh
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.cm as cm
from functools import partial

#-------------Generate Fock Basis, Stackexchange++-------------------------------
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
        return np.array(states.tolist())
    
    
    
#-------------Builds array of all ijkl a and a*T -------------------------------    
#To be used if we get this parallelized
def getBaseK(K):
    BK=np.zeros((K**4,4),dtype=int)
    ind=0
    for i in range(0,K):
        for j in range(0,K):
            for k in range(0,K):
                for l in range(0,K):
                    if (i+j+k+l)%2 !=0:
                        continue
                    else:
                        BK[ind,0:4]=[i,j,k,l]
                        ind+=1
    BaseKarr=BK[np.nonzero(BK)]
    return(BaseKarr)  

#-------------Get indices of operated state-------------------------------------
#very sexy algorithm for finding the exact index of a state
def searchBasis(state):
    #global BinomialTable
    ind=0
    for i in range(0,K):
        #ind+=int(comb(K-1-i + N - (1+ np.sum(state[0:i+1])),K-1-i))
        ind+=BinomialTable[i,int(np.sum(state[0:i+1]))]
    return(ind)
#---------------Algorytm Tomasza Sowinskiego------------------------------------ 
"""
def searchBasis2(state):
    global basis
    
    digit=0
    attempts=0
    nind=Fsize//2
    new=basis[nind]
    print(state)
    while new!=state and attempts<200:
        print(digit,nind,new)
        attempts+=1
        if new[digit]<state[digit]:
            nind-=nind//2 -1
        elif new[digit]>state[digit]:
            nind+=nind//2 -1
        else:
            if digit==K:
                break
            else:
                digit+=1
        new=basis[nind]
    return(ind)
"""    
"""
#Vectorized version of newstate algorithm
def newState(state):
    A=np.sqrt(state[baseK[bpi,0]]*state[baseK[bpi,1]])*np.sqrt(state[baseK[bpi,2]]+1)*np.sqrt(state[baseK[bpi,3]]+1)
    stateout=state+0
    stateout[baseK[bpi]]+=[-1,-1,1,1[]
    for i in range(0,len(stateout)):
        indarr[i]=searchBasis(stateout[i])
    return()
"""
#--------------Get new state by acting operators---------------------------------
def newState(state,i,j,k,l):
    A=np.sqrt(k*l)*np.sqrt(i+1)*np.sqrt(j+1)
    stateout=state+0
    stateout[i]+=1
    stateout[j]+=1
    stateout[k]-=1
    stateout[l]-=1
    index=searchBasis(stateout)
    return(index,A*U[i,j,k,l])
 

#---------------------simple trapezoidal integral-------------------------------
def integ(X):
    return(np.sum(X[:])*dx - (X[0]+X[-1])*dx/2)

#--------------------get all the wave function overlaps-------------------------
def getU():
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
    
#--------------------Energy for each N -----------------------------------------    
def getEarr(K):
    Earr=np.arange(0,K)+0.5
    return(Earr)
#--------------------Build the Hamiltonian--------------------------------------    
def createH(g):
    #global U
    #print(g)
    #Earr=np.arange(0,K)+0.5
    #global bi
    bi=np.arange(0,Fsize,dtype=int)
    H=np.zeros((Fsize,Fsize),dtype=np.float64)
    Earr=getEarr(K)
    H[bi,bi]=np.sum(basis[:]*(Earr),axis=1) #multiple N by (n+1/2)
    for ii in range(0,Fsize):
        statei=basis[ii]
        for k in range(0,K):
            if statei[k]==0:
                continue
            else:
                for l in range(0,K):
                    if statei[l]==0  or (l==k and statei[l]==1):
                        continue
                    else:
                        state=statei+0
                        A=np.sqrt(state[k])
                        state[k]-=1
                        A*=np.sqrt(state[l])
                        state[l]-=1
                        for i in range(0,K):
                            for j in range(0,K):
                                if ((i+j+k+l)%2!=0):
                                    continue
                                else:
                                   stateout=state+0
                                   B=A*np.sqrt(stateout[i]+1)
                                   stateout[i]+=1
                                   B*=np.sqrt(stateout[j]+1)
                                   stateout[j]+=1
    
                                  
                                   #print(stateout)
                                   index=searchBasis(stateout) 
                                   #print(state,ii,'in')
                                   #print(stateout,index,'\n')
                                   #print((g/2)*A*U[i,j,k,l])
                                   if index>ii:
                                       H[ii,index]+=(g/2)*B*U[i,j,k,l]
                                       H[index,ii]+=(g/2)*B*U[i,j,k,l]
                                   if index==ii:
                                       H[ii,index]+=(g/2)*B*U[i,j,k,l]
                                                              

    return(H)


def getPsi(L,K):
    Psi=np.zeros((K,L))
    global ix,x,dx
    for i in range(0,K):
        A=0
        Psi[i,ix]=eh(i,x[ix])*np.exp(-0.5*x[ix]**2)
        A=integ(Psi[i,:]**2)
        Psi[i,:]=Psi[i,:]/np.sqrt(A)
    return(Psi)
        
def getBTable():
    BinomialTable=np.zeros((K,N+1),dtype=int)
    for i in range(0,K):
        for j in range(0,N+1):
            #BinomialTable[i,j]=int(comb(K-1-i + N - (1+ j),K-1-i))
            BinomialTable[i,j]=int(comb(K-1-i + N - (1+ j),K-1-i))
    return(BinomialTable)   
    
#-------------------------single particle density matrix-------------------------   
def getp(gs):
    global Psi
    p=np.zeros((K,K)) 
    for k in range(0,Fsize):
        gsk=gs[k]
        for i in range(0,K):
            for j in range(0,K):
                if basis[k,j] !=0:
                    new=(basis[k]+0.0)
                    A=np.sqrt(new[j])
                    new[j]-=1
                    new[i]+=1
                    A=A*np.sqrt(new[i])
                    
                    ind=searchBasis(new)
                    
                    p[i,j]+=gs[ind]*gsk*A
                    

                else:
                    continue
                
    return(p)
    
    
#-------------------single particle correlation density correlation--------------    
#Slow serial
def getPxy(p):
    global Psi
    M=np.zeros((L,L))
    for i in range(0,K):
        for j in range(0,K): 
            for xi in range(0,L):
                for xj in range(0,L):
                    M[xi,xj]+=p[i,j]*np.conjugate(Psi[i,xi])*Psi[j,xj]    
    return(M)
#Parallel    
def getPxyParr(p):

    #nd=2
    M=np.zeros((L//nd,L//nd))
    itind=np.arange(0,L//nd)
    
    psismaller=Psi[:,np.arange(0,L,2)]
    
    func = partial(correlate,p,psismaller)
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=16)
    result_Corr= pool.map_async(func,itind)
    pool.close()
    pool.join()
    result=result_Corr.get(timeout=120)
    M += np.array(result)  
    return(M)

#Kernel for doing correlations, point*all other points ->vector    
def correlate(p,psi,it):
    global K
    out=np.zeros((psi.shape[1]))
    for i in range(0,K):
        for j in range(0,K):
            out+=p[i,j]*np.conjugate(psi[i,it])*psi[j,:]
    return(out)

#-------------------------------2 particle density matrix-------------------------
# 
def getp2(gs):
    global Psi
    p2=np.zeros((K**2,K**2)) 
    for n in range(0,Fsize):
        gsn=gs[n]
        state_n=basis[n]+0
        for k in range(0,K):
            if state_n[k] ==0: #state n 1 is not 0
                continue
            else:
                for l in range(0,K):
                    if (l==k and state_n[k]==1) or (state_n[l] ==0): #state n 2 is not 0 and also if l==k that its>1
                        continue                    
                    else:
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
                                
                                ind=searchBasis(new)
                                p2[int(i*K+j),int(k*K +l)]+=np.conjugate(gs[ind])*gsn*B
                    
    p2=np.reshape(p2,(K**2,K**2))          
    return(p2)


#Parallel    
def getPxxyyParr(p2):

    #nd=2 #simplifies calculation by using fewer visualizing points
    M=np.zeros((L//nd,L//nd))
    itind=np.arange(0,L//nd)

    psismaller=Psi[:,np.arange(0,L,2)]
    func = partial(correlate2,p2,psismaller)
    #OR IF WE WANT TO BE VERY COOL   
    pool = Pool(processes=16)
    result_Corr= pool.map_async(func,itind)
    pool.close()
    pool.join()
    result=result_Corr.get(timeout=120)
    M += np.array(result)  
    return(M)

#Kernel for doing correlations, point*all other points ->vector    
def correlate2(p2,psi,it):
    global K
    out=np.zeros(psi.shape[1])
    for i in range(0,K):
        for j in range(0,K):
            for k in range(0,K):
                for l in range(0,K):
                    out+=p2[i*K+j,k*K+l]*np.conjugate(psi[i,it])*np.conjugate(psi[j,:])*psi[k,it]*psi[l,:]
    return(out)











    
def main():
    #Input parameters
    global L,K,N,Fsize,xmax,ix,x,dx,basis,U,Psi,BinomialTable,nd
   
    L=int(2**10)
    K=int(4) #states
    N=int(10) #particles bosons
    Fsize=int(comb(K-1+N,K-1)) #size of Fock Basis
    xmax=int(10)
    xmin=int(-xmax)
    ix=np.arange(0,L)
    x=np.linspace(xmin,xmax,L)
    nd=2 #factor which we cut down for visualizing L*L corrs
    #The scaling for each 2body corr is about 2^(exponent - 6) seconds
    #where the exponent is the power in L-log2(nd)
    
    
    xnd=np.linspace(xmin,xmax,L//nd)
    
    dx=x[1]-x[0]
    allplots=True
    
    
    
    #Generate fock basis    
    basis = generate(K,N)
    
    #optional generate indexing in base K for fast operators
    #baseK=getBaseK(16)
    #and also indexing it
    #bpi=np.arange(0,K**4)
    #indarr=np.zeros((K**4))
    
    Psi=getPsi(L,K) #get K wavefunction
    BinomialTable=getBTable() #Generate table of all posible binomial K and N for fast search
    U=getU()     #get the wave function overlaps



    NG=9
    Neig=3
    gmax=int(12)
    gmin=-gmax
    G=np.linspace(gmin,gmax,NG)
    #G=np.array([2])
    #NG=1
    global E,EV
    E=np.zeros((NG,Neig))
    EV=np.zeros((NG),dtype='object')
    for iii in range(0,NG):
        print(iii, 'of ', NG )
        S=timer()
        H=createH(G[iii])
        print('     %.5f H build time'%(timer()-S))
        S=timer()
        E[iii],EV[iii]=spl.eigsh(H,k=Neig,which='SM')
        #E[iii],EV[iii]=spl.eigsh(H,k=Neig,sigma=-1000,maxiter=10000,tol=1E-7)
        print('     %.5f eigtime \n '%(timer()-S))
    
    
    
    
    cmap = cm.get_cmap('viridis')
    E=E.T
    fig=plt.figure()
    for i in range(0,Neig):
        plt.plot(G,E[i],label='En_%i'%i,color=cmap(i/Neig))
    plt.plot(0,N/2,'mx')

    #plt.savefig('/home/aleksczejdo/Dokumenty/FewBody/KatalogZdjęć/%iParticles/%iN_%iK__SpectrumBoson.png'%(N,N,K))
    plt.show()
    if allplots==True:
        for i in range(0,NG):
        
        
            print('g= ', G[i],',    %i  of %i'%(i,NG))
            
            #Get 1 particle Density matrix
            n1=getp(EV[i][:,0])
            #print(np.trace(n1),'traceP')
            p=n1*1.0
            
            #1 particle density corr
            S=timer()
            Pxy=getPxyParr(p)
            Spxy=timer()-S
            #print(timer()-S,'Pxy time')
    
            #2 particle dm
            n2=getp2(EV[i][:,0])
            #print(np.trace(n2),'traceP2')
            p2=n2*1.0
    
            #2particle desnity corr contracting so x'=x y'=y
            S=timer()
            Pxxyy=getPxxyyParr(p2)
            #print(timer()-S,'Pxxyy time')
            Spxxyy=timer()-S
            
            if i==0:
                print(np.trace(n1),'traceP')
                print(Spxy,'Pxy time')
                print(np.trace(n2),'traceP2')
                print(Spxxyy,'Pxxyy time')
            """plt.imshow(Pmat)
            plt.show()
            plt.imshow(Pxy)
            plt.show()
            fig=plt.figure()
            fig.clf()
            plt.plot(x,np.diag(Pxy))
            plt.title('diag Pxy')
            plt.show()
            plt.imshow(Pmat2)
            plt.show()
            plt.imshow(Pxxyy)
            plt.show()
            """
            plt.figure(figsize=(14,14))
            
            ax1= plt.subplot(3,2,1)
            ax1.imshow(p)
            plt.title('Pij')
            
            ax2= plt.subplot(3,2,2)
            ax2.imshow(p2)
            plt.title('Pij,kl')
        
            
            ax3=plt.subplot(3,1,3)
            ax3.plot(xnd,np.diag(Pxy))
            ax3.set_aspect(16)
            plt.title('diag Pij')
        
            
            ax4 = plt.subplot(3,2,3)
            ax4.imshow(Pxy)    
            plt.title('One Particle DM P(x,y)')
        
            
            ax5 = plt.subplot(3,2,4)
            ax5.imshow(Pxxyy)    
            plt.title('Two Particle DM P(xx,yy)')
            
            plt.tight_layout()
            plt.suptitle('%i Particles %i Exctited States \n %.4f Coupling'%(N,K,G[i]))
            #plt.savefig('/home/aleksczejdo/Dokumenty/FewBody/KatalogZdjęć/%iParticles/%iN_%iK_%.4fG_Boson.png'%(N,N,K,G[i]))
    return()
main()
    

        