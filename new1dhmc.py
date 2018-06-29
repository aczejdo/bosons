# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:07:41 2018

@author: Aleks
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse         
from argparse import RawTextHelpFormatter
import tqdm
import sys
import time

def hybrid_mc(n,omega,M,k,x,saves,T,dt,fBNC,nburn):
    #intitialize phi=0's, random momenta
    pi  = np.random.normal(0,size=(n+2))
    phi=np.zeros((n+2))
    
    #creating array to hold saved phi configuration
    a=np.array(phi.shape)
    a_t = np.append(saves,a)
    savedconfigs=np.zeros(a_t)   
    
    #count the time difference between field configurations
    dtimes=np.zeros((saves))
    
    i=0 #number of accepted configs
    j=0 #burn in saves
    rej=0 #counting the rejections 
    H=Hamiltonian(phi,pi,omega,M,k,x,fBNC)
    
    prog=1 #for progress bar
    #Burn in period
    while(j<nburn):
        
        #evolve fields
        phi_0, pi_0 =leapfrog(phi,pi,  d_action,dt,T,M,k,x,fBNC)
        H_0=Hamiltonian(phi_0,pi_0,   omega,M,k,x,fBNC)

        #metropolis step
        deltaH=  H -  H_0
        if (np.random.rand(1)  > np.min([1,np.exp(-deltaH)])):
            if(j%((nburn/10)*prog)==0):
                print(prog/10, 'percent burn')
                prog+=1
            phi  =  phi_0
            j +=1
      
        pi  = np.random.normal(0,size=(n+2))
        
    #Now saving configurations   
    while (i<saves):
       
        #evolve fields
        phi_0, pi_0 = leapfrog(phi,pi,d_action,dt,T,M,k,x,fBNC)
        H_0         = Hamiltonian(phi_0,pi_0,omega,M,k,x,fBNC)
        

        #metropolis step
        deltaH   =   H - H_0
        if (np.random.rand(1)  >  np.min([1,np.exp(-deltaH)])):
            
            phi  =  phi_0  #reset to new field 
            savedconfigs[i] =   phi_0 #save
            dtimes[i]       =   dtimes[i]+T #update time
            #print(phi_0)
            i +=1
            
            fig = plt.figure()
            plt.plot(np.arange(0,phi.shape[0]),phi)
            plt.title('phi')
            plt.show()
    
            print(i/saves)
        else:
            dtimes[i]  =  dtimes[i]+T
            rej +=1
            
        pi  = np.random.normal(0,size=(n+2))

        
    return(savedconfigs,rej,dtimes)







def leapfrog(phi_in,pi_in,d_action,dt,T0,m,k,x,fBNC):
    #leapfrog integrator 
    
    t=0
    
    #creating evolved field holding arrays
    pi_ev=np.zeros((int(T0//dt)),dtype= list)
    phi_ev=np.zeros((int(T0//dt)),dtype= list)
    
    #initial timestep gives phi[dt],pi[dt/2]
    pi_ev[0]     =   pi_in  - d_action(phi_in,m,k,x,fBNC)*dt*0.5
    phi_ev[0]    =   phi_in + pi_ev[0]*dt 
    
    #note actually pi[i]=pi(t+dt/2) 
    
    #now pi[t+dt/2],phi[t+dt]
    #evolving up to time T0
    for i in range(0,T0):
        pi_ev[i+1]     =   pi_ev[i]  - d_action(phi_ev[i],m,k,x,fBNC)*dt 
        phi_ev[i+1]    =   phi_ev[i] + pi_ev[i]*dt
        t += dt
      
    #final step brings pi to [T0]    
    pi_ev[T0]=pi_ev[T0-1]-(d_action(phi_ev[T0],m,k,x,fBNC))*dt*0.5
    
    return(phi_ev[T0],pi_ev[T0])

def bnc_2dperiodic(A):
    #2d
    J=A.shape[1]-1
    I=A.shape[0]-1
    A[0, 0:J] =  A[I-1, 0:J]
    A[I, 0:J] =  A[1  , 0:J]
    A[0:I, 0] =  A[0:I, J-1]
    A[0:I, J] =  A[0:I, 1]
    return(A)

def bnc_periodic(A):
    #1d
    A[0] = A[-2]
    A[-1] =  A[1]

    return(A)
    
def Hamiltonian(phi,pi,omega,m,k,x,fBNC):
    #print(a)
    dx=np.abs(x[1]-x[0])
    #uses periodic ghost cells
    N=phi.shape[0]    
    i=np.arange(1,N-1)
    #note no pi because its randomly selected 
    H = np.sum(omega*0.25*(1/(dx**2))*(phi[i+1] - phi[i-1])**2  + \
        (m**2)*(phi[i]**2 +k*phi[i]*x[i]**2))
        
    return(H)

def d_action(phi,m,k,x,fBNC):
    dx=np.abs(x[1]-x[0])
    n=phi.shape[0]-2
    phi1=np.zeros((phi.shape))
    phi=fBNC(phi)
    #im,jm,km    = np.meshgrid(np.arange(n,dtype=int)+1,np.arange(n,dtype=int)+1,np.arange(n,dtype=int)+1)
    #im,jm= np.meshgrid(np.arange(n,dtype=int)+1,np.arange(n,dtype=int)+1)
    
    
    im=np.arange(0,n,dtype=int)+1
    #print(phi1[im].shape)
    phi   = fBNC(phi)
    phi1[im] = (phi[im+1]-2*phi[im]+phi[im-1])/(dx**2) +(m**2 - 0.5* k * (x[im]**2))*phi[im]
    #phi1[im,jm] = (1/4)*(phi[im-1,jm]+phi[im+1,jm]+phi[im,jm-1]+phi[im,jm+1])
    #phi1[im,jm,km] = (1/6)*(phi[im-1,jm,km]+phi[im+1,jm,km]+phi[im,jm-1,km]+phi[im,jm+1,km]+phi[im,jm,km+1]+phi[im,jm,km-1])
    #phi1 = phi1*m**2
    return(phi1)

def H_stack(configs,omega,M,k,x,dt):
    dx=np.abs(x[1]-x[0])
    #create new array for adding bc's to saved field configs??
    arr=np.array(configs.shape)
    J=arr[0] # num of configs 
    K=arr[1] # num of sites (n+2)

    arr[0]=arr[0]+2 #adding periodic time cells?? Yieks
    phi=np.zeros(arr) 
    
    phi[1:arr[0]-1,:]=configs #insert original configs
    phi=bnc_2dperiodic(phi) #can has bc's????
    
    phi[0]=0
    phi[-1]=0
    
    
    H=np.zeros(phi.shape)
    #create numpy indices
    xs=phi.shape[1]
    ts=phi.shape[0]
    
    #grid out the #Bc's!!???
    im=np.arange(1,xs-1)
    tm=np.arange(1,ts-1)
    xi,ti = np.meshgrid(im,tm)
    
    #Hamiltonian  4grad(phi)^2  + m^2 phi^2
    #omega lets us control spacial grad more
    #H[1:J+1,1:K-1]= (1/dt[ti-1]**2)*(phi[ti+1,xi] - phi[ti-1,xi])**2 + \
    H[1:J+1,1:K-1]=  omega*0.25*(1/(dx**2))*(phi[ti,xi+1] - phi[ti,xi-1])**2  + \
        (M**2)*(phi[ti,xi]**2 +k*phi[ti,xi]*x[xi]**2)
        
    #Sum up aloing each field (minus bound cells) and divide by N    
    E=np.sum(H[1:J+1,1:K-1],axis=1)/(arr[1]-2)

    fig = plt.figure()
    ax      = fig.add_subplot(211)
    plt.plot(np.arange(0,configs.shape[0]),E)
    
    ax   = fig.add_subplot(212)
    plt.plot(np.arange(0,configs.shape[0]),np.abs(E))
    plt.title('energy sum')
    plt.show()
    
    E=np.sum(E)/(arr[0]-2)
    return(E)
def bnc_2dhardwall(A):
        #2d
    J=A.shape[1]-1
    I=A.shape[0]-1
    A[0, 0:J] =  0
    A[I, 0:J] =  0
    A[0:I, 0] =  0
    A[0:I, J] =  0
    return(A)

def bnc_hardwall(a):
    a[0]=0
    a[-1]=0
    return (a)

def main():  
    d=2
    N=20
    omega = 1
    M=1
    saves= 100
    nburn=100
    T=2
    k=1
    dt=1/100
    
    x=np.linspace(-N//2,(N+1)//2)
    
    stack,rej,dtimes = hybrid_mc(N,omega,M,k,x,saves,T,dt,bnc_hardwall,nburn)
    #print(stack)

    print(rej)
    onetimes=np.ones(dtimes.shape)
    E=H_stack(stack,omega,M,k,x,onetimes)
    print((E),'energy?')
    #wait = input("PRESS ENTER TO CONTINUE.")
    return(E)



main()


runs=10
#Nset=[4,8,16,32]
vec=np.zeros((runs))
for iii in range(0,runs):
    print(iii)
    vec[iii]=main()
    
print(vec)   
print(np.mean(vec),'mean') 
print(np.std(vec),'std')