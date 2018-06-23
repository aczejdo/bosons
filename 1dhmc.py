# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:07:41 2018

@author: Aleks
"""

import numpy as np
import argparse          
from argparse import RawTextHelpFormatter
import tqdm
import sys
import time

def hybrid_mc(n,d,omega,M,saves,T,dt):
    #inti phi, gen pi
    pi=gen_pi(n,d)
    phi=gen_phi(n,d)
    
    a=np.array(phi.shape)
    a_t = np.append(saves,a)
    savedconfigs=np.zeros(a_t)    
    dtimes=np.zeros((saves))
    i=0
    j=0 
    nburn=4000
    rej=0
    H=Hamiltonian(phi,pi,omega,M)
    while (i<saves):
       
        #evolve fields
        phi_0, pi_0 =leapfrog(phi,pi,d_action,dt,T,M)
        H_0=Hamiltonian(phi_0,pi_0,omega,M)
        deltaH=H-H_0
        #metropolis step
        if (np.random.rand(1)> np.min([1,np.exp(-deltaH)])):
            
            
            phi=phi_0
            j += 1 
            if (j>nburn):
                savedconfigs[i]=phi_0
                dtimes[i]=dtimes[i]+T
                #print(phi_0)
                i += 1
                print(i/saves)
        else:
            dtimes[i]=dtimes[i]+T
            rej+=1
            
        pi=gen_pi(n,d)
        
    return(savedconfigs,rej,dtimes)

def gen_phi(n,d):
    #n=n+2
    #dim=np.ones((d),dtype=int)*n
    #narr=np.array(dim)
    #phi=np.zeros(narr)
    phi=np.zeros((n+2))
    return(phi)

def gen_pi(n,d):
    #n=n+2
    #dim = np.ones((d),dtype=int)*n
    #narr= np.array(dim)
    #pi  = np.random.normal(0,size=(narr))
    pi  = np.random.normal(0,size=(n+2))
    return(pi)

def leapfrog(phi_gen,pi_gen,d_action,dt,T0,m):
    t=0
    T0=int(T0)
    
    pi_ev=np.zeros((int(T0//dt)),dtype= list)
    phi_ev=np.zeros((int(T0//dt)),dtype= list)
    
    pi_ev[0]     =   pi_gen-d_action(phi_gen,m)*dt*0.5 #pi[dt/2]
    phi_ev[0]    =   phi_gen + pi_ev[0]*dt 
    #print(pi_ev[0],phi_ev[0])
    #note actually pi[i]=pi(t+dt/2) 
    
    for i in range(0,T0):
        #pi[i+1]     =   pi[i]-(d_action(t + 0.5*dt))*dt 
        pi_ev[i+1]     =   pi_ev[i]  - d_action(phi_ev[i],m)*dt 
        phi_ev[i+1]    =   phi_ev[i] + pi_ev[i]*dt
        t += dt
        #print(pi_ev[i])
        
    pi_ev[T0]=pi_ev[T0-1]-(d_action(phi_ev[T0],m))*dt*0.5
    
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
    
def Hamiltonian(phi,pi,omega,M):
    #uses periodic ghost cells
    N=pi.shape[0]    
    i=np.arange(1,N-1)
    H=np.sum(0.5*(pi[i]**2  + omega**2 *(phi[i] - phi[i+1]) +M**2 * phi[i]**2 ))
    return(H)

def action(phi,omega,M,dt):
    N=phi.shape[0]
    ia=np.arange(0,N-1)
    na=np.arange(0,N)

    i,n=np.meshgrid(ia,na)
    
    S = (dt/2)*np.sum((1/dt**2)*(phi[i,n+1] - phi[i,n])**2 + 
        omega*(2 * (phi[i,n]) - phi[i+1,n])**2  
        +  (M**2)*(phi[i,n]**2))
        #its a gradient in the action right?
    return(S)

def d_action(phi,m):
    n=phi.shape[0]-2
    phi1=np.zeros((phi.shape))
    #im,jm,km    = np.meshgrid(np.arange(n,dtype=int)+1,np.arange(n,dtype=int)+1,np.arange(n,dtype=int)+1)
    #im,jm= np.meshgrid(np.arange(n,dtype=int)+1,np.arange(n,dtype=int)+1)
    im=np.arange(0,n,dtype=int)+1
    
    phi   = bnc_periodic(phi)
    phi1[im] = (0.5)*(phi[im-1]+phi[im+1])
    #phi1[im,jm] = (1/4)*(phi[im-1,jm]+phi[im+1,jm]+phi[im,jm-1]+phi[im,jm+1])
    #phi1[im,jm,km] = (1/6)*(phi[im-1,jm,km]+phi[im+1,jm,km]+phi[im,jm-1,km]+phi[im,jm+1,km]+phi[im,jm,km+1]+phi[im,jm,km-1])
    phi1 = phi1*m**2
    return(phi1)

def H_stack(configs,omega,M,dt):
    #uses periodic ghost cell
    # print(phi.shape,'phi shape')
    arr=np.array(configs.shape)
    arr[0]+=2
    #   print(arr, 'array')
    phi=np.zeros(arr)
    #  print(H.shape)
    phi[1:-1,:]=configs
    phi=bnc_2dperiodic(phi)
    H=np.zeros(phi.shape)
    xs=phi.shape[1]
    ts=phi.shape[0]
    # print(xs,ts,'xs ts')
    im=np.arange(1,xs-1)
    tm=np.arange(1,ts-1)
    
    xi,ti = np.meshgrid(im,tm)
    H= (1/dt[ti-1]**2)*(phi[ti+1,xi] - phi[ti-1,xi])**2 + \
        omega*(2 * (phi[ti,xi+1]) - phi[ti,xi-1])**2  + \
        (M**2)*(phi[ti,xi]**2)
    
    
    E=np.sum(H[1:-1,1:-1])
    return(E)

def main():
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)  
    parser.add_argument("prob",type=str,
                        help="field:\n"
                            "    Klein-Gordan   : default \n") 
    parser.add_argument("points",type=int,
                        help="number of spatial support points in each dimension")
       
    parser.add_argument("dim",type=int,
                        help="number of dimensions, includes a +1 for time (non functioning rn) \n")
    
                        
    N           = args.points
    d           = args.dim + 1
    problem     = args.prob
    """
    
    d=2
    N=8
    omega = 1
    M=1
    saves= 3000
    T=2
    dt=1/32
    stack,rej,dtimes = hybrid_mc(N,d,omega,M,saves,T,dt)
    #print(stack.shape)
    #print(stack)
    x=np.linspace(-1,1,N)
    print(rej)
    E=H_stack(stack,omega,M,np.ones(dtimes.shape))
    print(E/N)
    return()
main()