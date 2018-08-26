# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:24:40 2018

@author: Aleks
"""

import numpy as np
import matplotlib.pyplot as plt



def hybrid_mc():
    #Initialization
    #===================================
    pi_0   = heatbath_pi()
    phi_0  = np.zeros(pi_0.shape)
    
    a_t = np.append(N_saves,np.array(phi_0.shape))
    Saves=np.zeros(a_t)
    dE=np.zeros(N_saves)
    #===================================
    
    H0=H_MD(phi_0,pi_0)
    print(H0, 'H0')    

    
    rej=0
    temprej=0
    i=0
    while (i<N_therm):
        phi_new,pi_new = leapfrog(phi_0,pi_0,Tmax_MD)
        H_new          = H_MD(phi_0,pi_0)
        
        deltaH = H_new - H0
        P_acc = np.exp(-deltaH)
        if (np.random.rand()<=P_acc):
            #print(H_new,'H_new',P_acc,'exp dH','ACCEPTED SAVE %.3f'%(i/N_saves),iii)
            H0 =  H_new
            phi_0 = phi_new
            temprej=0
            i+=1
        else:
            #print(H_new,'H_new',P_acc,'exp dH','REJECTED SAVE')
            temprej+=1 
            if temprej>rejmax:
                os.exit()
        pi_0 = heatbath_pi()  
        #----------------------------------------------
    #print('saving',iii)
    i=0
    while (i<N_saves):
        #Thermalizing
        #phi_0,pi_0,H_0=thermalize(phi_0,pi_0,H0,T_therm)
        #---------------------------------------
        #now saving
        #---------------------------------------------        
        phi_new,pi_new = leapfrog(phi_0,pi_0,Tmax_MD)
        H_new          = H_MD(phi_0,pi_0)
        
        deltaH = H_new - H0
        P_acc = np.exp(-deltaH)
        if (np.random.rand()<=P_acc):
            print(H_new,'H_new',P_acc,'exp dH','ACCEPTED SAVE %.3f'%(i/N_saves),iii)
            H0 =  H_new
            phi_0 = phi_new
            Saves[i]=phi_new
            dE[i] = P_acc
            temprej=0
            i+=1
        else:
            print(H_new,'H_new',P_acc,'exp dH','REJECTED SAVE')
            temprej+=1 
            rej +=1
            if temprej>rejmax:
                os.exit()
        pi_0 = heatbath_pi()  
        #----------------------------------------------
        
        
        
    rate = (N_saves/(rej+N_saves))
    return(Saves,rate,dE)
def thermalize(phi,pi,H0,T_max):
    #--------------------------------------
    phi_new,pi_new = leapfrog(phi,pi,T_max)
    H_new          = H_MD(phi,pi)
    
    deltaH = H_new - H0
    P_acc = np.exp(-deltaH)
    if (np.random.rand()<=P_acc):
        H_0 =  H_new
        print(H_new,'H_new',P_acc,'exp dH','ACCEPTED THERM')
        phi_0 = phi_new
        pi_0=pi_new
    
    else:
        print(H_new,'H_new',P_acc,'exp dH','REJECTED THERM')
        pi_new = heatbath_pi()
        phi_0,pi_0,H_0=thermalize(phi_new,pi_new,H0,T_max)
        
    return(phi_0,pi_0,H_0)
def leapfrog(phi,pi,T_max):
    #leapfrog integrator 
    phi=phi+0.0
    pi=pi+0.0
    #initial timestep gives phi[dT],pi[dT/2]    
    pi_ev     =   pi  -  d_action(phi)* dT_MD*0.5
    phi_ev    =   phi

    t=0
    while t<T_max:
        phi_ev    =   phi_ev + pi_ev*dT_MD

        dS=d_action(phi_ev)
        
        #print(np.sum(dS),dS.shape,'sum,shape ds')
        pi_ev  =   pi_ev  - dS*dT_MD

        t += dT_MD
        t=np.round(t,6)

    #final step brings pi to [T0]    
    pi_ev=pi_ev-(d_action(phi_ev))*dT_MD*0.5

    return(phi_ev,pi_ev)

def H_MD(phi,pi):
    nx=phi.shape[0]-2
    xm=np.arange(0,nx)+1
    xi,yi,zi,ti =np.meshgrid(xm,xm,xm,xm,indexing='ij')

    pi=fBNC(pi)

    p_term  = 0.5*np.sum( pi[xi,yi,zi,ti]**2) 
    s_term  =   Hamiltonian(phi)
    H   =    p_term  + s_term
    return(H/(N**4))

def d_action(phi):
    nx=phi.shape[0]-2 #oscilators
    phi=fBNC(phi)

    
    force=np.zeros((np.array(phi.shape)))
    xm=np.arange(0,nx,dtype=int)+1
    xi,yi,zi,ti=np.meshgrid(xm,xm,xm,xm,indexing='ij')
    
    
    Jx = phi[xi+1,yi,zi,ti] + phi[xi,yi+1,zi,ti] + phi[xi,yi,zi+1,ti] + phi[xi,yi,zi,ti+1] + \
         phi[xi-1,yi,zi,ti] + phi[xi,yi-1,zi,ti] + phi[xi,yi,zi-1,ti] + phi[xi,yi,zi,ti-1]
    
    force[1:nx+1,1:nx+1,1:nx+1,1:nx+1]=    -2 * kappa * Jx + 2* phi[xi,yi,zi,ti] + 4*g*((phi[xi,yi,zi,ti]**2) - 1)* phi[xi,yi,zi,ti]
    return(force)



def Hamiltonian(phi):
    nx=phi.shape[0]-2
    xm=np.arange(0,nx)+1
    xi,yi,zi,ti =np.meshgrid(xm,xm,xm,xm,indexing='ij')
    phi=fBNC(phi)

    
    
    Js     =   (phi[xi+1,yi,zi,ti] + phi[xi,yi+1,zi,ti] + phi[xi,yi,zi+1,ti] + phi[xi,yi,zi,ti+1])   *phi[xi,yi,zi,ti]
    #print(np.sum(Js),'js')
    S      = -2*kappa*Js + phi[xi,yi,zi,ti]**2 + g*(phi[xi,yi,zi,ti]**2 -1)**2
    H=np.sum(S)

    return(H)


def Analysis(phiset):
    nx=phiset.shape[2]-2 #oscilators
    nT=phiset.shape[0]
    #xm=np.arange(0,nx,dtype=int)+1
    #xi,yi,zi,ti=np.meshgrid(xm,xm,xm,xm,indexing='ij')


    E_stack   = np.zeros((nT))
    mag       = np.zeros((nT))
    mag2       = np.zeros((nT))
    Bcmt      = np.zeros((nT))
    
    for i in range(0,nT):
        mag2[i]     = np.sum(phiset[i]**2)/(N**4)
        mag[i]  = np.sum(phiset[i])/(N**4)
        Bcmt[i]    = np.sum((phiset[i]**4)/((phiset[i]**2)**2))
        E_stack[i] =  Hamiltonian(phiset[i])

    
    return(E_stack,mag,mag2,Bcmt)



def plotter(name):
    fig = plt.figure()
    plt.plot(np.arange(0,name.shape[0]),name)
    plt.show()
    return(np.mean(name), '+-' , np.std(name))
    
def heatbath_pi():
    pi_rand  = np.random.normal(0,size=(N,N,N,N))
    pi  = np.zeros((N+2,N+2,N+2,N+2))
    pi[1:N+1,1:N+1,1:N+1,1:N+1] = pi_rand
    pi = fBNC(pi)
    return(pi)

def bnc_periodic(A):
    #N+2 by N+2 in
    J=A.shape[0]-1
    #K=A.shape[1]-1
    A[0,:,:,:] = A[J-1,:,:,:]
    A[J,:,:,:] = A[1,:,:,:]
    
    A[:,0,:,:] = A[:,J-1,:,:]
    A[:,J,:,:] = A[:,1,:,:]
    
    A[:,:,0,:] = A[:,:,J-1,:]
    A[:,:,J,:] = A[:,:,1,:]
    
    A[:,:,:,0] = A[:,:,:,J-1]
    A[:,:,:,J] = A[:,:,:,1] 
    return(A)

def main():
    stack,rate,dE=hybrid_mc()
    E_stack,magnetism,magnetism2,Bindercmt=Analysis(stack)
    
    print(rate*100,'% acceptance')
    plotter(dE)
    print('dE per step')
    plotter(magnetism)
    print('magnetism')
    plotter(magnetism2)
    print('magnetism2')
    plotter(Bindercmt)
    print('Bindercmt')
    return(np.sum(magnetism2)/N_saves)

#globals
N=8
fBNC = bnc_periodic
Tmax_MD = 1.0
dT_MD = 0.1
N_saves = 500
N_therm = 100

T_therm = 0.0
rejmax=100
iii=0
g=1.1689 #phi4
#kappa=0.185825 #hopping parameter
#main()

nplot=10
plotvec=np.zeros((nplot))
kappavec=np.linspace(0.10,0.2,nplot)
for iii in range(0,nplot):
    print(iii,kappavec[iii])    
    kappa=kappavec[iii]
    plotvec[iii]=main()
fig=plt.figure()
plt.plot(kappavec,plotvec,'ob')
plt.show()
