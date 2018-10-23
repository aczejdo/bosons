# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:00:32 2018

@author: Aleks
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:24:40 2018

@author: Aleks
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d
            

def corr2point(phi):
    
    #calc Stress tensor at 00
    #Tmunu=Pmu*delnu -Gmunu*L
    STensor0=np.zeros((4))
    STensor=np.zeros((4))
    STensorPair=np.zeros((N,N),dtype=list)
    xi=1
    ti=1
    L10=(phi[xi+1,ti]-phi[xi-1,ti])**2 +(phi[xi,ti+1]-phi[xi,ti-1])**2
    L20=(m**2)*phi[xi,ti]**2
    L30=(1/4)*lam1*phi[xi,ti]**4 + (1/6)*lam2*phi[xi,ti]**6
    L0      = 0.5*L10 + 0.5*L20 +L30
    
    STensor0=[(phi[2,1]-phi[0,1])*(phi[2,1]-phi[0,1]) - L0,
                        (phi[2,1]-phi[0,1])*(phi[1,2]-phi[1,0]),
                        (phi[2,1]-phi[0,1])*(phi[1,2]-phi[1,0]),
                          (phi[1,2]-phi[1,0])*(phi[1,2]-phi[1,0]) - L0]
    #print(STensor0,'zero')

    for i in range(1,N+1):
        for j in range(1,N+1):
            L1=(phi[xi+1,ti]-phi[xi-1,ti])**2 +(phi[xi,ti+1]-phi[xi,ti-1])**2
            L2=(m**2)*phi[xi,ti]**2
            L3=(1/4)*lam1*phi[xi,ti]**4 + (1/6)*lam2*phi[xi,ti]**6
            L      =  0.5*L1 + 0.5*L2 +L3
       
            STensor=[(phi[i+1,j]-phi[i-1,j])*(phi[i+1,j]-phi[i-1,j]) - L,
                                    (phi[i+1,j]-phi[i-1,j])*(phi[i,j+1]-phi[i,j-1]),
                                    (phi[i+1,j]-phi[i-1,j])*(phi[i,j+1]-phi[i,j-1]),
                                      (phi[i,j+1]-phi[i,j-1])*(phi[i,j+1]-phi[i,j-1]) - L]
            
            STensorPair[i-1,j-1]=np.multiply(STensor0,STensor)
    #print(STensorPair,'pair')         
    arr1d    = np.zeros((N**2,2),dtype=list)
    arr1d[:,1] = np.reshape(STensorPair,(N**2))
    #print(arr1d,'arr')
    distances=np.zeros((N,N))
    
  
    for i in range(0,N):
        for j in range(0,N):
            distances[i,j]=np.sqrt(i*i + j*j)

    arr1d[:,0] = np.reshape(distances,(N**2)) #flatten distance array

    
    global Sarr
    Sarr = np.zeros((N**2,2),dtype=list) #init Sorting array
    Sarr = arr1d[ np.argsort(arr1d[:,0])] #sorting arr1d based on dist, and sending to Sarr
    Sarr = np.abs(Sarr) #WHY

    uniquedist=np.max(np.shape(np.unique(arr1d[:,0]))) #couting unique distances-# is size of vector to plot corr
    corr=np.zeros(uniquedist,dtype=list)
    corr[0]=Sarr[0,1] #set 00

    corr[uniquedist-1]=Sarr[uniquedist-1,1] #set n,n
    i=1
    sarri=0
    global di
    di=0

    while(i<uniquedist):
    
        
        sumcorr,null1=checknext(1,0,0,sarri)
        corr[i]=(sumcorr)/(di)
        sarri=sarri+di
        #print(di)
        di=0
        i=i+1
  
    output=np.zeros((uniquedist,4))


    for i in range(0,uniquedist):
        output[i]=corr[i]

    return(output,uniquedist)


def corr2pointphi(phi):
    
    #calc Stress tensor at 00
    #Tmunu=Pmu*delnu -Gmunu*L
    Pair=np.zeros((N,N))

    for i in range(1,N+1):
        for j in range(1,N+1):

            
            Pair[i-1,j-1]=phi[1,1]*phi[i,j]
    #print(STensorPair,'pair')         
    arr1d    = np.zeros((N**2,2))
    arr1d[:,1] = np.reshape(Pair,(N**2))
    #print(arr1d,'arr')
    distances=np.zeros((N,N))
    
  
    for i in range(0,N):
        for j in range(0,N):
            distances[i,j]=np.sqrt(i*i + j*j)

    arr1d[:,0] = np.reshape(distances,(N**2)) #flatten distance array

    
    global Sarr
    Sarr = np.zeros((N**2,2)) #init Sorting array
    Sarr = arr1d[ np.argsort(arr1d[:,0])] #sorting arr1d based on dist, and sending to Sarr
    Sarr = np.abs(Sarr) #WHY

    uniquedist=np.max(np.shape(np.unique(arr1d[:,0]))) #couting unique distances-# is size of vector to plot corr
    corr=np.zeros(uniquedist)
    corr[0]=Sarr[0,1] #set 00

    corr[uniquedist-1]=Sarr[uniquedist-1,1] #set n,n
    i=1
    sarri=0
    global di
    di=0

    while(i<uniquedist):
    
        
        sumcorr,null1=checknext(1,0,0,sarri)
        corr[i]=(sumcorr)/(di)
        sarri=sarri+di
        #print(di)
        di=0
        i=i+1
  
    output=np.zeros((uniquedist,4))


    for i in range(0,uniquedist):
        output[i]=corr[i]

    return(output,uniquedist)

def checknext(level,bottom,runsum,ind):
    if (Sarr[ind,0]==Sarr[ind+1,0]):
            runsum,bottom=checknext(level+1,bottom,runsum,ind+1)
            runsum+=Sarr[ind,1] 
    else:
            runsum+=Sarr[ind,1]   
            bottom=ind
       
    if level==1:
        global di
        di=bottom-ind  +1
    return(runsum,bottom)

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
        phi_0,pi_0,H_0=thermalize(phi_0,pi_0,H0,T_therm)
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
    xi,ti =np.meshgrid(xm,xm,indexing='ij')

    pi=fBNC(pi)

    p_term  = 0.5*np.sum( pi[xi,ti]**2) 
    s_term  =   Hamiltonian(phi)
    H   =    p_term  + s_term
    return(H/(N**2))

def d_action(phi):
    nx=phi.shape[0]-2 #oscilators
    phi=fBNC(phi)

    
    force=np.zeros((np.array(phi.shape)))
    xm=np.arange(0,nx,dtype=int)+1
    xi,ti=np.meshgrid(xm,xm,indexing='ij')
    
    
    FS1=(phi[xi-1,ti]+phi[xi+1,ti]+phi[xi,ti-1]+phi[xi,ti+1] - 4*phi[xi,ti])
    FS2=(m**2)*phi[xi,ti]
    FS3=lam1*phi[xi,ti]**3 + lam2*phi[xi,ti]**5
    
    force[1:nx+1,1:nx+1] =  0.5*FS1 + 0.5*FS2 + FS3
    #force[1:nx+1,1:nx+1]=    -2 * kappa * Jx + 2* phi[xi,ti] + (lam1*(phi[xi,ti]**2) +(lam2*phi[xi,ti]**4) -  1)* phi[xi,ti]
    return(force)



def Hamiltonian(phi):
    nx=phi.shape[0]-2
    xm=np.arange(0,nx)+1
    xi,ti =np.meshgrid(xm,xm,indexing='ij')
    phi=fBNC(phi)

    
    
    S1=(phi[xi+1,ti]-phi[xi-1,ti])**2 +(phi[xi,ti+1]-phi[xi,ti-1])**2
    S2=(m**2)*phi[xi,ti]**2
    S3=(1/4)*lam1*phi[xi,ti]**4 + (1/6)*lam2*phi[xi,ti]**6
    S      = 0.5*S1 + 0.5*S2 +S3
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
    
    corr1,null=corr2point(phiset[0])

    a_t = np.append(nT,np.array(corr1.shape))
    corr_stackT=np.zeros(a_t,dtype=object)
    corr_stackP=np.zeros(a_t)

    
    
    for i in range(0,nT):
        mag2[i]     = np.sum(phiset[i]**2)/(N**2)
        mag[i]  = np.sum(phiset[i])/(N**2)
        Bcmt[i]    = np.sum((phiset[i]**2)/((phiset[i]**2)**2))
        E_stack[i] =  Hamiltonian(phiset[i])
        #print(corr2point(phiset[i]).shape)
        #print(corr_stack[i].shape)
        corr_stackT[i],null = corr2point(phiset[i])
        corr_stackP[i],null = corr2pointphi(phiset[i])


    corrT=np.sum(corr_stackT,axis=0)/nT    
    corrP=np.sum(corr_stackP,axis=0)/nT    
    
    return(corrT,corrP,E_stack,mag,mag2,Bcmt)



def plotter(name):
    fig = plt.figure()
    plt.plot(np.arange(0,name.shape[0]),name)
    plt.show()
    return(np.mean(name), '+-' , np.std(name))
    
def heatbath_pi():
    pi_rand  = np.random.normal(0,size=(N,N))
    pi  = np.zeros((N+2,N+2))
    pi[1:N+1,1:N+1] = pi_rand
    pi = fBNC(pi)
    return(pi)

def bnc_periodic(A):
    #N+2 by N+2 in
    J=A.shape[0]-1
    #K=A.shape[1]-1
    A[0,:] = A[J-1,:]
    A[J,:] = A[1,:]
    
    A[:,0] = A[:,J-1]
    A[:,J] = A[:,1]

    return(A)

def main():
    stack,rate,dE=hybrid_mc()
    corrt,corrp,E_stack,magnetism,magnetism2,Bindercmt=Analysis(stack)

    
    print(rate*100,'% acceptance')
    plotter(dE)
    print('dE per step')
    plotter(magnetism)
    print('magnetism')
    plotter(magnetism2)
    print('magnetism2')
    plotter(Bindercmt)
    print('Bindercmt')

    plotter(corrt[:,0])
    print('Txx corr')
    plotter(corrt[:,1])
    print('Txt corr')
    plotter(corrt[:,2])
    print('Ttx corr')
    plotter(corrt[:,3])
    print('Ttt corr')
    plotter(corrp[:,0])
    print('Phiphi corr')
    return(np.sum(magnetism2)/N_saves)

#globals
N=8
Dim=2
fBNC = bnc_periodic
Tmax_MD = 1.0
dT_MD = 0.05
N_saves = 2000
N_therm = 150

T_therm = 5.0
rejmax=500
iii=0
m=9.2
lam1=0. #phi4
lam2=0. #phi4
PHASE=3


if PHASE==0:
    main()
if PHASE==3:
    nplot=100
    plotvec=np.zeros((nplot))
    varvec=np.linspace(8,10,nplot)
    for iii in range(0,nplot):
        print(iii,varvec[iii])    
        m=varvec[iii]
        plotvec[iii]=main()
    fig=plt.figure()
    plt.plot(varvec,plotvec,'ob')
    plt.show()  
    
if PHASE==1:
    m=0.185825
    nplot=25
    plotvec=np.zeros((nplot))
    lamvec=np.linspace(0.,2,nplot)
    for iii in range(0,nplot):
        print(iii,lamvec[iii])    
        lam2=lamvec[iii]
        plotvec[iii]=main()
    fig=plt.figure()
    plt.plot(lam1/lamvec,plotvec,'ob')
    plt.show()
if PHASE==2:
    nlam=5
    nmam=5
    lamvec=np.linspace(0.01,0.1,nlam)
    mamvec=np.linspace(0.01,0.1,nmam)
    lv,mv=np.meshgrid(lamvec,mamvec,indexing='ij')
    plotmat=np.zeros((nlam,nmam))
    for iii in range(0,nlam):
        lam2=lamvec[iii]
        for jjj in range(0,nmam):
            lam1=mamvec[jjj]
            plotmat[iii,jjj]=main()
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.plot_surface(lv,mv,plotmat, cmap='bone')
    ax.set_ylabel('lam1')
    ax.set_xlabel('lam2')
    ax.set_zlabel('amp')
    plt.title('phi')
    plt.show()
            
