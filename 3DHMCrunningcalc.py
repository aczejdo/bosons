# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:24:40 2018

@author: Aleks

for things that need updates or answers
ctrl-f [#]
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def corr2point(phi):
    #dependent on dimensions of phi
    #phi comes in with size (N+2)^d (ghost cells for BC)
    Pair=np.zeros((N,N,N)) 
    Pair=phi[1,1,1]*phi[1:N+1,1:N+1,1:N+1] #trim BC cells
    
    #create 2layer flat array (really 2 columns)
    #N^2 = 2D
    #Top (left) layer = distances
    #bottom (right) layer = field amplitudes
    
    arr1d    = np.zeros((N**3,2))
    arr1d[:,1] = np.reshape(Pair,(N**3))
    distances=np.zeros((N,N,N))
    
    #fill in distances
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                distances[i,j,k]=np.sqrt(i*i + j*j + k*k)

    arr1d[:,0] = np.reshape(distances,(N**3)) #flatten distance array

    
    global Sarr #Sorted array variable for averaging alg checknext
    Sarr = np.zeros((N**3,2)) #init Sorting array
    Sarr = arr1d[ np.argsort(arr1d[:,0])] #sorting arr1d based on dist, and sending to Sarr
    #Sarr = np.abs(Sarr) # [#]

    uniquedist=np.max(np.shape(np.unique(arr1d[:,0]))) #counting unique distances-# is size of vector to plot corr
    
    dist   = np.zeros((uniquedist)) #array for storing radial distances
    corr   = np.zeros((uniquedist)) #array for storring ,phi_0*phi_r

    #corr[0]=Sarr[0,1] #set 000
    corr[-1]=Sarr[-1,1] #set n,n,n
    
    i=1 #uniqdist index
    sarri=0 #sorting array index
    global di #index for translating between sarri and i
    di=0 
    while(i<uniquedist):
    
        
        sumcorr,null1=checknext(1,0,0,sarri)
        corr[i-1]=(1/(uniquedist*kappa))*(sumcorr)/(di) #mean
        
        sarri=sarri+di #next nonunique distance
        dist[i]=Sarr[sarri,0] #save
        #print(di)
        di=0 #reset di
        i=i+1 #move to next uniquedist

  

        
    global AnalyticCorrNoninteracting #what it says
    AnalyticCorrNoninteracting=np.zeros((uniquedist)) 
    eigfuncvec=np.zeros((uniquedist)) #seperate var for organization
    
    #From both ch24 of Boudreau/ my own calculation
    #i will move this out of here and use returned vars soon
    #ku=(np.pi/N)*np.arange(0,N)
    for k in range(0,uniquedist):
        A=np.cos(dist[k]*np.pi*2/dist[-1])
        #A=1
        eigfuncvec[k] = A/ ( ((2-4*g)/kappa -(2*d)) + 4*(np.sin(dist[k]*np.pi/dist[-1]))**2)
        
    AnalyticCorrNoninteracting=eigfuncvec*(1/(uniquedist*kappa)) #this is finessed [#]  
    UNQD[0]=uniquedist+0. #global uniquedist var for plotting
    
    return(corr,dist,uniquedist)

def checknext(level,bottom,runsum,ind):
    #this function is independent of dimensions as Sarr is flattened
    #recursively avgs over amplitudes with the same radial distances
    
    if (Sarr[ind,0]==Sarr[ind+1,0]): #if next value is the same
            runsum,bottom=checknext(level+1,bottom,runsum,ind+1) #recurse with levl and ind+1
            runsum+=Sarr[ind,1] 
    else:
            runsum+=Sarr[ind,1] #add to sum for previous layer
            bottom=ind #bottom is number of layers
       
    if level==1: #if we are at the top
        global di #this is sloppy, but it measures the unique dist index vs the Sarr index
        di=bottom-ind  +1 
    return(runsum,bottom)

def hybrid_mc():
    #this function is independent of dimensions
    #Initialization
    #===================================
    pi_0   = heatbath_pi() 
    phi_0  = np.zeros(pi_0.shape)
    
    #a_t = np.append(N_saves,np.array(phi_0.shape)) #creating array for N_saves field configs
    #Saves=np.zeros(a_t)
    dE=np.zeros(N_saves) #for tracking energy change
    #===================================
    
    H0=H_MD(phi_0,pi_0)
    print(H0, 'H0')    

    
    rej=0
    temprej=0
    i=0
    
    #note
    #This section is for thermalizing code at the start of a run for a specific number of steps
    #I included this for measuring ergodicity
    #Normally users should set N_therm to 0
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
            if VERBOSE==1:
                print(H_new,'H_new',P_acc,'exp dH','ACCEPTED SAVE %.3f'%(i/N_saves),iii)
            H0 =  H_new +0.
            phi_0 = phi_new +0.
            #Saves[i]=phi_new +0.
            dE[i] = P_acc + 0.
            Analysis(phi_new,i)
            temprej=0
            i+=1
        else:
            if VERBOSE==1:
                print(H_new,'H_new',P_acc,'exp dH','REJECTED SAVE')
            temprej+=1 
            rej +=1
            if temprej>rejmax:
                os.exit()

        pi_0 = heatbath_pi()  
        #----------------------------------------------
        
        
        
    rate = (N_saves/(rej+N_saves))
    return(rate,dE)

def thermalize(phi,pi,H0,T_max):
    #this function is independent of dimensions
    #--------------------------------------
    phi_new,pi_new = leapfrog(phi,pi,T_max)
    H_new          = H_MD(phi,pi)
    
    deltaH = H_new - H0
    P_acc = np.exp(-deltaH)
    if (np.random.rand()<=P_acc):
        H_0 =  H_new
        if VERBOSE==1:
            print(H_new,'H_new',P_acc,'exp dH','ACCEPTED THERM')
        phi_0 = phi_new
        pi_0=pi_new
    
    else:
        if VERBOSE==1:
            print(H_new,'H_new',P_acc,'exp dH','REJECTED THERM')
        pi_new = heatbath_pi()
        phi_0,pi_0,H_0=thermalize(phi_new,pi_new,H0,T_max)
        
    return(phi_0,pi_0,H_0)
def leapfrog(phi,pi,T_max):
    #this function is independent of dimensions
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
    pi=fBNC(pi)

    p_term  = 0.5*np.sum( pi[xi,yi,zi]**2) 
    s_term  =   Hamiltonian(phi)
    H   =    p_term  + s_term
    return(H/(N**3))

def d_action(phi):
    phi=fBNC(phi) 
    force=np.zeros((np.array(phi.shape)))

    
    #from ch24
    Jx = phi[xi+1,yi,zi] + phi[xi,yi+1,zi] +  phi[xi,yi,zi+1] +\
         phi[xi-1,yi,zi] + phi[xi,yi-1,zi] +  phi[xi,yi,zi-1] 
    
    #hmctut
    """
    force[1:N+1,1:N+1,1:N+1]=    -2*kappa * Jx +\
        2*phi[xi,yi,zi] + \
        4*g*phi[xi,yi,zi]*(phi[xi,yi,zi]**2 - 1)
    #ch24
    """
    force[1:N+1,1:N+1,1:N+1]=    a*(-kappa * Jx + \
         2 * (1 - 2*g) * phi[xi,yi,zi] + \
         4 * g * phi[xi,yi,zi]**3 ) #+\
         #(1 - 2*g + 6*g*phi[xi,yi,zi]**2 )*a +\ small higher order terms
         #4*g*phi[xi,yi,zi] *a**2 +\ 
         #g*a**3)

    return(force)



def Hamiltonian(phi):
    Js     =   (phi[xi+1,yi,zi] + phi[xi,yi+1,zi] + phi[xi,yi,zi+1]) *phi[xi,yi,zi]
    #print(np.sum(Js),'js')
    #hmctut
    #S=  -2*kappa*Js + phi[xi,yi,zi]**2 +  g * (phi[xi,yi,zi]**2 - 1)**2
    #ch24
    S      = -kappa*Js + (1-2*g)*phi[xi,yi,zi]**2 +  g * phi[xi,yi,zi]**4
    H=np.mean(S[1:N+1,1:N+1,1:N+1])

    return(H)



def plotter(name):
    fig = plt.figure()
    plt.plot(np.arange(0,name.shape[0]),name)
    plt.show()
    return(np.mean(name), '+-' , np.std(name))
    
def heatbath_pi():
    pi_rand  = np.random.normal(0,size=(N,N,N))
    pi  = np.zeros((N+2,N+2,N+2))
    pi[1:N+1,1:N+1,1:N+1] = pi_rand
    pi = fBNC(pi)
    return(pi)

def bnc_periodic(A):
    #N+2 by N+2 in
    J=A.shape[0]-1
    #K=A.shape[1]-1
    A[0,:,:] = A[J-1,:,:]
    A[J,:,:] = A[1,:,:]
    
    A[:,0,:] = A[:,J-1,:]
    A[:,J,:] = A[:,1,:]
    
    A[:,:,0] =A[:,:,J-1]
    A[:,:,J] =A[:,:,1]
    

    
    return(A)

def main():
    rate,dE=hybrid_mc()
    print(rate*100,'% acceptance')
    if (PLOTS==1):   
        plotter(dE)
        print('dE per step')
        plotter(mag)
        print('magnetism')
        plotter(magsqr)
        print('magnetism2')
        plotter(Bcmt)
        print('Bindercmt')
        if (CORR==1):
            corravgd=np.sum(Corr2pt,axis=0)/N_saves
            fig = plt.figure()
            plt.plot(distances,corravgd,distances,AnalyticCorrNoninteracting)
            plt.show()
            print('Phiphi corr')
    return(np.sum(magsqr)/N_saves,np.std(magsqr))


def Analysis(phiconfig,ind):
    #done at each saved configuration
    magsqr[ind]     = np.mean(np.dot(phiconfig,phiconfig))
    mag[ind]  = np.mean(phiconfig)
    Bcmt[ind]    = np.mean(phiconfig**4)/((np.mean(phiconfig**2))**2)
    Energy[ind] =  Hamiltonian(phiconfig)
        
    if (CORR==1):
        if (ind<N_saves-1):
            Corr2pt[ind],null,null=corr2point(phiconfig)  
        else:
            global distances
            Corr2pt[ind],distances,null=corr2point(phiconfig)  
            #print(distances.shape,'distshape328')

    return()







#globals
#==============================================================================
#========
#Lattice
#========
N=8
a = 1 #lattice spacing
d=3
xm=np.arange(0,N,dtype=int)+1
xi,yi,zi=np.meshgrid(xm,xm,xm,indexing='ij')
#======
#MC
#======
Tmax_MD = 1.0 #MD time
dT_MD = 0.05 #timestep in MD time
N_saves = 5000 #number of saved field configs
N_therm = 0 #initial thermalization see note in hybrid_mc

T_therm = 5 #therm between measurements
rejmax=20 #max consecutive rejections

#==========
#Physics
#=========
fBNC = bnc_periodic #the most important, shoutout to Joaquin
g=0 #coupling
kappa=.3 #hopping parameter

#observables
Energy   = np.zeros((N_saves))
mag      = np.zeros((N_saves))
magsqr   = np.zeros((N_saves))
Bcmt     = np.zeros((N_saves))
Cp       = np.zeros((N_saves))
Corr2pt  = np.zeros((N_saves),dtype='object')
global UNQD
UNQD=np.zeros((1))




#============
#organization
#=============
iii=0 #current 1d run for many main runs
jjj=0 #current 2d run for many main runs
RUNTYPE=0 #see below
VERBOSE=1 #whether or not to print individual Met-Hast steps 1 is on
PLOTS=1 #plot observables like magnetism or correlations
CORR=1


#==============================================================================









if RUNTYPE==0:
    #single run
    main()    
if RUNTYPE==1:
    
    nplot=8
    plotvec=np.zeros((nplot,2))
    varvec=np.linspace(0.15,.24,nplot)
    #phase transition in coupling
    for iii in range(0,nplot):
        print(iii,varvec[iii])    
        kappa=varvec[iii]
        plotvec[iii]=main()
    fig=plt.figure()
    #plt.plot(varvec,plotvec[0],'ob')
    err=plotvec[:,1]
    plt.errorbar(varvec[:],plotvec[:,0],yerr=err,fmt='o')
    plt.title('Mean Magnitude N=12')
    plt.show()
if RUNTYPE==2:
    nplot=15
    plotvec=np.zeros((nplot,2))
    varvec=np.linspace(0.15,.24,nplot)
    #phase transition in coupling
    for iii in range(0,nplot):
        print(iii,varvec[iii])    
        kappa=varvec[iii]
        plotvec[iii]=main()
    fig=plt.figure()
    #plt.plot(varvec,plotvec[0],'ob')
    err=plotvec[:,1]
    plt.errorbar(varvec[:],plotvec[:,0],yerr=err,fmt='o')
    plt.title('Mean Magnitude')
    plt.axit('kappa')
    plt.show()
if RUNTYPE==3:
    #phase transition in two variables
    nlam=5
    nmam=5
    lamvec=np.linspace(0.01,0.1,nlam)
    mamvec=np.linspace(0.01,0.1,nmam)
    lv,mv=np.meshgrid(lamvec,mamvec,indexing='ij')
    plotmat=np.zeros((nlam,nmam))
    for iii in range(0,nlam):
        g=lamvec[iii]
        for jjj in range(0,nmam):
            k=mamvec[jjj]
            plotmat[iii,jjj]=main()
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.plot_surface(lv,mv,plotmat, cmap='bone')
    ax.set_ylabel('lam1')
    ax.set_xlabel('lam2')
    ax.set_zlabel('amp')
    plt.title('phi')
    plt.show()

"""if RUNTYPE==3:
    #parallel!            
    if __name__ == '__main__':
        nplot=16
        varvec=np.linspace(4,10,nplot)
        plotvec=np.zeros((nplot))
        agents = 4
        chunksize = 4
        with Pool(processes=agents) as pool:
            plotvec = pool.map(main, m=varvec, chunksize)        
        plt.plot(varvec,plotvec,'ob')
        plt.show()  
"""     