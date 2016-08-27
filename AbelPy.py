# -*- coding: utf-8 -*-
"""
Created on Fri May 08 10:50:55 2015

@author: Andrew B. Swantek, May 2015
Email:aswantek@anl.gov or andrew.b.swantek@gmail.com

#series of abel transforms from the 

Dasch paper: One-dimensional tomography: a comparison of abel, onion-peeling, and filtered back projection methods
Appied Optics V. 31 No. 9 1992 

THE DASCH PAPER WAS FIXED IN JOSEPH RANALLI'S DISSERTATION AND THOSE EQUATIONS ARE USED.

The problem is formulated such that:

            1      inf
f(r_i) =  -------- = sum D_ij * P(r_j)
            dr     j=0

+ f(f_i) is the radial field distribution (what we want)

+ P(r_j) is the projected distribution (what we measure)

+ D_ij is a linear operator which can take be constructed several ways depending on the projection method
  |
  --> D_ij is EXCLUSIVELY  a function of i and j, nothing else.
  
+ dr is the inter point spacing (this must be constant)


In this code I will construct several D_ij values and the user can select which they want

THE GRID MUST BE EQUALLY SPACED! I will leave this up to the user to do what they want.




5/13/15 update:
I have added in all the corrections to the Dasch Paper including:

Three point Abel from Ranalli's Dissertation and Villarreal& Varghese App. Opt. 2005
Two point Abel: I fixed it myself. modification is noted in the J() function





"""

import numpy as np
import matplotlib.pyplot as plt

class Abelject:

    """This is going to be a class which will contain the data and have methods for performing the inverse abel transforms.
       
       Abel Transform + Object = Abelject
    
    Attributes
    --------
    y : numpy.array or python list
        y coordinates - needs to be given
    P : numpy.array or python list
        projection data - needs to be given  \n
    dr : float
        grid spacing in units - needs to be given  \n


    r : numpy.array 
        r coordinates which are None until a transform has taken place  \n
    R : float 
        maximum value from y, i.e. outer radius
    F : numpy.array 
        field data from Abel inversion  \n
    D_ij : 2D numpy.array  
        Linear operator matrix  \n
    method : '*Onion*', *TwoPoint*', '*ThreePoint*', 'BesselMethod'
        Different methods of computing the **D_ij** matrix from the Dasch paper. Default is 'Onion'
    MethodTypes : list
        List of method types that can currently be used
    MethodNames : Dictionary
        Dictionary of method names (clean for plotting) are indexed by elements in MethodTypes
    Ny, Nz : ints
        Number of points used for 2-D interpolation, default to **201** for both Attributes
    OV : float
        The max radius for 2-D interpolated plots
    YY, ZZ : numpy.arrays()
        Contain the Y and Z coordinates where the transform will be projected
    MM : 2D numpy array        
        Contains the inverted abel data which has been turned into a 2D axisymmetric contour plot
        
    Methods
    --------
    D_ij_construct : function
        Builds the 2-D D_ij matrix based on the method in Abelject.method
    Onion, TwoPoint, ThreePoint, ThreePointModified: functions
        Called by D_ij_construct, loop over D_ij numpy array and build it up 
    J I0 I1 : functions()
        Used by Onion, Two Point, and ThreePoint functions for caclulating indivudal components of the matricies
    reconstruct : function
        Performs the reconstruction based on Equation 1 in Dasch
    abel_inversion : function
        Wrapper function which does all steps for inversion
    """
    
    
    def __init__(self, Y, P_y,dr,rmethod='ThreePoint'):
        self.y=np.array(Y)
        self.P=np.array(P_y)
        self.dr=dr
        self.R=np.max(Y)
        self.r=None
        self.F=None
        self.size=len(self.y)
        self.D_ij=None
        self.method=rmethod
        self.W_ij=None
        self.MethodTypes=['Onion','TwoPoint','ThreePoint','ThreePointModified']
        self.MethodNames={'Onion':'Onion','TwoPoint':'Two-Point',
                          'ThreePoint':'Three-Point','ThreePointModified':'Three-Point Modified'} 
        self.Ny=301
        self.Nz=301
        self.OV=self.R
        self.YY=None
        self.ZZ=None
        self.MM = None
    
    def D_ij_construct(self):
        
        #use this as a pass through to construct D_ij based on the reconstruction methods
        if self.method == 'Onion':
            self.D_ij=self.Onion()
        if self.method == 'TwoPoint':
            self.D_ij == self.TwoPoint()
        if self.method == 'ThreePoint':
            self.D_ij=self.ThreePoint()
        if self.method == 'ThreePointModified':
            self.D_ij=self.ThreePointModified()
            

    ################################################################################################
    ############# CALCULATION METHODS FOR THE ABEL RECONSTRUCTION ##################################
    ################################################################################################
            
    def Onion(self):
        """ Build the D_ij maxtrix for the onion method
            Verified by hand computations on 5/11/2015
        """
        
        self.D_ij=np.zeros( (self.size,self.size) )
        self.W_ij=np.zeros(self.D_ij.shape)
        #loop over i locations
        for i in range(self.D_ij.shape[0]):
            #loop over j locations
            for j in range(self.D_ij.shape[1]):                
                #only need two cases since we initialized W_ij to zeros
                if j == i:
                    self.W_ij[i,j] = ( (2.0*j+1)**2.0 - 4.0*i**2)**(1.0/2.0)
                if j > i:
                    self.W_ij[i,j] = ( (2.0*j+1)**2.0 - 4.0*i**2)**(1.0/2.0)  - ( (2.0*j-1)**2.0 - 4.0*i**2)**(1.0/2.0)
        
        #finally invert the array to get D_ij
        self.D_ij=np.linalg.inv(np.asmatrix(self.W_ij))
        
        return self.D_ij
    ################################################################################################        
    def OnePoint(self):
        """ Build the D_ij maxtrix for the 1-point Abel method
        This is based off the Kolhe and Agrawal paper but uses the same convention
        Note, this method runs from 1 to N+1 whereas the Dasch paper is 0 to N
        I will just setup temp varaibles to be ii=i+1 and jj=j+1 but D_ij is still indexed by i and j
         
        6/15/2015
        I haven't tested this as of 6/2015 and was having some issues with it, last I recall
        """
        self.D_ij=np.zeros( (self.size,self.size) )
        #loop over i locations
        for i in range(self.D_ij.shape[0]):
            #loop over j locations
            ii=i+1.0
            for j in range(self.D_ij.shape[1]):
                jj=j+1.0
                
                if jj==1.0 and ii==1.0:
                    self.D_ij[i,j] = 0.0
                elif jj < ii:
                    self.D_ij[i,j] = 0.0
                elif jj==ii and ii!=1.0:
                    self.D_ij[i,j] = -1/(2.0*np.pi)*np.log( ( jj + np.sqrt(jj**2.0 - (ii-1.0)*2.0 ) )
                                                          / ( (jj-1.0) + np.sqrt((jj-1.0)**2.0 - (ii-1.0)**2.0 ) ) )
                elif jj>ii and jj==2.0:
                    self.D_ij[i,j] = -1/(2.0*np.pi)*np.log(2.0 + ( jj + np.sqrt(jj**2.0 - (ii-1.0)*2.0 ) )
                                                          / ( (jj-1.0) + np.sqrt((jj-1.0)**2.0 - (ii-1.0)**2.0 ) ) )
                elif jj>ii and jj!=2.0:
                    self.D_ij[i,j] = -1/(2.0*np.pi)*np.log(( jj + np.sqrt(jj**2.0-(ii-1.0)*2.0 ) )
                                                          / ( (jj-2.0) + np.sqrt((jj-2.0)**2.0 - (ii-1.0)**2.0 ) ) )
        return self.D_ij

    ################################################################################################  
    def TwoPoint(self):
        """ Build the D_ij maxtrix for the 2-point Abel method
            
        """
        self.D_ij=np.zeros( (self.size,self.size) )
        #loop over i locations
        for i in range(self.D_ij.shape[0]):
            #loop over j locations
            for j in range(self.D_ij.shape[1]):
  
                if j < i:
                    self.D_ij[i,j]= 0
                elif j==i:
                    self.D_ij[i,j]= self.J(i,j)
                elif j>i:
                    self.D_ij[i,j]=self.J(i,j)-self.J(i,j-1)
        return self.D_ij
        
    ################################################################################################        
    def ThreePoint(self):
        """ Build the D_ij maxtrix for the 2-point Abel method
            NOTE HAS BEEN MODIFIED BASED ON ERRORS FOUND VIA RANALLI DISSERTATION            
        """
        self.D_ij=np.zeros( (self.size,self.size) )
        #loop over i locations
        for i in range(self.D_ij.shape[0]):
            #loop over j locations
            for j in range(self.D_ij.shape[1]):
  
                if j < (i-1):
                    self.D_ij[i,j] = 0
                elif j==(i-1):
                    self.D_ij[i,j] = self.I0(i,j+1)-self.I1(i,j+1)
                elif j==i:
                    self.D_ij[i,j] = self.I0(i,j+1)-self.I1(i,j+1)+2.0*self.I1(i,j)           
                elif j>=(i+1):
                    self.D_ij[i,j] = self.I0(i,j+1)-self.I1(i,j+1)+2.0*self.I1(i,j)-self.I0(i,j-1)-self.I1(i,j-1)
                elif i==0 and j==1:
                    self.D_ij[i,j] = self.I0(i,j+1)-self.I1(i,j+1)+2.0*self.I1(i,j)-2.0*self.I1(i,j-1)  

        return self.D_ij

    ################################################################################################         
    def ThreePointModified(self):
        """ Build the D_ij maxtrix for the 2-point Abel method
            NOTE HAS BEEN MODIFIED BASED ON ERRORS FOUND VIA RANALLI DISSERTATION            
        """
        self.D_ij=np.zeros( (self.size,self.size) )
        #loop over i locations
        for i in range(self.D_ij.shape[0]):
            #loop over j locations
            for j in range(self.D_ij.shape[1]):
                #print (i,j)
                if j < (i-1):
                    self.D_ij[i,j] = 0
                elif j==(i-1):
                    self.D_ij[i,j] = self.I0m(i,j+1)-self.I1m(i,j+1)
                elif j==i:
                    self.D_ij[i,j] = self.I0m(i,j+1)-self.I1m(i,j+1)+2.0*self.I1m(i,j)           
                elif j>=(i+1):
                    self.D_ij[i,j] = self.I0m(i,j+1)-self.I1m(i,j+1)+2.0*self.I1m(i,j)-self.I0m(i,j-1)-self.I1m(i,j-1)
                elif i==0 and j==1:
                    self.D_ij[i,j] = self.I0m(i,j+1)-self.I1m(i,j+1)+2.0*self.I1m(i,j)-2.0*self.I1m(i,j-1)  

        return self.D_ij
        
    ################################################################################################
    ############# SUPPORT FUNCTIONS FOR THE ABEL METHODS ###########################################
    ################################################################################################

 
    def J(self,ii,jj):
        """ used for calculatign the J_ij matrix in the 2-point abel method.
            NOTE THERE IS A CORRECTION FROM DASCH
            Dasch has the denominator as : np.sqrt( (jj-1)**2.0-ii**2.0) + jj 
            It should be np.sqrt( (jj)**2.0-ii**2.0) + jj 
        """
        ii=np.float(ii)
        jj=np.float(jj)
        
        if jj<ii:
            JJ=0.0
        elif jj==0.0 and ii==0.0:
            JJ=2.0/np.pi
        elif jj>=ii:
            JJ=1.0/np.pi*np.log( ( np.sqrt( (jj+1.0)**2.0-ii**2.0) + jj + 1.0)
                               / ( np.sqrt( (jj)**2.0-ii**2.0) + jj ) ) 
        return JJ
        
    ################################################################################################
    def I0(self,i,j):
        """ used for calculating the D_ij matrix in the 3-point abel method.
            
        """
        i=np.float(i)
        j=np.float(j)
        
        if j==0.0 and i == 0.0:
            J=0.0
        elif j<i:
            J=0.0
        elif j==i and i!=0:
            J=1.0/(2.0*np.pi)*np.log( ( np.sqrt( (2.0*j + 1.0)**2.0 - 4.0*i**2.0 ) + 2.0*j + 1.0 ) 
                                    / (2.0*j) )
        elif j>i:
            J=1.0/(2.0*np.pi)*np.log( ( np.sqrt( (2.0*j + 1.0)**2.0 - 4.0*i**2.0 ) + 2.0*j + 1.0 ) 
                                     / ( np.sqrt( (2.0*j - 1.0)**2.0 - 4.0*i**2.0 ) + 2.0*j - 1.0 ) )
        return J   
    ################################################################################################
    def I1(self,i,j):
        """ used for calculating the D_ij matrix in the 3-point abel method.
            NOTE HAS BEEN MODIFIED BASED ON ERRORS FOUND VIA RANALLI DISSERTATION
            in the last two elifs:
            Dasch has +2.0*j*I0(i,j)
            Ranalli shows that it should be -2.0*j*I0(i,j)
        """
        i=np.float(i)
        j=np.float(j)
        
        if j==0.0 and i==0.0:
            J=0.0
        elif j<i:
            J=0.0
        elif j==i and i!=0.0:
            J=1.0/(2.0*np.pi)*np.sqrt( (2.0*j+1.0)**2.0 - 4.0*i**2.0 ) - 2.0*j*self.I0(i,j)
        elif j>i:
            J=1.0/(2.0*np.pi)*( np.sqrt( (2.0*j+1.0)**2.0 - 4.0*i**2.0 ) 
                               - np.sqrt( (2.0*j-1.0)**2.0 - 4.0*i**2.0 ) ) - 2.0*j*self.I0(i,j)
        return J

    ################################################################################################
    def I0m(self,i,j):
        """ used for calculating the D_ij matrix in the 3-point abel method.
            Based on the modified method of Villarreal and Varghese
        """
        i=np.float(i)
        j=np.float(j)
        M=np.float(self.size-1)

        if j==0.0 and i == 0.0:
            J=0.0
        elif j<i:
            J=0.0
        elif j==M:
            J=0.0
        elif j==i and i!=0 and j<M-1:
            J=1.0/(2.0*np.pi)*np.log( ( np.sqrt( 4.0*j + 1.0) + 2.0*j + 1.0 ) 
                                    / (2.0*j) )
        elif j>i and j<M-1:
            J=1.0/(2.0*np.pi)*np.log( ( np.sqrt( (2.0*j + 1.0)**2.0 - 4.0*i**2.0 ) + 2.0*j + 1.0 ) 
                                     / ( np.sqrt( (2.0*j - 1.0)**2.0 - 4.0*i**2.0 ) + 2.0*j - 1.0 ) )
        elif j==i and j==M-1:
            J=1.0/(2.0*np.pi)*np.log( ( np.sqrt( 2.0*j + 1.0)  + j + 1.0 ) 
                                    / j )
        elif j==M-1 and j>i:
            J=1.0/(2.0*np.pi)*np.log( ( np.sqrt( (2.0*j + 2.0)**2.0 - 4.0*i**2.0 ) + 2.0*j + 2.0 ) 
                                     / ( np.sqrt( (2.0*j - 1.0)**2.0 - 4.0*i**2.0 ) + 2.0*j - 1.0 ) )
        else:
            J=0.0
        return J   

    ################################################################################################
    def I1m(self,i,j):
        """ used for calculating the D_ij matrix in the 3-point abel method.
            Based on the modified method of Villarreal and Varghese
        """
        i=np.float(i)
        j=np.float(j)
        M=np.float(self.size-1)

        if j==0.0 and i==0.0:
            J=0.0
        elif j==M:
            J=0.0
        elif j<i:
            J=0.0
        elif j==i and j<M-1:
            J=1.0/(2.0*np.pi)*np.sqrt( 4.0*j+1.0) - 2.0*j*self.I0m(j,j)
        elif j>i and j<M-1:
            J=1.0/(2.0*np.pi)*( np.sqrt( (2.0*j+1.0)**2.0 - 4.0*i**2.0 ) 
                               - np.sqrt( (2.0*j-1.0)**2.0 - 4.0*i**2.0 ) ) - 2.0*j*self.I0m(i,j)
        elif j==M-1 and j==i:
            J=1.0/(2.0*np.pi)*np.sqrt(8.0*j+4.0) - 2.0*j*self.I0m(j,j)
        elif j==M-1 and j>i:
            J=1.0/(2.0*np.pi)*( np.sqrt( (2.0*j+2.0)**2.0 - 4.0*i**2.0 ) 
                               - np.sqrt( (2.0*j-1.0)**2.0 - 4.0*i**2.0 ) ) - 2.0*j*self.I0m(i,j)
        else:
            J=0.0
        return J

    ################################################################################################
    ################# BUILD THE RECONSTRUCTION #####################################################
    ################################################################################################
    
    def reconstruct(self):
        """ Calculate the reconstruction based on Equation 1 in Dasch
        """

        if isinstance(self.D_ij,np.ndarray):
            self.F=np.zeros(self.size)
            for i in range(self.size):                    
                self.F[i]=1.0/self.dr*np.sum(np.array(self.D_ij[i,:])*self.P[:])
        else:
            return "Please create D_ij"
            
    
    ################################################################################################
    ################# MAIN RUN FUNCTION ############################################################
    ################################################################################################
    def abel_inversion(self):
        """This does the inversion without having to call each individual function
        """

        self.D_ij_construct()
        self.reconstruct()
        self.r=self.y
    
    ################################################################################################
    ################# GRIDDING AND 2D PLOTTING  FUNCTIONS ##########################################
    ################################################################################################    
    
    def make_2D_grid(self):
        #define here in case we change OV before hand
        self.MM= np.zeros((self.Ny,self.Nz),dtype=np.float)
        self.YY=np.linspace(-1.0*self.OV,self.OV,self.Ny)
        self.ZZ=np.linspace(-1.0*self.OV,self.OV,self.Nz)
        if isinstance(self.F,np.ndarray):
            #loop over rows
            for i in range(self.MM.shape[0]):
            #loop over columns
                for j in range(self.MM.shape[1]):
                    # calculate radius
                    r_local=np.sqrt(self.YY[i]**2+self.ZZ[j]**2)
                    #check if r exceeds the maximum r of our data
                    if r_local>self.R:
                        #hardwire to zero
                        self.MM[i,j]=0.0
                    else:

                        
                        self.MM[i,j]=np.interp(r_local,self.r,self.F)
        else:
            print "You need to do the reconstruction before you can plot!"
            
  
################################################################################################
################# RUNS IF FUNCTION IS MAIN, AS AN EXAMPLE ######################################
################################################################################################
if __name__=='__main__':
    """ Run two test cases so the user can see how the class is used.
        One will be an ellipse, the other a gaussian
    """
    plt.close('all')
    from matplotlib import rcParams
    rcParams['font.family'] = 'Garamond'
    ################################################################################################
    ################# Ellipse Benchmark ############################################################
    ################################################################################################
    
    ###### Create Elliptic Function and it's Projection ######
    x1=np.linspace(0,2,50)
    y=2*np.sqrt(1-x1[x1<1]**2)
    
    y1=np.zeros(len(x1)) #keeping y for later indexing
    y1[0:len(y)]=y 
    
    #analytic inversion
    y1_I=np.zeros(len(x1))
    y1_I[0:len(y)]=1
    ################################################################
    #create list to hold error for each abel method.
    Abel_err1=[]    
    
    #numerical inversion
    AbelTest1=Abelject(x1,y1,dr=x1[1]-x1[0],rmethod='Onion')
    AbelTest1.abel_inversion()
    Abel_err1.append(np.abs(AbelTest1.F-y1_I))
    #make a figure
    fig1,ax1=plt.subplots(figsize=(11,10))
    ax1.plot(x1,y1,'-b',lw=2,label='Original Ellipse Function, f(x)') #plot analytic function
    ax1.plot(x1,y1_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analytic abel inversion
    ax1.plot(AbelTest1.r,AbelTest1.F,'--ok',label='Onion Method', ms=10)

    #other numerical inversions
    AbelTest1.method='TwoPoint'
    AbelTest1.abel_inversion()     
    ax1.plot(AbelTest1.r,AbelTest1.F,'--oc',label='Two Point Method', ms=10) 
    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list
    
    AbelTest1.method='ThreePoint'
    AbelTest1.abel_inversion()        
    ax1.plot(AbelTest1.r,AbelTest1.F,'--om',label='Three Point Method', ms=10)
    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list
    
    AbelTest1.method='ThreePointModified'
    AbelTest1.abel_inversion()        
    ax1.plot(AbelTest1.r,AbelTest1.F,'--v',label='Modified Three Point Method',mfc='orange', ms=10)
    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list

        
    #Make plot look nice
    ax1.set_xlabel('x, r', fontsize=32)
    ax1.set_ylabel('f(x), f(r)', fontsize=32) 
    plt.xticks(size=23)
    plt.yticks(size=23)
    plt.title('Elliptic Function Test',fontsize=36,y=1.01)
    plt.grid(b='on',lw=2)
    ax1.tick_params(axis='both', pad = 10,labelsize=32)
    ax1.set_ylim(0,2.5)
    plt.legend(fontsize=22)
    
    ################################################################################################
    ################# Gaussian Benchmark ###########################################################
    ################################################################################################
    
    ###### Create Gaussian and it's Projection ######
    x2=np.linspace(0,3,50)
    sig=2/np.sqrt(np.pi)
    y2=sig*np.sqrt(np.pi)*np.exp(-x2**2/sig**2)
    #analytic inversion
    y2_I=np.exp(-x2**2/sig**2)
    ###################################################
    #create list to hold error for each abel method.
    Abel_err2=[]    
    
    #numerical inversion
    AbelTest2=Abelject(x2,y2,dr=x2[1]-x2[0],rmethod='Onion')
    AbelTest2.abel_inversion()
    
    #make a figure
    fig2,ax2=plt.subplots(figsize=(11,10))
    ax2.plot(x2,y2,'-b',lw=2,label='Original Gaussian Function, f(x)') #plot analytic function
    ax2.plot(x2,y2_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analtic abel inversion
    ax2.plot(AbelTest2.r,AbelTest2.F,'--ok',label='Onion Method', ms=10)
    Abel_err2.append(np.abs(AbelTest2.F-y2_I))#append to error list

    #other numerical inversions
    AbelTest2.method='TwoPoint'
    AbelTest2.abel_inversion()     
    ax2.plot(AbelTest2.r,AbelTest2.F,'--oc',label='Two Point Method', ms=10)  
    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    AbelTest2.method='ThreePoint'
    AbelTest2.abel_inversion()        
    ax2.plot(AbelTest2.r,AbelTest2.F,'--om',label='Three Point Method', ms=10)
    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    AbelTest2.method='ThreePointModified'
    AbelTest2.abel_inversion()        
    ax2.plot(AbelTest2.r,AbelTest2.F,'--v',label='Modified Three Point Method',mfc='orange', ms=10)
    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list
    
    #Make plot look nice
    ax2.set_xlabel('x, r', fontsize=32)
    ax2.set_ylabel('f(x), f(r)', fontsize=32) 
    plt.xticks(size=23)
    plt.yticks(size=23)
    plt.title('Gaussian Function Test',fontsize=36,y=1.01)
    plt.grid(b='on',lw=2)
    ax2.tick_params(axis='both', pad = 10,labelsize=32)
    ax2.set_ylim(0,2.5)
    plt.legend(fontsize=22)
    
    plt.show()
    
    ################################################################################################
    ################# Gaussian w/Noise Benchmark ###################################################
    ################################################################################################
    
    ###### Create Gaussian and it's Projection ######
    x3=np.linspace(0,3,50)
    sig=2/np.sqrt(np.pi)
    y3n=sig*np.sqrt(np.pi)*np.exp(-x3**2/sig**2)
    
    ###
    ###Add in noise
    ###
    
    #set the random seed for repeatability
    np.random.seed(seed=1000)

    #add in 3% noise from the peak value of the Gaussian   
    #need to subtract by 0.5 and mulitply by 2 to get interval from [0,1] to [-1,1], then multiply by
    #sig*np.sqrt(np.pi) to get the peak value 
    #finally, multiply by 0.03 to get 3% of that
    y3=y3n+(np.random.random(size=len(y3n))-0.5)*2*sig*np.sqrt(np.pi)*0.03

    #analytic inversion(w/0 noise)
    y3_I=np.exp(-x3**2/sig**2)

    ###################################################    
    Abel_err3=[]
    
    #numerical inversion
    AbelTest3=Abelject(x3,y3,dr=x3[1]-x3[0],rmethod='Onion')
    AbelTest3.abel_inversion()
    
    #make a figure
    fig3,ax3=plt.subplots(figsize=(11,10))
    ax3.plot(x3,y3n,'-b',lw=2,label='Original Gaussian Function, f(x)') #plot analytic function
    ax3.plot(x3,y3,'--bo',lw=2,label='Gaussian Function w/ Noise, f(x)',mfc='w', ms=10,mew=3,mec='b')
    ax3.plot(x3,y3_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analtic abel inversion
    ax3.plot(AbelTest3.r,AbelTest3.F,'--ok',label='Onion Method', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    #other numerical inversions
    AbelTest3.method='TwoPoint'
    AbelTest3.abel_inversion()     
    ax3.plot(AbelTest3.r,AbelTest3.F,'--oc',label='Two Point Method', ms=10)   
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    AbelTest3.method='ThreePoint'
    AbelTest3.abel_inversion()        
    ax3.plot(AbelTest3.r,AbelTest3.F,'--om',label='Three Point Method', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    AbelTest3.method='ThreePointModified'
    AbelTest3.abel_inversion()        
    ax3.plot(AbelTest3.r,AbelTest3.F,'--v',label='Modified Three Point Method',mfc='orange', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list
    
    #Make plot look nice
    ax3.set_xlabel('x, r', fontsize=32)
    ax3.set_ylabel('f(x), f(r)', fontsize=32) 
    plt.xticks(size=32)
    plt.yticks(size=32)
    plt.title('Gaussian Function w/ Noise Test',fontsize=36,y=1.01)
    plt.grid(b='on',lw=2)
    ax3.tick_params(axis='x', pad = 10,labelsize=32)
    ax3.set_ylim(0,2.5)
    plt.legend(fontsize=22)
    
    plt.show()
    
    ###### Cacluatled and print abel errors ##########################
    Fun_names=['Elliptic','Gaussian','Gaussian with noise']    
    Ab_names=['Onion', 'Two-point', 'Three-point', 'Modified 3 point']
    Ab_errorList=[Abel_err1,Abel_err2,Abel_err3]
    
    print "Error calculations for previous plots:\n"
    for i,F_nm in enumerate(Fun_names):
        print F_nm
        for j,Ab_nm in enumerate(Ab_names):
            print '\t' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] ) )
            if i==0:
                print '\t\tNoise percentage, ' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] )/np.mean(y1_I)*100 )
            if i==1:
                print '\t\tNoise percentage, ' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] )/np.mean(y2_I)*100 )
            if i==2:
                print '\t\tNoise percentage, ' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] )/np.mean(y3_I)*100 )
        
        print '\n'
        
    
    