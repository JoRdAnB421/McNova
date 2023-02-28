
version = '0'
'''
    Version 0 : Written by Jordan Barber (CU), 2021

    Based on "superbol.py" by Matt Nicholl 2015

    Computes pseudobolometric light curves and estimates full bolometric with blackbody corrections through MCMC sampling of a Gaussian processes best fit.

    Requirements and usage:
    Needs numpy, scipy and matplotlib, george, emcee and corner (warnings can also be used to reduce the amount of cluter in the output terminal)
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as itg
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interpolate as interp
import glob
import sys
import os
import emcee
import corner
import george
from george import kernels
from george.modeling import Model
from astropy.coordinates import Distance
import warnings

# This allows us to ignore the RuntimeWarnings when computing exponentials of very large numbers
warnings.filterwarnings('ignore', category=RuntimeWarning)

print('\n    * * * * * * * * * * * * * * * * * * * * * *')
print('    *           Welcome to `MCNOVA`           *')
print('    *    Supernova Bolometric Light Curves    *')
print('    *            From MCMC Sampling           *')
print('    *            Jordan  Barber (2021)        *')
print('    *                    ,                    *')
print('    *                 \  :  /                 *')
print("    *              `. __/ \__ .'              *")
print('    *              _ _\     /_ _              *')
print('    *                 /_   _\                 *')
print("    *               .'  \ /  `.               *")
print('    *                 /  :  \                 *')
print("    *                    '                    *")
print('    *                                         *')
print('    *                    V'+version+'                   *')
print('    *                                         *')
print('    * * * * * * * * * * * * * * * * * * * * * *\n\n')

# interactive plotting
ax = plt.ion()

# Define some functions:
def plt_show_interactive(pythonIDEInformed = True):
    """Show plot windows, with interaction if possible; prompt to continue.""" 
    in_ipython = ('get_ipython' in globals())
    inline_backend = ('inline' in matplotlib.get_backend())
    in_linux = (sys.platform == 'linux')
    
    if inline_backend:
        plt.show()
    elif not in_linux and in_ipython:
        print("Press Ctrl-C to continue.")
        try:    
            while True:
                plt.pause(0.5)
        except KeyboardInterrupt:
            print("Continuing.")
    elif in_linux and not in_ipython:
        # Command line: plots are interactive during wait for input.
        plt.show(block=False)
        input("Press ENTER to continue.")
    elif in_linux and in_ipython:
        # Loop with plt.pause(1) causes repeated focus stealing.
        plt.pause(1)
        if pythonIDEInformed == False:
            print("Sorry, plots might not be interactive until program has finished.")

    elif not in_linux and not in_ipython:
        # Ctrl-C is handled differently here.
        plt.pause(1)
        if pythonIDEInformed == False:           
            input("Sorry, plots might not be interactive due to python IDE setup. Press ENTER to continue.")

     
        
def cardelli(lamb, Av, Rv=3.1, Alambda = True, debug=False):
    """
    Cardelli extinction Law
    input:
        lamb    <float>    wavelength of the extinction point !! in microns !!
    output:
        tau        <float> returns tau as in redflux = flux*exp(-tau)
    keywords:
        Alambda        <bool>  returns +2.5*1./log(10.)*tau
        Av        <float>    extinction value (def: 1.0)
        Rv        <float> extinction param. (def: 3.1)
    """
    if type(lamb) == float:
        _lamb = np.asarray([lamb])
    else:
        _lamb = lamb[:]

    #init variables
    x = 1/(_lamb) #wavenumber in um^-1
    a = np.zeros(np.size(x))
    b = np.zeros(np.size(x))
    #Infrared (Eq 2a,2b)
    ind = np.where ((x >= 0.3) & (x < 1.1))
    a[ind] =  0.574*x[ind]**1.61
    b[ind] = -0.527*x[ind]**1.61
    #Optical & Near IR
    #Eq 3a, 3b
    ind = np.where ((x >= 1.1) & (x <= 3.3))
    y = x[ind]-1.82
    a[ind] = 1. + 0.17699*y   - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[ind] =      1.41338*y   + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    #UV
    #Eq 4a, 4b
    ind = np.where ((x >= 3.3) & (x <= 8.0))
    a[ind] =  1.752 - 0.316*x[ind] - 0.104/((x[ind]-4.67)**2+0.341)
    b[ind] = -3.090 + 1.825*x[ind] + 1.206/((x[ind]-4.62)**2+0.263)

    ind = np.where ((x >= 5.9) & (x <= 8.0))
    Fa     = -0.04473*(x[ind]-5.9)**2 - 0.009779*(x[ind]-5.9)**3
    Fb     =  0.21300*(x[ind]-5.9)**2 + 0.120700*(x[ind]-5.9)**3
    a[ind] = a[ind] + Fa
    b[ind] = b[ind] + Fb
    #Far UV
    #Eq 5a, 5b
    ind = np.where ((x >= 8.0) & (x <= 10.0))
    #Fa = Fb = 0
    a[ind] = -1.073 - 0.628*(x[ind]-8.) + 0.137*((x[ind]-8.)**2) - 0.070*(x[ind]-8.)**3
    b[ind] = 13.670 + 4.257*(x[ind]-8.) + 0.420*((x[ind]-8.)**2) + 0.374*(x[ind]-8.)**3

    # Case of -values x out of range [0.3,10.0]
    ind = np.where ((x > 10.0) | (x < 0.3))
    a[ind] = 0.0
    b[ind] = 0.0

    #Return Extinction vector
    #Eq 1
    if (Alambda == True):
        return ( 2.5*1./np.log(10.)*( a + b/Rv ) * Av)
    else:
        return ( ( a + b/Rv ) * Av)

def mag2flux(filter, mag, mag_err, dist, zp, corK = True):
    '''
    Transform the magnitude and error in magnitude into flux
    for any filter
    Parameters
    ----------------
    mag:     Must be in apparent magnitudes
    mag_err: Must be in apparent magnitudes
    Output
    ----------------
    Flux with the flux error
    '''
    # Finding the zeropoint of the reference in that filter
    fref=zp[filter]*1e-11
    fref = np.array(fref)
    # Correcting for redshift
    if corK: fref *= (1+z)

    flux = 4*np.pi*dist**2*fref*10**(-0.4*mag)
    flux_err = np.log(10)/2.5 * flux * mag_err
    
    return flux, flux_err

def flux2mag(filter, flux, flux_err, dist, zp, corK = True):
    '''
    Transform the flux and flux error into magnitude for any
    filter
    Parameters
    ----------------
    flux:     Flux
    flux_err: Error in Flux
    Output
    ----------------
    Apparent magnitude with associated error
    '''

    # Finding the zeropoint of the reference in that filter
    fref=zp[filter]*1e-11
    fref = np.array(fref)
    # Correcting for redshift
    if corK: fref *= (1+z)

    mag = -2.5*np.log10(np.abs(flux)/(fref*4*np.pi*dist**2))
    mag_err = flux_err/(np.log(10)*flux)*2.5
    
    return mag, mag_err

class Bbody(Model):
    parameter_names = ('Temp', 'Radius')

    def get_value(self, lam):
        '''
        Calculate the corresponding blackbody radiance for a set
        of wavelengths given a temperature and radiance.
        Parameters
        ---------------
        lam: Reference wavelengths in Angstroms
        T:   Temperature in Kelvin
        R:   Radius in cm
        Output
        ---------------
        Spectral radiance in units of erg/s/Angstrom
        (calculation and constants checked by Sebastian Gomez)
        '''
        lam = lam.flatten()

        # Planck Constant in cm^2 * g / s
        h = 6.62607E-27
        # Speed of light in cm/s
        c = 2.99792458E10

        # Convert wavelength to cm
        lam_cm = lam * 1E-8

        # Boltzmann Constant in cm^2 * g / s^2 / K
        k_B = 1.38064852E-16

        # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
        exponential = (h * c) / (lam_cm * k_B * self.Temp)
        B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)

        # Multiply by the surface area
        A = 4*np.pi*self.Radius**2

        # Output radiance in units of (erg / s) / Angstrom
        Radiance = B_lam * A / 1E8

        return Radiance

def easyint(x,y,err,xref,yref):
    '''
    Adapt scipy interpolation to include extrapolation for filters missing early/late data
    Originally based on `bolom.py` by Enrico Cappellaro (2008)
    Returns light curve mapped to reference epochs and errors on each point
    '''
    ir = (xref>=min(x))&(xref<=max(x))
    # for times where observed and reference band overlap, do simple interpolation
    yint = interp.interp1d(x[np.argsort(x)],y[np.argsort(x)])(xref[ir])
    yout = np.zeros(len(xref),dtype=float)
    # For times before or after observed filter has observations, use constant colour with reference band
    try:
        ylow = yint[np.argmin(xref[ir])]-yref[ir][np.argmin(xref[ir])]+yref[xref<min(x)]
    except:
        ylow = y[0]-yref[-1]+yref[xref<min(x)]
    try:
        yup  = yint[np.argmax(xref[ir])]-yref[ir][np.argmax(xref[ir])]+yref[xref>max(x)]
    except:
        yup  = y[-1]-yref[0]+yref[xref>max(x)]
    yout[ir] = yint
    yout[xref<min(x)] = ylow
    yout[xref>max(x)] = yup
    errout = np.zeros(len(xref),dtype=float)
    # put error floor of 0.1 mag on any interpolated data
    errout[ir] = max(np.mean(err),0.1)
    # for extrapolations, apply mean error for interpolated data, plus 0.01 mag per day of extrapolation (added in quadrature)
    errout[xref<min(x)] = np.sqrt((min(x) - xref[xref<min(x)])**2/1.e4 + np.mean(err)**2)
    errout[xref>max(x)] = np.sqrt((xref[xref>max(x)] - max(x))**2/1.e4 + np.mean(err)**2)

    return yout,errout

def cosmocalc(z):
    ################# cosmocalc by N. Wright ##################

    '''
    This was used in an older version of superbol, but can still
    be used in place of astropy if desired - just uncomment cosmocalc in step 3
    '''
    # initialize constants
    H0 = 70                         # Hubble constant
    WM = 0.27                        # Omega(matter)
    WV = 1.0 - WM - 0.4165/(H0*H0)  # Omega(vacuum) or lambda

    WR = 0.        # Omega(radiation)
    WK = 0.        # Omega curvaturve = 1-Omega(total)
    c = 299792.458 # velocity of light in km/sec
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    DTT = 0.0      # time from z to now in units of 1/H0
    DCMR = 0.0     # comoving radial distance in units of c/H0
    DA = 0.0       # angular size distance
    DL = 0.0       # luminosity distance
    DL_Mpc = 0.0
    a = 1.0        # 1/(1+z), the scale factor of the Universe
    az = 0.5       # 1/(1+z(object))

    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    n=1000         # number of points in integrals


    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)

    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n

    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio =  0.5*(np.exp(x)-np.exp(-x))/x
        else:
            ratio = np.sin(x)/x
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.

    DCMT = ratio*DCMR
    DA = az*DCMT

    DL = DA/(az*az)

    DL_Mpc = (c/H0)*DL

    return DL_Mpc

# Filter information

#SDSS filters and AB mags:
#These effective wavelengths for SDSS filters are from Fukugita et al. (1996, AJ, 111, 1748) and are
#the wavelength weighted averages (effective wavelengths in their Table 2a, first row)

#Effective wavelengths (in Angs)
wle = {'u': 3560,  'g': 4830, 'r': 6260, 'i': 7670, 'z': 8890, 'y': 9600, 'w':5985, 'Y': 9600,
       'U': 3600,  'B': 4380, 'V': 5450, 'R': 6410, 'G': 6730, 'I': 7980, 'J': 12200, 'H': 16300,
       'K': 21900, 'S': 2030, 'D': 2231, 'A': 2634, 'F': 1516, 'N': 2267, 'o': 6790, 'c': 5330,
       'W': 33526, 'Q': 46028
}
# For Swift UVOT: S=UVW2, D=UVM2, A=UVW1
# For GALEX: F=FUV, N=NUV
# For NEOWISE: W=W1, Q=W2


# The below zeropoints are needed to convert magnitudes to fluxes
#For AB mags,
#     m(AB) = -2.5 log(f_nu) - 48.60.
# f_nu is in units of ergs/s/cm2/Hz such that
#    m(AB) = 0 has a flux of f_nu = 3.63E-20 erg/s/cm2/Hz  = 3631 Jy
# Therefore, AB magnitudes are directly related to a physical flux.
# Working through the conversion to ergs/s/cm2/Angs, gives
# f_lam = 0.1089/(lambda_eff^2)  where lambda_eff is the effective wavelength of the filter in angstroms
# Note then that the AB flux zeropoint is defined ONLY by the choice of effective wavelength of the bandpass

# However, not all bands here are AB mag, so for consistency across all filters the zeropoints are stored in the following dictionary

# Matt originally listed the following from  Paul Martini's page : http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
# That is not an original source, for AB mags it simply uses the f_lam =0.1089/(lambda_eff^2) relation, and the effective wavelengths from Fukugita et al.

# ugriz and GALEX NUV/FUV are in AB mag system, UBVRI are Johnson-Cousins in Vega mag, JHK are Glass system Vega mags, Swift UVOT SDA and WISE WQ are in Vega mag system
#
#The values for UBVRIJHK are for the Johnson-Cousins-Glass system and are taken directly from Bessell et al. 1998, A&A, 333, 231 (Paul Martini's page lists these verbatim)
#Note that these Bessell et al. (1998) values were calculated not from the spectrum of Vega itself, but from a Kurucz model atmosphere of an AOV star.
#GALEX effective wavelengths from here: http://galex.stsci.edu/gr6/?page=faq
# WISE data from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=WISE&asttype=

# ATLAS values taken from Tonry et al 2018

#All values in 1e-11 erg/s/cm2/Angs
zp = {'u': 859.5, 'g': 466.9, 'r': 278.0, 'i': 185.2, 'z': 137.8, 'y': 118.2, 'w': 245.7, 'Y': 118.2,
      'U': 417.5, 'B': 632.0, 'V': 363.1, 'R': 217.7, 'G': 240.0, 'I': 112.6, 'J': 31.47, 'H': 11.38,
      'K': 3.961, 'S': 536.2, 'D': 463.7, 'A': 412.3, 'F': 4801., 'N': 2119., 'o': 236.2, 'c': 383.3,
      'W': 0.818, 'Q': 0.242}

#Filter widths (in Angs)
width = {'u': 458,  'g': 928, 'r': 812, 'i': 894,  'z': 1183, 'y': 628, 'w': 2560, 'Y': 628,
         'U': 485,  'B': 831, 'V': 827, 'R': 1389, 'G': 4203, 'I': 899, 'J': 1759, 'H': 2041,
         'K': 2800, 'S': 671, 'D': 446, 'A': 821,  'F': 268,  'N': 732, 'o': 2580, 'c': 2280,
         'W': 6626, 'Q': 10422}

#Extinction coefficients in A_lam / E(B-V). Uses York Extinction Solver (http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/YorkExtinctionSolver/coefficients.cgi)
extco = {'u': 4.786,  'g': 3.587, 'r': 2.471, 'i': 1.798,  'z': 1.403, 'y': 1.228, 'w':2.762, 'Y': 1.228,
         'U': 4.744,  'B': 4.016, 'V': 3.011, 'R': 2.386, 'G': 2.216, 'I': 1.684, 'J': 0.813, 'H': 0.516,
         'K': 0.337, 'S': 8.795, 'D': 9.270, 'A': 6.432,  'F': 8.054,  'N': 8.969, 'o': 2.185, 'c': 3.111,
         'W': 0.190, 'Q': 0.127}

# Colours for plots
cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k', 'y': '0.5', 'w': 'firebrick',
        'Y': '0.5', 'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'G': 'salmon',
        'I': 'chocolate', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
        'S': 'mediumorchid', 'D': 'purple', 'A': 'midnightblue',
        'F': 'hotpink', 'N': 'magenta', 'o': 'darkorange', 'c': 'cyan',
        'W': 'forestgreen', 'Q': 'peru'}

# Maintains order from blue to red effective wavelength
bandlist = 'FSDNAuUBgcVwrRoGiIzyYJHKWQ'

# Keeps track of possible kernels to use
av_kernels = {'RationalQuadraticKernel': kernels.RationalQuadraticKernel,
                'ExpSquaredKernel': kernels.ExpSquaredKernel,
                'Matern52Kernel': kernels.Matern52Kernel,
                'ExpKernel': kernels.ExpKernel,
                'Matern32Kernel': kernels.Matern32Kernel}

# First step is to search directory for existing mcnova files, or photometry files matching our naming conventions
print('\n######### Step 1: input files and filters ##########')

# keep tabs on whether interpolated LCs exist
useInt = 'n'

# SN name defines names of input and output files
sn = input('\n> Enter SN name:   ')

if not sn:
    print('\n* No name given; lets just call it `SN`...')
    sn = 'SN'

# Keep outputs in this directory
outdir = 'mcnova_output_'+sn
if not os.path.exists(outdir): os.makedirs(outdir)



# Get photometry files
do1 = input('\n> Find input files automatically?[y]   ')
if not do1: do1='y'
# User will almost always want to do this automatically, if files follow naming convention!

use1 = []

if do1 == 'y':
    # first check for previous mcnova interpolations
    files = glob.glob(outdir+'/interpolated_lcs_'+sn+'*.txt')
    if len(files)>0:
        print('\n* Interpolated LC(s) already available:')

        # If multiple interpolations exist, ask user which they want
        for i in range(len(files)):
            print('  ', i, ':', files[i])

        use = input('\n> Use interpolated LC? (e.g. 0,2 for files 0 and 2, or n for no) [0]\n (Warning: using multiple interpolation files can cause problems unless times match!)   ')
        # Default is to read in the first interpolation file
        # Multiple interpolations can be read using commas, BUT if time axes don't match then the phases can end up incorrectly defined for some bands!!!
        if not use: use1.append(0)

        if use!='n':
            # if previous interpolations are used, need to keep tabs so we don't interpolate again later!
            useInt = 'y'
            if len(use)>0:
                for i in use.split(','):
                    use1.append(i)
        else: print('\n* Not using interpolated data')


    if len(files)==0 or use=='n':
        # And here is if we don't have (or want) previously interpolated data
        # search for any files matching with SN name
        files = glob.glob(sn+'_*')

        if len(files)>0:
            # If files are found, print them and let the user choose which ones to read in
            print('\n* Available files:')

            for i in range(len(files)):
                print('  ', i, ':', files[i])

            use = input('\n> Specify files to use (e.g. 0,2 for files 0 and 2) [all]   ')
            if len(use)>0:
                # Include only specified files
                for i in use.split(','):
                    use1.append(i)
            else:
                # Or include all files
                for i in range(len(files)):
                    use1.append(i)

        else:
            # If no files found, keep track and print message
            do1 = 'n'
            print('* No files found for '+sn)


if do1 != 'y':
    # If we did not find any input data, you can specify files manually - BUT should still follow filter conventions and end in _<filters>.EXT
    files1 = input('\n> Enter all file names separated by commas:\n')
    if not files1:
        # But if no files specified by this point, we give up prompting!
        print('No files given - exiting!')
        sys.exit(0)

    files = []
    for i in files1.split(','):
        # If manually specified files given, add them to input list
        files.append(i)
    for i in range(len(files)):
        # Also need to keep an integer index for each file, so we can treat them the same as we would the automatically-detected files
        use1.append(i)


# This dictionary is vital, will hold all light curve data!
lc = {}

# This keeps track of filters used (don't remember why I used strings in place of lists...)
filts2 = str()

for i in use1:
    # These integers map to the list of input files
    i = int(i)
    # get filter from file name and add to list
    # filts1 keeps track of filters IN THAT FILE ONLY, filts2 is ALL filters across ALL files.
    filts1 = files[i].split('.')[0]
    filts1 = filts1.split('_')[-1].split('[')[0]
    filts2 += filts1
    
    # Here we read in the files using genfromtxt. Uses try statements to catch a few common variants of the input, e.g. with csv or header rows
    try:
        d = np.genfromtxt(files[i])
        x = 1
        for j in filts1:
            # loop over filters (j) in file and add each light curve to dictionary
            # column 0 is time, odd columns (x) are magnitudes, even columns (x+2) are errors
            lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
            x+=2
    except:
        try:
            d = np.genfromtxt(files[i],skip_header=1)
            x = 1
            for j in filts1:
                lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
                x+=2
        except:
            try:
                d= np.genfromtxt(files[i],delimiter=',')
                x = 1
                for j in filts1:
                    lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
                    x+=2
            except:
                try:
                    d= np.genfromtxt(files[i],delimiter=',',skip_header=1)
                    x = 1
                    for j in filts1:
                        lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
                        x+=2
                except:
                    raise ValueError('Could not read file')


# sort list of recognised filters from filts2 into wavelength order:
filters = str()
for i in bandlist:
    if i in filts2:
        filters += i

# If a filter name is not recognised, prompt user to add its properties manually
for i in filts2:
    if not i in wle:
        print('\n* Unknown filter '+i+'!')
        print('* Please enter details for filter',i)
        wle[i] = float(input(' >Lambda_eff (angstroms):   '))
        zp[i] = float(input(' >Flux zero point (1e11 erg/cm2/s/ang):   '))
        width[i] = float(input(' >Filter width (angstroms):   '))
        ftmp = str()
        cols[i] = 'grey'
        for j in filters:
            if wle[j]<wle[i]:
                ftmp += j
        ftmp += i
        for j in filters:
            if wle[j]>wle[i]:
                ftmp += j
        filters = ftmp

# This ends the data import

print('\n######### Step 2: reference band for phase info ##########')


plt.figure(1,(8,6))
plt.clf()

# Default time axis label
xlab = 'Time'

# Plot all light curves on same axes
for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

plt.gca().invert_yaxis()
plt.xlabel(xlab)
plt.ylabel('Magnitude')
plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
plt.tight_layout(pad=0.5)
plt.draw()

plt_show_interactive(pythonIDEInformed = False)

# Loop through dictionary and determine which filter has the most data
ref1 = 0
for i in filters:
    ref2 = len(lc[i])
    if ref2>ref1:
        ref1 = ref2
        ref3 = i



print('\n* Displaying all available photometry...')

# User can choose to include only a subset of filters, e.g. if they see that some don't have very useful data
t3 = input('\n> Enter bands to use (blue to red) ['+filters+']   ')
if not t3: t3 = filters

filters = t3

if len(filters) < 2:
    # If only one filter, no need to interpolate, and can't apply BB fits, so makes no sense to use mcnova!
    print('At least two filters required - exiting!')
    sys.exit(0)

# If using light curves that have not yet been interpolated by a previous mcnova run, we need a reference filter
if useInt!='y':
    ref = input('\n> Choose reference filter for sampling epochs\n   Suggested (most LC points): ['+ref3+']   ')
    # Defaults to the band with the most data
    if not ref: ref = ref3

# If light curves are already interpolated, reference is mainly for plotting so just pick first band
else: ref = filters[0]

print('\n* Using '+ref+'-band for reference')

# Input redshift or distance modulus, needed for flux -> luminosity
z = input('\n> Please enter SN redshift or distance modulus:[0]  ')
# Default to zero
if not z: z=0
z = float(z)

DL_Mpc = Distance(z = z).Mpc

# convert Mpc to cm, since flux in erg/s/cm2/A
dist = DL_Mpc*3.086e24

# User may want to have output in terms of days from maximum, so here we find max light in reference band
# Two options: fit light curve interactively, or just use brightest point. User specifies what they want to do
t1 = input('\n> Interactively find '+ref+'-band maximum?[n] ')
if not t1:
    # Default to not doing interactive fit
    t1 = 'n'

    # in this case check if user wants quick approximation
    doSh = input('\n> Shift to approx maximum?[n] ')
    # Default to not doing this either - i.e. leave light curve as it is
    if not doSh: doSh = 'n'

    if doSh=='y':
        # If approx shift wanted, find time of brightest point in ref band to set as t=0
        d = lc[ref]
        shift = d[:,0][np.argmin(d[:,1])]
        # Loop over all bands and shift them
        for j in lc:
            lc[j][:,0]-=shift

        # update x-axis label
        xlab += ' from approx '+ref+'-band maximum'

        print('\n* Approx shift done')

if t1!='n':
    # Here's where date of maximum is fit interactively, if user wanted it
    # Start with approx shift of reference band
    d = lc[ref]
    shift = d[:,0][np.argmin(d[:,1])]
    d[:,0]-=shift

    plt.clf()
    # Plot reference band centred roughly on brightest point
    plt.errorbar(d[:,0],d[:,1],d[:,2],fmt='o',color=cols[ref])

    plt.ylim(max(d[:,1])+0.2,min(d[:,1])-0.2)
    plt.xlabel(xlab + ' from approx maximum')
    plt.ylabel('Magnitude')
    plt.tight_layout(pad=0.5)
    plt.draw()
    
    plt_show_interactive()

    # As long as happy ='n', user can keep refitting til they get a good result
    happy = 'n'

    print('\n### Begin Gaussian process fit to peak... ###')
    
    # Only fit data at times < Xup from maximum light. Default is 50 days
    Xup1 = 50

    # Default value for the metric (this can be changed in future when I understand it more)
    metric1 = 500

    while happy == 'n':

        print('\n### Select data range ###')

        # Interactively set upper limit on times to fit
        Xup = input('>> Cut-off phase for GP fit?['+str(Xup1)+']   ')
        if not Xup: Xup = Xup1
        else:
            Xup = float(Xup)
        Xup1 = Xup

        d1 = d[d[:,0]<Xup]

        plt.clf()

        # Plot only times < Xup
        plt.errorbar(d1[:,0],d1[:,1],d1[:,2],fmt='o',color=cols[ref])

        plt.ylim(max(d1[:,1])+0.4,min(d1[:,1])-0.2)
        plt.tight_layout(pad=0.5)
        plt.draw()
        
        plt_show_interactive()        

        # Interactively choose a Kernal and set the metric (for stationary kernels)
        print('\n> Available kernels: ')
        print(av_kernels.keys())

        kernel = input('\n>> What type of kernel would you like to use?[Matern32Kernel]   ')
        metric = input('\n>> What value do you want to use for the metric?[500]   ')
        if not metric: metric = metric1
        metric = float(metric)
        if not kernel: kernel = av_kernels['Matern32Kernel'](metric)
        elif kernel == 'RationalQuadraticKernel':
            log_alpha = input('\n>> What value to use for the gamma distribution parameter (log(alpha))?[1]   ')
            if not log_alpha: log_alpha = 1
            log_alpha = float(log_alpha)

            kernel = av_kernels[kernel](log_alpha, metric)
        else:
            kernel = av_kernels[kernel](metric)

        # Convert magnitudes into flux and defining the kernel
        ref_flux, ref_flux_err = mag2flux(ref, d1[:,1], d1[:,2], dist=dist, zp = zp)
        kernel = np.var(ref_flux) * kernel

        # Computing the best fit from the gaussian process
        gp = george.GP(kernel)
        gp.compute(d1[:,0], ref_flux_err)

        # defining the shortened time range
        days = np.arange(min(-40,min(d1[:,0]))-10,Xup)
        pred, pred_var = gp.predict(ref_flux, days, return_var=True)

        ref_mag, ref_mag_err = flux2mag(ref, pred, np.sqrt(np.abs(pred_var)), dist = dist, zp = zp)

        plt.fill_between(days, ref_mag - ref_mag_err, ref_mag + ref_mag_err, color = 'k', alpha = 0.2)
        plt.plot(days, ref_mag, 'k', lw = 1.5, label = "GP", alpha = 0.5)

        plt.ylabel('Magnitude')
        plt.xlabel(xlab + ' from approx maximum')
        plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
        plt.xlim(min(d[:,0])-5,Xup)
        plt.tight_layout(pad=0.5)
        plt.draw()
        
        plt_show_interactive()        

        # Check if user likes fit
        happy = input('\n> Happy with fit?(y/[n])   ')
        # Default is to try again!
        if not happy: happy = 'n'
    
    # After user is satisfied with fit, check whether they want to use the peak of the most recent GP as t=0, or default to the brightest point
    new_peak = input('> Use [g-aussian process] or o-bserved peak date?    ')
    # Default is to use Gaussian process for peak date
    if not new_peak: new_peak = 'g'

    xlab += ' from '+ref+'-band maximum'

    # Plot reference band shifted to match GP peak
    if new_peak=='g':
        days = np.arange(d[:,0][np.argmin(d[:,1])]-10, d[:,0][np.argmin(d[:,1])]+10)
        pred, pred_var = gp.predict(d1[:,1], days, return_var=True)
        peak = days[np.argmin(pred)]
        d[:,0] -= peak
        plt.clf()
        plt.errorbar(d[:,0],d[:,1],d[:,2],fmt='o',color=cols[ref])
        plt.ylabel('Magnitude')
        plt.xlabel(xlab)
        plt.ylim(max(d[:,1])+0.2,min(d[:,1])-0.2)
        plt.tight_layout(pad=0.5)
        plt.draw()
        
        plt_show_interactive()

    # If user instead wants observed peak, that shift was already done!
    if new_peak == 'o':
        peak = 0

    # Shift all light curves by same amount as reference band
    for j in lc:
        lc[j][:,0]-=(shift+peak)

    # Need to un-shift the reference band, since it's now been shifted twice!
    lc[ref][:,0]+=(shift+peak)
        
plt.figure(1)
plt.clf()

# Re-plot the light curves after shifting
for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)



plt.gca().invert_yaxis()
plt.xlabel(xlab)
plt.ylabel('Magnitude')
plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
plt.tight_layout(pad=0.5)
plt.draw()

plt_show_interactive()

# Needed for K-correction step a bit later
skipK = 'n'


if z<10:
    # Redshift always less than 10, distance modulus always greater, so easy to distinguish
    print('Redshift entered (or DM=0)')

    t2 = ''

    # Check if user wants to correct time axis for cosmological time dilation
    if lc[ref][0,0]>25000 or useInt=='y':
        # If time is in MJD or input light curves were already interpolated, default to no
        t2 = input('\n> Correct for time-dilation?[n] ')
        if not t2: t2 = 'n'
    else:
        # Otherwise default to yes
        t2 = input('\n> Correct for time-dilation?[y] ')
        if not t2: t2 = 'y'

    if t2=='y':
        # Apply correction for time dilation
        for j in lc:
            lc[j][:,0]/=(1+z)
        print('\n* Displaying corrected phases')

        xlab += ' (rest-frame)'
        plt.xlabel(xlab)


    plt.figure(1)
    plt.clf()

    # Re-plot light curves in rest-frame times
    for i in filters:
        plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

    plt.gca().invert_yaxis()
    plt.xlabel(xlab)
    plt.ylabel('Magnitude')
    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
    plt.tight_layout(pad=0.5)
    plt.draw()

    plt_show_interactive()

    print('\n######### Step 3: Flux scale ##########')

    # New version uses astropy coordinates.Distance
    # Old version used cosmocalc (thanks to Sebastian Gomez for change)
    # Options for cosmologies
    # WMAP9, H0 = 69.3, Om0 = 0.286, Tcmb0 = 2.725, Neff = 3.04, m_nu = 0, Ob0 = 0.0463
    # And many others...
    # from astropy.cosmology import WMAP9
    # cosmology.set_current(WMAP9)
    DL_Mpc = Distance(z = z).Mpc

    # To use cosmocalc instead, uncomment below:
    # DL_Mpc = cosmocalc(z)

    #############################################

    # Check value of first light curve point to see if likely absolute or apparent mag
    print('\n* First '+ref+'-band mag = %.2f' %lc[ref][0,1])
    absol='n'
    if lc[ref][0,1] < 0:
        # If negative mag, must be absolute (but check!)
        absol = input('> Magnitudes are in Absolute mags, correct?[y] ')
        if not absol: absol='y'
    else:
        # If positive mag, must be apparent (but check!)
        absol = input('> Magnitudes are in Apparent mags, correct?[y] ')
        if not absol: absol ='n'

    if absol=='y':
        # If absolute mag, set distance to 10 parsecs
        DL_Mpc = 1e-5
        print('\n* Absolute mags; Luminosity distance = 10 pc')
    else:
        # Otherwise use luminosity distance from redshift
        print('\n* Luminosity distance = %.2e Mpc' %DL_Mpc)

    # convert Mpc to cm, since flux in erg/s/cm2/A
    dist = DL_Mpc*3.086e24

else:
    # If distance modulus entered, different approach needed!
    print('Distance modulus entered')

    # No k correction if no redshift!
    skipK = 'y'

    for i in lc:
        # Subtract distance modulus to convert to absolute mags (assuming no one would ever supply absolute mags and still enter a DM...)
        lc[i][:,1]-=z
        # Once absolute, distance = 10 pc
        dist = 1e-5*3.086e24

print('\n######### Step 4: Interpolate LCs to ref epochs ##########')

# If light curves are not already interpolated we will need to do that now
if useInt!='y':
    # Sort light curves by phase (incase this isn't already done)
    for i in lc:
        lc[i] = lc[i][lc[i][:,0].argsort()]

    # New dictionary for interpolated light curves
    lc_int = {}

    # Reference light curve is already 'interpolated' by definition
    lc_int[ref] = lc[ref]

    # User decides whether to fit each light curve interactively
    t4 = input('\n> Interpolate light curves interactively?[y]  ')
    # Default is yes
    if not t4: t4 = 'y'

    if t4=='y':
        print('\n### Begin gaussian process to fit... ###')

        # Keeo tab of the reference filter that is being used
        intKey = '\n# Reference band was '+ref

        for i in filters:
            # Need to loop for every band so that the fits can be interactively plotted
            if i!=ref:
                print('\n### '+i+'-band ###')

                # Default value for the metric (this can be changed in future when I understand it more)
                metric1 = 500   

                # Keep looking until happy
                happy = 'n'
                while happy == 'n':
                    #plot current band and reference band 
                    plt.clf()
                    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)
                    plt.errorbar(lc[ref][:,0],lc[ref][:,1],lc[ref][:,2],fmt='o',color=cols[ref],label=ref)
                    plt.gca().invert_yaxis()
                    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
                    plt.xlabel(xlab)
                    plt.ylabel('Magnitude')
                    plt.ylim(max(max(lc[ref][:,1]),max(lc[i][:,1]))+0.5,min(min(lc[ref][:,1]),min(lc[i][:,1]))-0.5)
                    plt.tight_layout(pad=0.5)
                    plt.draw()
                    plt.show()

                    plt_show_interactive()

                    # Interactively choose a Kernal and set the metric (for stationary kernels)
                    print('\n> Available kernels: ')
                    print(av_kernels.keys())

                    kernel = input('\n>> What type of kernel would you like to use?[Matern32Kernel]   ')
                    metric = input('\n>> What value do you want to use for the metric?[500](q to quit and use constant colour)   ')
                    
                    # If user decides that they cannot get a good fit 
                    if metric == 'q':
                        break
                    
                    # Go to default metric if only return is pressed
                    elif not metric: metric = metric1
                    metric = float(metric)

                    if not kernel: kernel = av_kernels['Matern32Kernel'](metric)
                    elif kernel == 'RationalQuadraticKernel':
                        # The RationalQuadraticKernel requires an addition input with the metric, the gamma distribution parameter
                        log_alpha = input('\n>> What value to use for the gamma distribution parameter (log(alpha))?[1]   ')
                        if not log_alpha: log_alpha = 1
                        log_alpha = float(log_alpha)

                        kernel = av_kernels[kernel](log_alpha, metric)
                    else:
                        kernel = av_kernels[kernel](metric)

                    # Converting from magnitudes to flux so that the GP fits correctly
                    tmp_flux, tmp_flux_err = mag2flux(i, lc[i][:,1], lc[i][:,2], dist=dist, zp = zp)
                    kernel = np.var(tmp_flux)*kernel
                    
                    # Using the choosen kernel to compute the fit of the data
                    gp = george.GP(kernel)
                    gp.compute(lc[i][:,0], tmp_flux_err)

                    if len(lc[i][:,0]) <= 3:
                        # In the case where a data file only has a couple of points the, the fit can sometimes not be placed in the right location
                        # This section here will use scipy minimise to find the optimal location of the fit, given the metric provided by the user
                        metric_name = gp.get_parameter_names()[1]
                        gp.freeze_parameter(metric_name)

                        def neg_ln_like(p):
                            gp.set_parameter_vector(p)
                            return -gp.log_likelihood(tmp_flux)

                        def grad_neg_ln_like(p):
                            gp.set_parameter_vector(p)
                            return -gp.grad_log_likelihood(tmp_flux)

                        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)

                        gp.set_parameter_vector(result.x)
                        gp.thaw_parameter(metric_name)

                    # Finding where to end the fit (either reference filter or current filter in question)
                    days = np.arange(min(lc[ref][:,0]), max(lc[ref][:,0]))

                    # Find the fit by gaussian process 
                    pred_flux, pred_flux_var = gp.predict(tmp_flux, days, return_var = True)
                    
                    # Converting back to magnitudes for plotting
                    pred_mag, pred_mag_err = flux2mag(i, pred_flux, np.sqrt(np.abs(pred_flux_var)), dist = dist, zp=zp)

                    mag_unc = plt.fill_between(days, pred_mag - pred_mag_err, pred_mag + pred_mag_err, color = 'k', alpha = 0.2)
                    plt.plot(days, pred_mag, 'k', lw = 1.5, label = "GP", alpha = 0.5)

                    plt.ylabel('Magnitude')
                    plt.xlabel(xlab)
                    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
                    plt.xlim(min(lc[ref][:,0])-5,max(lc[ref][:,0]+5))
                    plt.tight_layout(pad=0.5)
                    plt.draw()
                    
                    plt_show_interactive()
                    
                    change_all = False
                    change_outside = False

                    # Check if the user would like to restrict the GP uncertainty for time points where there is no data
                    change_unc = input('\n> Would you like to restrict the uncertainty for time steps where the GP error rapidly diverges?[n]  ')
                    if not change_unc: change_unc = 'n'

                    if change_unc == 'y':
                        # Subsection of happy so that they can manipulate the uncertainty without needed to do the whole plot again
                        hp = 'n'

                        while hp == 'n':
                            # Setting a boolen to keep track of which method of error reduction is used
                            change_all = False
                            change_outside = False

                            mag_unc.remove()                           
                            mag_unc = plt.fill_between(days, pred_mag - pred_mag_err, pred_mag + pred_mag_err, color = 'k', alpha = 0.2)

                            plt_show_interactive()

                            # resetting uncertainties until it is decided that these new uncertainties are worth keeping
                            new_mag_unc = np.array(pred_mag_err)
                            
                            # Asks the user to choose to restrict only outside of the time range for the filter or for any point in the filter where the uncertainty exceeds some value
                            print('\n> Would you like to restrict only time steps outside of the range for filter ' + str(i) + ' or for all time steps with large uncertainty?')
                            where_to_restrict = input('\n>> (all), (out) or (q) to quit: ')
                               
                            if where_to_restrict == 'all': 
                                change_all = True

                                # Ask the user how large an uncertainty from the gaussian process to accept
                                max_unc_allowed = input('\n> What is the maximum gaussian process uncertainty to keep (as a fraction of the gp fit)?  ')
                                max_unc_allowed = float(max_unc_allowed)
                                
                                # Ask the user what to set the uncertainty to, all following time points will have the same uncertainty 
                                percent_from_fit = input('\n> What fraction of gp fit for uncertainty? (0 -> 1)  ')
                                percent_from_fit = float(percent_from_fit)

                                # Find the time points where the max allowed uncertainty is allowed and at these points set the uncertainty to the user quoted amount
                                index = np.where(new_mag_unc > max_unc_allowed*pred_mag)
                                new_mag_unc[index] = (percent_from_fit * pred_mag[index])

                                # Clear old uncertainty
                                mag_unc.remove()
                                # Fill in the uncertainty on the plots
                                mag_unc = plt.fill_between(days, pred_mag - new_mag_unc, pred_mag + new_mag_unc, color = 'k', alpha = 0.2)

                                plt_show_interactive()

                                hp = input('\n> Happy with uncertainties?(y/[n])  ')
                                if not hp: hp = 'n'

                            elif where_to_restrict == 'out':
                                change_outside = True

                                # Ask the user how many days before the first time point and after the last time point at which to begin applying the user defined max uncertainty
                                days_after = input('\n> How many days after to start restricting the unc?[0]  ')
                                days_before = input('\n> How many days before to start restricting the unc?[0]  ')
                                if not days_after: days_after = 0
                                if not days_before: days_before = 0
                                days_after = float(days_after)
                                days_before = float(days_before)

                                # Find the time points where the max allowed uncertainty is allowed
                                index_high = np.where(days > max(lc[i][:,0])+days_after)
                                index_low = np.where(days < min(lc[i][:,0])-days_before)
                                
                                '''
                                # Find what the error value is at this point, as a fraction of the best fit line
                                high_err = pred_mag[index_high[0]]/new_mag_unc[index_high[0]]
                                low_err = pred_mag[index_low[0]]/new_mag_unc[index_low[0]]

                                # Inform the user of these fractions
                                print(str(days_after) + ' days after the last time point, the uncertainty is ' + str(high_err))
                                print(str(days_before) + ' days before the first time point, the uncertainty is '+ str(low_err))
                                '''

                                # Ask the user what to set the uncertainty to, all following time points will have the same uncertainty 
                                percent_from_fit = input('\n> What fraction of gp fit for uncertainty? (0 -> 1)  ')
                                percent_from_fit = float(percent_from_fit)
                                
                                # Find the time points in the after and before time regime and set the uncertainty to the user defined amount
                                new_mag_unc[index_high] = percent_from_fit*pred_mag[index_high]
                                new_mag_unc[index_low] = percent_from_fit*pred_mag[index_low]
                                
                                # Clear old uncertainty
                                mag_unc.remove()
                                # Fill in the uncertainty on the plots
                                mag_unc = plt.fill_between(days, pred_mag - new_mag_unc, pred_mag + new_mag_unc, color = 'k', alpha = 0.2)

                                plt_show_interactive()

                                hp = input('\n> Happy with uncertainties?(y/[n])  ')
                                if not hp: hp = 'n'
                            
                            elif where_to_restrict == 'q':                                
                                mag_unc.remove()
                                mag_unc = plt.fill_between(days, pred_mag - pred_mag_err, pred_mag + pred_mag_err, color = 'k', alpha = 0.2)
                                hp = 'y'
                    
                    plt_show_interactive()
                    # Check if happy with fit
                    happy = input('\n> Happy with fit?(y/[n])   ')
                    # Default is no
                    if not happy: happy = 'n'


                if metric == 'q':
                    # Breaks if no overlap in time with ref band
                    tmp1,tmp2 = easyint(lc[i][:,0],lc[i][:,1],lc[i][:,2],lc[ref][:,0],lc[ref][:,1])
                    tmp = list(zip(lc[ref][:,0],tmp1,tmp2))
                    lc_int[i] = np.array(tmp)
                    print('\n* Interpolating linearly; extrapolating assuming constant colour...')
                    # Add method to output
                    intKey += '\n# '+i+': Linear interp; extrap=c'
                else:
                    # If user was happy with fit, add different interpolation string to output
                    intKey += '\n# '+i+'GP using metric='+str(metric)+'; extrap method '

                    # ConstructGP interpolation
                    pred_flux_at_ref, pred_flux_var_at_ref = gp.predict(tmp_flux, lc[ref][:,0], return_var=True)
                    pred_mag_at_ref, pred_mag_err_at_ref = flux2mag(i, pred_flux_at_ref, np.sqrt(pred_flux_var_at_ref), dist=dist, zp=zp)

                    # Restricting the errors the same way the previous interpolations were restricted
                    new_ref_mag_unc = np.array(pred_mag_err_at_ref)
                    if change_all:
                        index = np.where(new_ref_mag_unc > max_unc_allowed*pred_mag_at_ref)
                        new_ref_mag_unc[index] = (percent_from_fit * pred_mag_at_ref[index])

                    
                    elif change_outside:
                        # Find the time points where the max allowed uncertainty is allowed
                        index_high = np.where(lc[ref][:,0] > max(lc[i][:,0])+days_after)
                        index_low = np.where(lc[ref][:,0] < min(lc[i][:,0])-days_before)
                        
                        # Find the time points in the after and before time regime and set the uncertainty to the user defined amount
                        new_ref_mag_unc[index_high] = percent_from_fit*pred_mag_at_ref[index_high]
                        new_ref_mag_unc[index_low] = percent_from_fit*pred_mag_at_ref[index_low]
                        
                        
                    # Goal: if band has a point at same epoch as ref banc, use point, otherwise, use GP prediction with uncertainties

                    mag_int = []

                    for k in lc[ref]:
                        # Check each light curve point against each reference time
                        # If match, add that point to the interpolated curve
                        k1 = np.where(lc[i][:,0]==k[0])
                        if len(k1[0])>0:
                            mag_int.append(lc[i][k1][0])

                    # Convert matches to numpy array (just to compare with reference array)
                    tmp_arr = np.array(mag_int)
                    
                    if tmp_arr.size:
                        print('Found matches')
                        # Do this loop if there were some temporal matches between current and reference band
                        for k in lc[ref]:
                            # Iterate over each reference time
                            if k[0] not in tmp_arr[:,0]:
                                # If no match in current band, calculate magnitude from GP fit
                                # Finds the magnitude from the GP prediction at that time
                                mag = pred_mag_at_ref[np.where(k[0]==lc[ref][:,0])]

                                # Finding the uncertainty in the magnitude at that time
                                mag_unc = new_ref_mag_unc[np.where(k[0]==lc[ref][:,0])]

                                
                                # Appending the GP magnitude to the light curve, with the uncertainty
                                out = np.array([k[0],mag[0],mag_unc[0]])
                                mag_int.append(out)
                    else:
                        print('No matches')
                        # If none of the points match then the extrapolation occurs solely via the GP defined prior
                        mag_int = np.array([lc[ref][:,0], pred_mag_at_ref, new_ref_mag_unc])
                        mag_int = mag_int.transpose()

                    # Convert full interpolated light curve to np array
                    mag_int = np.array(mag_int)
                
                    # Sort chronologically
                    tmp = mag_int[np.argsort(mag_int[:,0])]

                    # Here we will also do the extrapolation via constant colour
                    # The user is then able to check which they prefer to use

                    # Earliest time in band (extrapolating if necessary to get some overlap with ref band)
                    low = min(lc[i][:,0])
                    low = min(low,max(tmp[:,0]))
                    # Latest time in band
                    up = max(lc[i][:,0])
                    up = max(up,min(tmp[:,0]))

                    # Colour wrt reference band at earliest and latest interpolated epochs
                    col1 = tmp[tmp[:,0]>=low][0,1] - lc[ref][tmp[:,0]>=low][0,1]
                    col2 = tmp[tmp[:,0]<=up][-1,1] - lc[ref][tmp[:,0]<=up][-1,1]
                    # Get extrapolated points in current band by adding colour to reference band
                    early = lc[ref][tmp[:,0]<low][:,1]+col1
                    late = lc[ref][tmp[:,0]>up][:,1]+col2

                    # Plot light curve from GP fit
                    plt.errorbar(tmp[:,0],tmp[:,1],fmt='s',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i],label='Gaussian process')
                    # Plot constant colour extrapolation
                    plt.errorbar(tmp[tmp[:,0]<low][:,0],early,fmt='o',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i],label='Constant colour')
                    plt.errorbar(tmp[tmp[:,0]>up][:,0],late,fmt='o',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i])
                    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
                    plt.tight_layout(pad=0.5)
                    plt.draw()

                    plt_show_interactive()
                    
                    if len(tmp[tmp[:,0]<low])>0:
                        # If there are early extrapolated points, ask user whether they prefer polynomial, constant colour, or want to hedge their bets
                        extraptype = input('\n> Early-time extrapolation:\n  [g-aussian process], c-onstant colour, or a-verage of two methods?\n')
                        # Default to polynomial
                        if not extraptype: extraptype = 'g'
                        if extraptype == 'c':
                            # Constant colour
                            tmp[:,1][tmp[:,0]<low]=early

                            # Creating some uncertainties to go along with the constant colour 
                            # Check what the next point with known uncertainty and set that as the error for the constant colour points
                            next_unc = tmp[:,2][tmp[:,0]>low][0]
                            tmp[:,2][tmp[:,0]<low] = next_unc
                        if extraptype == 'a':
                            # Average
                            tmp[:,1][tmp[:,0]<low]=0.5*(tmp[:,1][tmp[:,0]<low]+early)
                    # If no need to extrapolate:
                    else: extraptype = 'n'

                    # Keep tabs on which extrapolation method was used!
                    intKey += 'early='+extraptype+';'

                    # Now do same for late times
                    if len(tmp[tmp[:,0]>up])>0:
                        extraptype = input('\n> Late-time extrapolation:\n  [g-aussian process], c-onstant colour, or a-verage of two methods?\n')
                        if not extraptype: extraptype = 'g'
                        if extraptype == 'c':
                            tmp[:,1][tmp[:,0]>up]=late
                            
                            # Creating some uncertainties to go along with the constant colour 
                            # Check what the next point with known uncertainty and set that as the error for the constant colour points
                            next_unc = tmp[:,2][tmp[:,0]<up][0]
                            tmp[:,2][tmp[:,0]>up] = next_unc
                        if extraptype == 'a':
                            tmp[:,1][tmp[:,0]>up]=0.5*(tmp[:,1][tmp[:,0]>up]+late)
                    else: extraptype = 'n'

                    intKey += 'late='+extraptype

                    # Add the final interpolated and extrapolated light curve to the dictionary
                    lc_int[i] = tmp

        # Key for output file
        intKey += '\n# g = gaussian process, c = constant colour, a = average'

    # If user does not want to do interpolation interactively:
    else:
        for i in filters:
            # For every band except reference, use easyint for linear interpolation between points, and constant colour extrapolation
            if i!=ref:
                tmp1,tmp2 = easyint(lc[i][:,0],lc[i][:,1],lc[i][:,2],lc[ref][:,0],lc[ref][:,1])
                tmp = list(zip(lc[ref][:,0],tmp1,tmp2))
                lc_int[i] = np.array(tmp)
        print('\n* Interpolating linearly; extrapolating assuming constant colour...')

        intKey = '\n# All light curves linearly interpolated\n# Extrapolation done by assuming constant colour with reference band ('+ref+')'

    # Need to save interpolated light curves for future re-runs
    int_out = np.empty([len(lc[ref][:,0]),1+2*len(filters)])
    # Start with reference times
    int_out[:,0] = lc[ref][:,0]

    for i in range(len(filters)):
        # Append magnitudes and errors, in order from bluest to reddest bands
        int_out[:,2*i+1] = lc_int[filters[i]][:,1]
        int_out[:,2*i+2] = lc_int[filters[i]][:,2]

    # Open file in mcnova output directory to write light curves
    int_file = open(outdir+'/interpolated_lcs_'+sn+'_'+filters+'.txt','wb')

    # Construct header
    cap = '#phase\t'
    for i in filters:
        # Add a column heading for each filter
        cap = cap+'\t'+i+'\terr'
    cap +='\n'

    # Save to file, including header and footer containing log of interpolation methods
    np.savetxt(int_file,int_out,fmt='%.2f',delimiter='\t',header=cap,footer=intKey,comments='#')
    # Close output file
    int_file.close()

    # Plot interpolated lcs
    print('\n* Displaying all interpolated/extrapolated LCs')
    plt.figure(1)
    plt.clf()
    for i in filters:
        plt.errorbar(lc_int[i][:,0],lc_int[i][:,1],lc_int[i][:,2],fmt='o',color=cols[i],label=i)
    plt.gca().invert_yaxis()
    plt.xlabel(xlab)
    plt.ylabel('Magnitude')
    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
    # plt.ylim(max(max(lc_int[ref][:,1]),max(lc_int[i][:,1]))+0.5,min(min(lc_int[ref][:,1]),min(lc_int[i][:,1]))-0.5)
    plt.tight_layout(pad=0.5)
    plt.draw()

    plt_show_interactive()

# Or if light curves were already interpolated, no need for the last 250 lines!
else:
    print('\n* Interpolation already done, skipping step 4!')

    # Put pre-interpolated lcs into dictionary
    lc_int = {}
    for i in filters:
        lc_int[i] = lc[i]


print('\n######### Step 5: Extinction and K-corrections #########')

# Extinction correction
ebv = input('\n> Please enter Galactic reddening E(B-V): \n'
                        '  (0 if data are already extinction-corrected) [0]   ')
if not ebv: ebv=0
ebv = float(ebv)

# Using the Cardelli extinction law to find the reddening amount
wavelengths = list(wle.values())
wavelengths = np.array(wavelengths, dtype = float)
wavelengths *= 0.0001

extinction_coeff = list(extco.values())
extinction_coeff = np.array(extinction_coeff, dtype = float)
redvec = cardelli(wavelengths, Av=3.1*ebv)


for i in lc_int:
    # Using the Cardelli extinction law to find the reddening amount
    wavelength = float(wle[i])*0.0001
    extinction_coeff = float(extco[i])
    redvec = cardelli(wavelength, Av=extinction_coeff*ebv, Rv=extinction_coeff)
    
    # Adjusting the magnitude for the reddening affect
    lc_int[i][:,1] = lc_int[i][:,1] - 2.5*np.log10(np.exp(-redvec))

# If UVOT bands are in AB, need to convert to Vega
if 'S' in lc_int or 'D' in lc_int or 'A' in lc_int:
    shiftSwift = input('\n> UVOT bands detected. These must be in Vega mags.\n'
                            '  Apply AB->Vega correction for these bands? [n]   ')
    if not shiftSwift: shiftSwift = 'n'

    if shiftSwift == 'y':
        if 'S' in lc_int:
            lc_int['S'][:,1] -= 1.51
        if 'D' in lc_int:
            lc_int['D'][:,1] -= 1.69
        if 'A' in lc_int:
            lc_int['A'][:,1] -= 1.73

# Whether to apply approximate K correction
doKcorr = 'n'
# i.e. if we have a redshift:
if skipK == 'n':
    # converting to rest-frame means wavelength /= 1+z and flux *= 1+z. But if input magnitudes were K-corrected, this has already been done implicitly!
    doKcorr = input('\n> Do you want to covert flux and wavelength to rest-frame?\n'
                            '  (skip this step if data are already K-corrected) [n]   ')


######### Now comes the main course - time to build SEDs and integrate luminosity

# Build list of wavelengths
wlref = []
# First wavelength is roughly blue edge of bluest band (effective wavelength + half the width)
wlref1 = [wle[filters[0]]-width[filters[0]]/2]
# wlref contains band centres only (for BB fit), whereas wlref1 also has outer band edges (for numerical integration)

# List of flux zeropoints matching wavelengths
fref = []

# List of widths for each band (needed for error estimates)
bandwidths = []

# Loop over used filters and populate lists from dictionaries of band properties
for i in filters:
    wlref.append(float(wle[i]))
    fref.append(zp[i]*1e-11)
    wlref1.append(float(wle[i]))
    bandwidths.append(float(width[i]))

# Final reference wavelength is red edge of reddest band
wlref1.append(wle[filters[-1]]+width[filters[-1]]/2)
# Flux will be set to zero at red and blue extrema of SED when integrating pseudobolometric light curve

# Make everything a numpy array
wlref1 = np.array(wlref1)
wlref = np.array(wlref)
fref = np.array(fref)
bandwidths = np.array(bandwidths)

# Get phases with photometry to loop over
phase = lc_int[ref][:,0]

# Correct flux and wavelength to rest-frame, if user chose that option earlier
if doKcorr == 'y':
    wlref /= (1+z)
    wlref1 /= (1+z)
    fref *= (1+z)
    bandwidths /= (1+z)

# construct some notes for output file
method = '\n# Methodology:'
method += '\n# filters used:'+filters
method += '\n# redshift used:'+str(z)
method += '\n# extinction used:'+str(ebv)

if doKcorr == 'y':
    method += '\n# Flux and wavelength converted to rest-frame'
else:
    method += '\n# Wavelengths used in observer frame (data already K-corrected?)'



print('\n######### Step 6: Fit blackbodies and integrate flux #########')

# These are needed to scale and offset SEDs when plotting, to help visibility
k = 1
fscale, _ = mag2flux(ref, min(lc[ref][:,1]), min(lc[ref][:,2]), dist=dist, zp=zp)
fscale *= len(phase)/10
# These lists will be populated with luminosities as we loop through the data and integrate SEDs
L1arr = []
L2arr = []
L1err_arr = []
L2err_arr = []
Lbb_full_arr = []
Lbb_full_err_arr = []
Lbb_opt_arr = []
Lbb_opt_err_arr = []

# Set up some parameters for the BB fits and integrations:
# First, if there are sufficient UV data, best to fit UV and optical separately
# Optical fit gives better colour temperature by excluding line-blanketed region
# UV fit used only for extrapolating bluewards of bluest band
sep = 'n'
# If multiple UV filters
if len(wlref[wlref<3000])>2:
    # Prompt for separate fits
    sep = input('\n> Multiple UV filters detected! Fitting optical and UV separately can\n give better estimates of continuum temperature and UV flux\n Fit separately? [y] ')
    # Default is yes
    if not sep: sep = 'y'
else:
    # Cannot do separate UV fit if no UV data!
    sep = 'n'

# If no UV data or user chooses not to do separate fit, allow for suppression in blue relative to BB
# -  If UV data, suppress to the blue of the bluest band
# -  If no UV data, start suppression at 3000A
# Functional form comes from Nicholl, Guillochon & Berger 2017 / Yan et al 2018:
# - power law in (lambda / lambda_cutoff) joins smoothly to BB at lambda_cutoff
bluecut = 1
# These default parameters give an unattenuated blackbody
sup = 0
if sep == 'n':
    # cutoff wavelength is either the bluest band (if data constrain SED below 3000A), or else fixed at 3000A (where deviation from BB usually starts becoming clear)
    bluecut = float(min(wlref[0],3000))
    # User specifies degree of suppression - higher polynomial order takes flux to zero faster. Value of x~1 is recommended for most cases
    sup = input('\n> Suppression index for BB flux bluewards of '+str(bluecut)+'A?\n  i.e. L_uv(lam) = L_bb(lam)*(lam/'+str(bluecut)+')^x\n [x=0 (i.e. no suppression)] ')
    # Default is no suppression
    if not sup: sup = 0
    sup = float(sup)


def blackbody(lam, T, R, lambda_cutoff=bluecut, alpha=sup):
        '''
        Calculate the corresponding blackbody radiance for a set
        of wavelengths given a temperature and radiance.
        Parameters
        ---------------
        lam: Reference wavelengths in Angstroms
        T:   Temperature in Kelvin
        R:   Radius in cm
        Output
        ---------------
        Spectral radiance in units of erg/s/Angstrom
        (calculation and constants checked by Sebastian Gomez)
        '''

        # Planck Constant in cm^2 * g / s
        h = 6.62607E-27
        # Speed of light in cm/s
        c = 2.99792458E10

        # Convert wavelength to cm
        lam_cm = lam * 1E-8

        # Boltzmann Constant in cm^2 * g / s^2 / K
        k_B = 1.38064852E-16

        # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
        exponential = (h * c) / (lam_cm * k_B * T)        
        B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
        B_lam[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha

        # Multiply by the surface area
        A = 4*np.pi*R**2

        # Output radiance in units of (erg / s) / Angstrom
        Radiance = B_lam * A / 1E8

        return Radiance

# Open output files for bolometric light curve and blackbody parameters
out1 = open(outdir+'/bol_'+sn+'_'+filters+'.txt','w')
out2 = open(outdir+'/BB_params_'+sn+'_'+filters+'.txt','w')

# Write header for bol file
out1.write('# ph\tLobs\terr\tL+BB\terr\t\n\n')

# Write header for BB params file - if separate UV/optical fits, need another set of columns for the optical-only filts
# T_bb etc are fits to all data, T_opt are fits to data at lambda>3000A (i.e. not affected by line blanketing)
if sep=='y':
    out2.write('# ph\tT_bb\terr\tR_bb\terr\tL_bb\terr\tT_opt\terr\tR_opt\terr\tL_opt\terr\n\n')
else:
    out2.write('# ph\tT_bb\terr\tR_bb\terr\tL_bb\terr\n\n')

# Display various lines for different fitting assumptions, tell user here rather than cluttering figure legend
print('\n*** Fitting Blackbodies to SED ***')
print('\n* Solid line = blackbody fit for flux extrapolation')

if sep=='y':
    # show separate fits to UV and optical, if they exist, and tell output file
    print('* Dashed lines = separate fit to optical and UV for T and R estimates')
    method += '\n# Separate BB fits above/below 3000A'

if sup!=0:
    # plot suppression if used, and tell output file where suppression began and what was the index
    print('* Dotted lines = UV flux with assumed blanketing')
    method += '\n# BB fit below '+str(bluecut)+'A suppressed by factor (lamda/'+str(bluecut)+')^'+str(sup)

if sep!='y' and sup==0:
    # if a single un-suppressed BB was used, add this to output file
    method += '\n# Single BB fit to all wavelengths, with no UV suppression'

# Plotting to see the affect of reddening and UV filtering
plt.clf()
for i in filters:
    plt.errorbar(lc_int[i][:,0],lc_int[i][:,1],lc_int[i][:,2],fmt='o',color=cols[i],label=i)
plt.gca().invert_yaxis()
plt.xlabel(xlab)
plt.ylabel('Magnitude')
plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
# plt.ylim(max(max(lc_int[ref][:,1]),max(lc_int[i][:,1]))+0.5,min(min(lc_int[ref][:,1]),min(lc_int[i][:,1]))-0.5)
plt.tight_layout(pad=0.5)
plt.draw()

plt_show_interactive()

# New figure to display SEDs
plt.figure(2,(8,8))
plt.clf()

# Letting the User decide if they want to consider the covariance between data points in modelling the blackbody
inc_noise = input('\n> Would you like to consider the covariance between data points when modelling the blackbody \n (Warning! this can be quite computationally expensive)[n]   ')
if not inc_noise: inc_noise='n'

# Asking the user if they want to change the number of burn-ins and mcmc steps at each timestep
ask_at_each_step = input('\n> Would you like to choose the number of burn-ins and mcmc steps at every timestep?[n]    ')
if not ask_at_each_step: 
    ask_at_each_step = 'n'

    # Setting the number of burn-ins and mcmc steps
    burnin_num = input('\n> Choose the burnin number for all time steps.[500]    ')
    if not burnin_num: burnin_num = 500
    burnin_num = int(burnin_num)
    
    num_steps = input('\n> Choose the number of MCMC steps for all time steps.[10000]    ')
    if not num_steps: num_steps = 10000
    num_steps = int(num_steps)

    # Asking the user how many walkers and MCMC steps to perform
    nwalkers = input('\n> How many walkers do you want for the MCMC?[32]    ')
    if not nwalkers: nwalkers = 32
    nwalkers = int(nwalkers)

    percent_unc = input ('\n> What percentile would you like to use for the uncertainy in temperature and radius?[75 percentile]     ')
    if not percent_unc: percent_unc = 75
    percent_unc = float(percent_unc)
    

corner_dir = outdir+'/Corner_plots_'+sn
if not os.path.exists(corner_dir): os.makedirs(corner_dir)

curve_fit_temp = 10000
curve_fit_rad = 1e15

total_number = len(phase)

print('\n################################################\nHere we are defining the temperature and radius parameter space for each epoch. \n\nFor example if you choose the default 3000K the temperature parameter space will be set up as T-3000 --> T+3000 where T is the temperature found for the previous epoch.')
temp_param_space = input('\n> Set an initial upper/lower limit for the temperature (K) parameter space:[3000]    ')
if not temp_param_space: temp_param_space = 3000
temp_param_space = float(temp_param_space)

print('\nThe radius parameter space is set up as a fraction of the previous radius. For example if you input the default 0.8, the parameter space will be R-0.8R --> R+0.8R where R is the radius found at the previous epoch.')
rad_param_space = input('\n> Set an initial limit (as a factor of initial guess) for the radius parameter space:[0.8]    ')
if not rad_param_space: rad_param_space = 0.8
rad_param_space = float(rad_param_space)

curve_fit_temp = input('\nInitial guess for starting temperature:[10000]  ')
if not curve_fit_temp: curve_fit_temp=10000
curve_fit_temp = float(curve_fit_temp)

curve_fit_rad = input('\nInitial guess for starting raidus:[1.0e15cm]  ')
if not curve_fit_rad: curve_fit_rad=1e15
curve_fit_rad = float(curve_fit_rad)

# Looping through reference epochs
for i in range(len(phase)):
    step_factor = 0.00001

    # Get date
    ph = phase[i]
    # Get list of mags and errors in all filters at epoch - start with blank arrays to add all filters
    mags = np.zeros(len(filters))
    errs = np.zeros(len(filters))
    #for j in range(len(filters)):


    for j in range(len(filters)): 
        # Loop through filters and populate the SED tables with interpolated light curves
        mags[j] = lc_int[filters[j]][i,1]
        errs[j] = lc_int[filters[j]][i,2]
    
    # Convert magnitudes to physical fluxes using zeropoints and distance
    flux = 4*np.pi*dist**2*fref*10**(-0.4*mags)
    # Convert mag errors to flux errors
    ferr = 2.5/np.log(10)*flux*errs

    # Set flux to zero at red and blue extrema matching wlref1
    flux1 = np.insert(flux,0,0)
    flux1 = np.append(flux1,0)

    # Sometimes the curve fit is unable to find optimal parameters when considering a sigma so if it fails then try without sigma
    try:
        BBparams, covar = curve_fit(blackbody,wlref,flux,p0=(curve_fit_temp,curve_fit_rad), sigma = ferr, bounds = [[0, 1e13], [25000, 1e20]])
    except:
        BBparams, covar = curve_fit(blackbody,wlref,flux,p0=(curve_fit_temp,curve_fit_rad), bounds = [[0, 1e13], [25000, 1e20]])


    ini_temp = BBparams[0]
    ini_rad = np.abs(BBparams[1])
    
    print('\n\nCurvefit used to find best initial guess for the black body fit:')
    print('Best Initial temperature guess: {} K'.format(float(ini_temp)))
    print('Best Initial radius guess: {:.3g} cm'.format(float(ini_rad)))

    if inc_noise == 'n':            
        # Model the blackbody at each time step without considering the correlated noise between the data points
        # Defining the truth array
        truth = dict(Temp = ini_temp, Radius = ini_rad)

        # Creating the george model and computing the fit
        BBmodel = george.GP(mean=Bbody(Temp=ini_temp, Radius=ini_rad), fit_mean=True)
        BBmodel.compute(wlref, yerr = ferr)
        
        # Defining the log likelihood and log prior
        def lnprob(p):
            # Uniform prior across the two parameters
            if (ini_temp-temp_param_space>p[0] or p[0]>ini_temp+temp_param_space):
                return -np.inf
            elif (ini_rad-ini_rad*rad_param_space>p[1] or p[1]>ini_rad+ini_rad*rad_param_space):
                return -np.inf
            BBmodel.set_parameter_vector(p)
            return BBmodel.log_likelihood(flux, quiet=True)


        if ask_at_each_step == 'y':
            # User input to how many burn-in's to perform and how many steps to run the mcmc for 
            burnin_num = input('\n> How many steps do you want to burn?[500]')
            if not burnin_num: burnin_num = 500
            burnin_num = int(burnin_num)
            num_steps = input('\n> How many steps do you want to run?[10000]')
            if not num_steps: num_steps = 10000
            num_steps = int(num_steps)
        
            # Asking the user how many walkers and MCMC steps to perform
            nwalkers = input('\n> How many walkers do you want for the MCMC?[32]    ')
            if not nwalkers: nwalkers = 32
            nwalkers = int(nwalkers)

        # Calling the initial parameters and defining the number of dimensions by the two parameters in the BB model
        initial = BBmodel.get_parameter_vector()
        ndim = len(initial)        

        p0 = initial + step_factor*initial*np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
                
        print("Running 1st burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, burnin_num)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0, num_steps)
        
        # Printing the loglikelihood
        log_like = BBmodel.log_likelihood(flux)
        print('Log Likelihood: ' + str(log_like))

        if abs(log_like) > 5000:
            print("\nThis time step has a very large log-likelihood, the plot is produced and the user decides whether to keep it or not.")
            
            plt.figure(5)
            plt.errorbar(wlref,flux-fscale*k,yerr = ferr,fmt = 'o',color=cols[filters[k%len(filters)]],label='%.1f' %ph)
        
            plt.plot(np.arange(100,25000), BBmodel.mean.get_value(np.arange(100,25000))-fscale*k, color=cols[filters[k%len(filters)]],linestyle='-')
            plt.plot(np.arange(100,bluecut), BBmodel.mean.get_value(np.arange(100,bluecut))*(np.arange(100,bluecut)/bluecut)**sup-fscale*k, color=cols[filters[k%len(filters)]],linestyle=':')
            plt_show_interactive()

            keep = input("\n>Would you like to keep this fit in the final plot of all time stamps?\nWarning! if the fit shown does not seem to accurately represent the points it is recommended to leave it out of the final plot.\nThe results will still be kept in the txt file. ('y', 'n')['n']    ")
            if not keep: keep = 'n'
            plt.close(5)

        else:
             keep = 'y'

        # Plotting the data and the fits for this time step
        plt.figure(2)
        plt.errorbar(wlref,flux-fscale*k,yerr = ferr,fmt = 'o',color=cols[filters[k%len(filters)]],label='%.1f' %ph)
        
        if keep == 'y':
            # Only plot the fits if the log likelihood is reasonable or the user decides that they want to keep it
            plt.plot(np.arange(100,25000), BBmodel.mean.get_value(np.arange(100,25000))-fscale*k, color=cols[filters[k%len(filters)]],linestyle='-')
            plt.plot(np.arange(100,bluecut), BBmodel.mean.get_value(np.arange(100,bluecut))*(np.arange(100,bluecut)/bluecut)**sup-fscale*k, color=cols[filters[k%len(filters)]],linestyle=':')
            
            plt_show_interactive()

        # Defining the names of the two parameters
        tri_cols = ["Temp", "Radius"]
        tri_labels = ["Temp", "Radius"]
        tri_truths = [truth[t] for t in tri_cols]

        # Plotting and saving corner plots for the two parameters 
        names = BBmodel.get_parameter_names()
        inds = np.array([names.index("mean:"+t) for t in tri_cols])
        
        # This corner plot can sometimes fail due to some range error and so wrap it in a try statement to test what the problem is
        try:
            corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels)
            corner_file_name = 'Corner plot for time step ' + str(i) + '.pdf'
            plt.savefig(os.path.join(corner_dir, corner_file_name))

            # Close the corner plot so we don't end up having a ton of corner plots
            plt.close()
        except ValueError:
            print(sampler.flatchain[:,inds])


        
    elif inc_noise == 'y':        
        # Model the blackbody at each time step without considering the correlated noise between the data points
        # Defining the truth array
        truth = dict(Temp = ini_temp, Radius = ini_rad)
        
        kwargs = dict(**truth)
        kwargs['bounds'] = dict(Temp = (ini_temp-temp_param_space, ini_temp+temp_param_space), Radius = (ini_rad-ini_rad*rad_param_space, ini_rad+ini_rad*rad_param_space))
        #kernel = input('\n>> What type of kernel would you like to use?[Matern32Kernel]   ')
        #metric = input('\n>> What value do you want to use for the metric?[500](q to quit and use constant colour)   ')
        # If user decides that they cannot get a good fit 
        metric1 = 5000
        
        # Or use default metric
        metric = metric1
        kernel = av_kernels['Matern32Kernel'](metric)

        mean_model = Bbody(**kwargs)

        np.var(flux)*kernel
        # Creating the george model and computing the fit
        BBmodel = george.GP(np.var(flux)*kernel, mean=mean_model)
        BBmodel.compute(wlref, yerr = ferr)
        
        # Defining the log likelihood and log prior
        def lnprob(p):
            # Uniform prior across the two parameters
            if (ini_temp-temp_param_space>p[0] or p[0]>ini_temp+temp_param_space):
                return -np.inf
            elif (ini_rad-ini_rad*rad_param_space>p[1] or p[1]>ini_rad+ini_rad*rad_param_space):
                return -np.inf
            
            BBmodel.set_parameter_vector(p)
            return BBmodel.log_likelihood(flux, quiet=True)

        # Calling the initial parameters and defining the number of dimensions by the two parameters in the BB model
        initial = BBmodel.get_parameter_vector()
        ndim = len(initial)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        
        if ask_at_each_step == 'y':
            # User input to how many burn-in's to perform and how many steps to run the mcmc for 
            burnin_num = input('\n> How many steps do you want to burn?[100]    ')
            if not burnin_num: burnin_num = 100
            burnin_num = int(burnin_num)
            num_steps = input('\n> How many steps do you want to run?[1000]    ')
            if not num_steps: num_steps = 1000
            num_steps = int(num_steps)

            # Asking the user how many walkers and MCMC steps to perform
            nwalkers = input('\n> How many walkers do you want for the MCMC?[32]    ')
            if not nwalkers: nwalkers = 32
            nwalkers = int(nwalkers)

        print("Running 1st burn-in...")
        p0 = initial + 0.00001*initial*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, burnin_num)

        print('Running 2nd burnin ... ')
        p0 = p0[np.argmax(lp)] + 0.00001*initial*np.random.randn(nwalkers, ndim)
        sampler.reset()
        p0, _, _ = sampler.run_mcmc(p0, burnin_num)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0, num_steps)

        # Printing the loglikelihood
        try:
            log_like = BBmodel.log_likelihood(flux)
            print('Log Likelihood: ' + str(log_like))
        except ValueError:
            print('Log Likelihood absolute value approached infinity')

        if abs(log_like) > 5000:
            print("\nThis time step has a very large log-likelihood, the plot is produced and the user decides whether to keep it or not.")
            
            plt.figure(5)
            plt.errorbar(wlref,flux-fscale*k,yerr = ferr,fmt = 'o',color=cols[filters[k%len(filters)]],label='%.1f' %ph)
        
            plt.plot(np.arange(100,25000), BBmodel.mean.get_value(np.arange(100,25000))-fscale*k, color=cols[filters[k%len(filters)]],linestyle='-')
            plt.plot(np.arange(100,bluecut), BBmodel.mean.get_value(np.arange(100,bluecut))*(np.arange(100,bluecut)/bluecut)**sup-fscale*k, color=cols[filters[k%len(filters)]],linestyle=':')
            plt_show_interactive()

            keep = input("\n>Would you like to keep this fit in the final plot of all time stamps?\nWarning! if the fit shown does not seem to accurately represent the points it is recommended to leave it out of the final plot.\nThe results will still be kept in the txt file. ('y', 'n')['n']    ")
            if not keep: keep = 'n'
            plt.close(5)

        else:
             keep = 'y'


        plt.figure(2)
        # Plotting the data for this time step
        plt.errorbar(wlref,flux-fscale*k,yerr = ferr,fmt = 'o',color=cols[filters[k%len(filters)]],label='%.1f' %ph)
        
        if keep == 'y':
            plt.plot(np.arange(100,25000), BBmodel.mean.get_value(np.arange(100,25000))-fscale*k, color=cols[filters[k%len(filters)]],linestyle='-')
            plt.plot(np.arange(100,bluecut), BBmodel.mean.get_value(np.arange(100,bluecut))*(np.arange(100,bluecut)/bluecut)**sup-fscale*k, color=cols[filters[k%len(filters)]],linestyle=':')

        plt_show_interactive()

        tri_cols = ["Temp", "Radius"]
        tri_labels = ["Temp", "Radius"]
        tri_truths = [truth[m] for m in tri_cols]

        names = BBmodel.get_parameter_names()
        inds = np.array([names.index("mean:"+m) for m in tri_cols])
        
        try:
            corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels)
            corner_file_name = 'Corner plot for time step ' + str(i) + '.pdf'
            plt.savefig(os.path.join(corner_dir, corner_file_name))
            plt.close()
        except:
            print('Error occured when making the corner plot.')
        

    # Asking the User to choose some percentile to assign for the errors in temperature and radius
    # Finding the percentile change over temperature and radius
    if ask_at_each_step == 'y':
        percent_unc = input ('\n> What percentile would you like to use for the uncertainy in temperature and radius?[75 percentile]     ')
        if not percent_unc: percent_unc = 75
        percent_unc = float(percent_unc)
    

    mcmc = np.percentile(sampler.flatchain[:,0], [50, percent_unc])
    T1 = mcmc[0]
    T1_err = np.diff(mcmc)

    mcmc = np.percentile(sampler.flatchain[:,1], [50, percent_unc])
    R1 = mcmc[0]
    R1_err = np.diff(mcmc)

    # Get pseudobolometric luminosity by trapezoidal integration, with flux set to zero outside of observed bands
    L1 = itg.trapz(flux1[np.argsort(wlref1)],wlref1[np.argsort(wlref1)])
    # Use flux errors and bandwidths to get luminosity error
    L1_err = np.sqrt(np.sum((bandwidths*ferr)**2))
    # Add luminosity to array (i.e. pseudobolometric light curve)
    L1arr.append(L1)
    L1err_arr.append(L1_err)

    # Calculate luminosity using alternative method of Stefan-Boltzmann, and T and R from fit
    L1bb = 4*np.pi*R1**2*5.67e-5*T1**4
    L1bb_err = L1bb*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)

    # Get UV luminosity (i.e. bluewards of bluest band)
    Luv = itg.trapz(blackbody(np.arange(100,bluecut),T1,R1)*(np.arange(100,bluecut)/bluecut)**sup,np.arange(100,bluecut))
    if bluecut < wlref[0]:
        # If no UV data and cutoff defaults to 3000A, need to further integrate (unabsorbed) BB from cutoff up to the bluest band
        Luv += itg.trapz(blackbody(np.arange(bluecut,wlref[0]),T1,R1),np.arange(bluecut,wlref[0]))
    
    # Use uncertainty in BB fit T and R to estimate error in UV flux
    Luv_err = Luv*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)

    # NIR luminosity from integrating blackbody above reddest band
    Lnir = itg.trapz(blackbody(np.arange(wlref[-1],25000),T1,R1),np.arange(wlref[-1],25000))
    Lnir_err = Lnir*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)

    # Treating UV and optical separately if user so decided:
    if sep=='y':
        # Used to occasionally crash, wrap in try statement
        try:
            # Grabbing the optical wavelenghts and corresponding flux
            opt_wlref = wlref[wlref>3000]
            opt_flux = flux[wlref>3000]
            opt_ferr = ferr[wlref>3000]

            # Fit BB only to data above 3000A
            try:
                BBparams, covar = curve_fit(blackbody,opt_wlref,opt_flux,p0=(10000,1e15),sigma=opt_ferr, bounds = [[0, 1e13], [25000, 1e20]])
            except:
                BBparams, covar = curve_fit(blackbody,opt_wlref,opt_flux,p0=(10000,1e15), bounds = [[0, 1e13], [25000, 1e20]])

            # This gives better estimate of optical colour temperature
            ini_temp_opt = BBparams[0]
            ini_rad_opt = np.abs(BBparams[1])

            # Grabbing the UV wavelenghts and corresponding flux
            uv_wlref = wlref[wlref<4000]
            uv_flux = flux[wlref<4000]
            uv_ferr = ferr[wlref<4000]

            # Fit BB only to data above 3000A
            try:    
                BBparams, covar = curve_fit(blackbody,uv_wlref,uv_flux,p0=(10000,1e15),sigma=uv_ferr, bounds = [[0, 1e13], [25000, 1e20]])
            except:
                BBparams, covar = curve_fit(blackbody,uv_wlref,uv_flux,p0=(10000,1e15), bounds = [[0, 1e13], [25000, 1e20]])

            # This gives better estimate of optical colour temperature
            ini_temp_uv = BBparams[0]
            ini_rad_uv = np.abs(BBparams[1])

            # Modelling the blackbody in the optical at each time step without considering the covariance between points
            print('\nBeginning MCMC for the optical regime')
            
            # Defining the truth array
            truth = dict(Temp = ini_temp_opt, Radius = ini_rad_opt)

            # Creating the george model and computing the fit
            BBmodel = george.GP(mean=Bbody(Temp=ini_temp_opt, Radius=ini_rad_opt), fit_mean = True)
            BBmodel.compute(opt_wlref, yerr = opt_ferr)

                # Defining the log likelihood and log prior
            def lnprob(p):
                # Uniform prior across the two parameters
                if (0>p[0] or p[0]>25000):
                    return -np.inf
                elif (1e13>p[1] or p[1]>1e20):
                    return -np.inf
                BBmodel.set_parameter_vector(p)
                return BBmodel.log_likelihood(opt_flux, quiet=True)
            
            # Calling the initial parameters and defining the number of dimensions by the two parameters in the BB model
            initial = BBmodel.get_parameter_vector()
            ndim = len(initial) 

            p0 = initial + 0.01*initial*np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
                    
            print("Running 1st burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 100)
            sampler.reset()

            print("Running production...")
            sampler.run_mcmc(p0, 1000)

            mcmc = np.percentile(sampler.flatchain[:,0], [50, percent_unc])
            Topt = mcmc[0]
            Topt_err = np.diff(mcmc)

            mcmc = np.percentile(sampler.flatchain[:,1], [50, percent_unc])
            Ropt = mcmc[0]
            Ropt_err = np.diff(mcmc)

            ###########################################################
            # Modelling the blackbody in the optical at each time step without considering the covariance between points
            print('\nBeginning MCMC for the uv regime')
            
            # Defining the truth array
            truth = dict(Temp = ini_temp_uv, Radius = ini_rad_uv)

            # Creating the george model and computing the fit
            BBmodel = george.GP(mean=Bbody(Temp=ini_temp_uv, Radius=ini_rad_uv), fit_mean = True)
            BBmodel.compute(uv_wlref, yerr = uv_ferr)

                # Defining the log likelihood and log prior
            def lnprob(p):
                # Uniform prior across the two parameters
                if (0>p[0] or p[0]>25000):
                    return -np.inf
                elif (1e13>p[1] or p[1]>1e20):
                    return -np.inf
                BBmodel.set_parameter_vector(p)
                return BBmodel.log_likelihood(uv_flux, quiet=True)
            
            # Calling the initial parameters and defining the number of dimensions by the two parameters in the BB model
            initial = BBmodel.get_parameter_vector()
            ndim = len(initial) 

            p0 = initial + 0.01*initial*np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
                    
            print("Running 1st burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 100)
            sampler.reset()

            print("Running production...")
            sampler.run_mcmc(p0, 1000)

            mcmc = np.percentile(sampler.flatchain[:,0], [50, percent_unc])
            Tuv = mcmc[0]
            Tuv_err = np.diff(mcmc)

            mcmc = np.percentile(sampler.flatchain[:,1], [50, percent_unc])
            Ruv = mcmc[0]
            Ruv_err = np.diff(mcmc)

            #####################################################################
            # Calculate luminosity predicted by Stefan-Boltzmann law for optical T and R
            L2bb = 4*np.pi*Ropt**2*5.67e-5*Topt**4
            L2bb_err = L2bb*np.sqrt((2*Ropt_err/Ropt)**2+(4*Topt_err/Topt)**2)

            # Use this BB fit to get NIR extrapolation, rather than the fit that included UV
            Lnir = itg.trapz(blackbody(np.arange(wlref[-1],25000),Topt,Ropt),np.arange(wlref[-1],25000))
            Lnir_err = Lnir*np.sqrt((2*Ropt_err/Ropt)**2+(4*Topt_err/Topt)**2)

            # Now do the separate fit to the UV
            # Because of line blanketing, this temperature and radius are not very meaningful physically, but shape of function useful for extrapolating flux bluewards of bluest band
            Luv = itg.trapz(blackbody(np.arange(100,wlref[0]),Tuv,Ruv),np.arange(100,wlref[0]))
            Luv_err = Luv*np.sqrt((2*Ruv_err/Ruv)**2+(4*Tuv_err/Tuv)**2)

            # Plot UV- and optical-only BBs for comparison to single BB
            plt.figure(2)
            plt.plot(np.arange(3000,25000),blackbody(np.arange(3000,25000),Topt,Ropt)-fscale*k,color=cols[filters[k%len(filters)]],linestyle='--',linewidth=1.5)
            plt.plot(np.arange(100,3600),blackbody(np.arange(100,3600),Tuv,Ruv)-fscale*k,color=cols[filters[k%len(filters)]],linestyle='-.',linewidth=1.5)

            plt_show_interactive()

        except:
            # If UV fits failed, just write out the single BB fits
            Topt,Topt_err,Ropt,Ropt_err,L2bb,L2bb_err = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Write out BB params, and optical-only BB params, to file
        out2.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,T1,T1_err,R1,R1_err,L1bb,L1bb_err,Topt,Topt_err,Ropt,Ropt_err,L2bb,L2bb_err))
    else:
        # If separate fits were not used, just write out the single BB fits
        out2.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,T1,T1_err,R1,R1_err,L1bb,L1bb_err))

    # Estimate total bolometric luminosity as integration over observed flux, plus corrections in UV and NIR from the blackbody extrapolations
    # If separate UV fit was used, Luv comes from this fit and Lnir comes from optical-only fit
    # If no separate fits, Luv and Lnir come from the same BB (inferior fit and therefore less accurate extrapolation)
    L2 = Luv + itg.trapz(flux,wlref) + Lnir
    # Add errors on each part of the luminosity in quadrature
    L2_err = np.sqrt(L1_err**2 + (Luv_err)**2 + (Lnir_err)**2)
    # Append to light curve
    L2arr.append(L2)
    L2err_arr.append(L2_err)

    # Write light curve to file: L1 is pseudobolometric, L2 is full bolometric
    out1.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,L1,L1_err,L2,L2_err))

    plt.draw()
    plt.xlabel('Wavelength (Ang)')
    plt.ylabel(r'$\mathit{L}_\lambda$ + constant')

    if total_number <= 100:
        plt.legend(numpoints=1,ncol=2,fontsize=11,frameon=True, loc = 'center right')

    plt_show_interactive()

    curve_fit_temp = T1
    curve_fit_rad = R1

    # Counter shifts down next SED on plot for visibility
    k += 1

    print('Completed epoch {0} out of {1}'.format(k-1, total_number))

plt.figure(2)
plt.yticks([])
plt.xlim(min(wlref)-2000,max(wlref)+3000)
plt.tight_layout(pad=0.5)

plt_show_interactive()

# Add methodologies and keys to output files so user knows which approximations were made in this run
out1.write('\n#KEY\n# Lobs = integrate observed fluxes with no BB fit\n# L+BB = observed flux + BB fit extrapolation')
out1.write('\n# See logL_obs_'+sn+'_'+filters+'.txt and logL_bb_'+sn+'_'+filters+'.txt for simple LC files')
out1.write(method)
out2.write('\n#KEY\n# _bb = blackbody fit to all wavelengths, _opt = fit only data redwards of 3000A\n# L_bb = luminosity from Stefan-Boltzman; L_opt = same but using T_opt and R_opt')
out2.write('\n# (in contrast, bol_'+sn+'_'+filters+'.txt file contains trapezoidal integration over observed wavelengths)')

# Close output files
out1.close()
out2.close()

# Make final light curves into numpy arrays
L1arr = np.array(L1arr)
L1err_arr = np.array(L1err_arr)
L2arr = np.array(L2arr)
L2err_arr = np.array(L2err_arr)

print('\n\n*** Done! Displaying bolometric light curve ***')

# Save convenient log versions of light curves
logout = np.array(list(zip(phase,np.log10(L1arr),0.434*L1err_arr/L1arr))).astype('float')
#logoutBB = np.array(list(zip(phase,np.log10(L2arr),0.434*L2err_arr/L2arr)))
try:
    logoutBB = np.array(list(zip(phase,np.log10(L2arr),0.434*L2err_arr.flatten()/L2arr))).astype('float')
except:
    print('If this error appears check the shape of the input arrays as they may be in a strange shape.')

np.savetxt(outdir+'/logL_obs_'+sn+'_'+filters+'.txt',logout,fmt='%.3f',delimiter='\t')
np.savetxt(outdir+'/logL_bb_'+sn+'_'+filters+'.txt',logoutBB,fmt='%.3f',delimiter='\t')


# Plot final outputs
plt.figure(3,(8,8))
plt.clf()

plt.subplot(311)

# Plot pseudobolometric and bolometric (including BB) light curves (logarithmic versions)
plt.errorbar(logout[:,0],logout[:,1],logout[:,2],fmt='o',color='k',markersize=12,label='Observed flux only')
plt.errorbar(logoutBB[:,0],logoutBB[:,1],logoutBB[:,2],fmt='d',color='r',markersize=9,label='Plus BB correction')
plt.ylabel(r'$log_{10} \mathit{L}_{bol}\,(erg\,s^{-1})$')
plt.legend(numpoints=1,fontsize=16)
plt.xticks(visible=False)

# Get blackbody temperature and radius
bbresults = np.genfromtxt(outdir+'/BB_params_'+sn+'_'+filters+'.txt')

# Plot temperature in units of 10^3 K
plt.subplot(312)
plt.errorbar(bbresults[:,0],bbresults[:,1]/1e3,bbresults[:,2]/1e3,fmt='o',color='k',markersize=12,label='Fit all bands')
plt.ylabel(r'$\mathit{T}_{BB}\,(10^3K)$')
plt.xticks(visible=False)

if len(bbresults[0])==13:
    # If separate fit to optical-only, plot this too
    plt.errorbar(bbresults[:,0],bbresults[:,7]/1e3,bbresults[:,8]/1e3,fmt='s',color='c',markersize=8,label=r'Fit >3000$\AA$')
    plt.legend(numpoints=1,fontsize=16)

# Plot radius in units of 10^15 cm
plt.subplot(313)
plt.errorbar(bbresults[:,0],bbresults[:,3]/1e15,bbresults[:,4]/1e15,fmt='o',color='k',markersize=12,label='Fit all bands')
plt.ylabel(r'$\mathit{R}_{BB}\,(10^{15}cm)$')

if len(bbresults[0])==13:
    plt.errorbar(bbresults[:,0],bbresults[:,9]/1e15,bbresults[:,10]/1e15,fmt='s',color='c',markersize=8,label='Exclude UV')

# X-label for all subplots
plt.xlabel(xlab)

plt.subplots_adjust(hspace=0)
plt.tight_layout(pad=0.5)
plt.draw()

plt_show_interactive()


plt.figure(1)
plt.savefig(outdir+'/interpolated_lcs_'+sn+'_'+filters+'.pdf')

plt.figure(2)
plt.savefig(outdir+'/bb_fits_'+sn+'_'+filters+'.pdf')

plt.figure(3)
plt.savefig(outdir+'/results_'+sn+'_'+filters+'.pdf')
# Wait for key press before closing plots!
fin = input('\n\n> PRESS RETURN TO EXIT...\n')