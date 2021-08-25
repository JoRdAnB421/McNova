# McNova
This is a python program for calculating bolometric luminosities of supernovae using gaussian processes and MCMC sampling. This is done using a set of observed magnitudes which can be either apparent or absolute.

This program was based on the ["Superbol.py"](https://github.com/mnicholl/superbol) program created by Matt Nicholl in 2015. However, while "Superbol.py" using curve_fit to model the observed flux evolution and calculate the bolometric luminosity, "McNova" uses gaussian processes to model the observed flux and then performs and MCMC sampling to search for the optimal parameters needed to calculate bolometric luminosities.

Requirements and Usage
 - numpy
 - scipy
 - matplotlib
 - astropy
 - george
 - emcee
 - corner

Code is written in Python 3 (Should work for all verisons) 

To run code:
```python McNova.py```

Depending on your python IDE, you may face trouble with the plots not showing or being unresponsive. Particular true when using Spyder 4 or previous versions, to get around this you can change the namespace that the code is run in. As default Spyder 4.0 and previous versions run in an empty namespace, however by changing this to the console's namespace, you will have fully interactive plots without breaking the code. 

To change the namespace follow these steps:
- Click on run on the top toolbar.
- Click Configuration per file.
- Under 'General settings' make sure that the option 'Run in console's namespace instead of an empty one' is highlighted.
- You may need to restart the terminal before running the code. 

# Versions
```Version 0   : Written by Jordan Barber (CU), 2021```

# Usage
Run the program in the same directory as your photometry data.

The aim of McNova is that anyone can use it without prior knowledge of its workings, therfore the user will be prompted at every step.

All data files should follow the naming convention SNname_filters.txt, e.g:
```PTF12dam_ugriz.txt```
Multiple files per event are allowed and the user will be prompted which (or all) files are to be used. The format of teh files must be:
```MJD filter1 err1 filter2 err2 ...```
MJD can be replaced by phase or some other time parameter but this must be consistent between files.
- Important: Bands must be in their common systems: AB mag for ugrizy, Gaia, ATLAS, GALEX; Vega mags for UBVRIJHK, Swift (S=UVW2 D=UVM2 A=UVW1) and NEOWISE (W=W1 Q=W2).
- Important: Order of filter magnitudes in file must match order of filters in file name.

Computes pseudobolometric light curves by integrating flux over observed filters only ("Lobs") as well as full bolometric light curves with blackbody extrapolations ("Lbb"). BB fit parameters are also stored as an output (temperature and radius). User is also able to fit optical and UV separateley to mimic line blanketing.

The user will be prompted to input a distance to the event, this can be in redshift or distance modulus. If redshift is entered then a standard cosmology (Default is using astropy though Ned Wrights cosmocalc can be used if that section of the code is uncommented) is used to calculate the distance modulus. 

User will be prompted to choose a reference filter, this defaults to the filter with the most data points. Where there is no match in epoch between reference filter and other filters, the user can choose to interpolate/extrapolate based on either constant colours or interactive fitting with a gaussian process (using the gp object in george). Where constant colours are used, their uncertainties are given to be the same as that of the closest data point with errors. Where interactive fitting is picked, the uncertainties are given by the resulting covariance matrix from the fitting. Where there are large gaps between epochs lacking any data to fit, the gaussian process can give some extremely high uncertainties (this is particularly true for extrapolating past the last given epoch). In this case the user is given the option to artificially reduce these uncertainties to some fraction of the expected magnitude at that time. These uncertainties are also interactively plotted so the user can spend time ensuring sensible uncertainties.

All interpolated light curves are saved on output are saved as an output. On subsequent runs, the code will first search for these interpolated light curves so you can choose to skip the interpolation stage. All output files at the end of the code will contain all of the filters used in the integration within the filenames. 

# Steps
 - Find files associated with SN and determine available filters and data
 - Correct for time dilation, distance, and approximate K-correction if desired
 - Map light curves in each filter to a common set of times
    (typically defined by the filter with most observations)
    - Interpolation options: assuming constant colour with respect to reference filter or gaussian process fits
       (user determines the type of kernel and the size of the metric used)
    - Extrapolation: using gaussian process or assuming constant colour with respect to reference filter.
      (user determines the type of kernel and the size of the metric used)
      - Where uncertainties are extremely large, give the user option to restrict to realistic errors.
        (The user can choose to change only uncertainties outside of the epoch range where the is data, or at any epoch which has uncertainties larger than a given amount)
    - Save interpolated light curves for reproducability
 - User sets up the parameters for the posterior sampling of a Monte Carlo Markov Cain, which attempts to fit a blackbody to SED at each epoch and find the optimal parameters        (temperature and radius) (most SNe can be reasonably approximated by blackbody above ~3000 Angstroms)
    - For each epoch, corner plots are created to show the sampled parameter space of both temperature and radius, the user is able to choose some upper percentile to take as the       error on the parameters.
    - For a given epoch, it is possible that the BB fit somewhat breaks, this seems to be the case where the errors on some of the data points are significantly less than               the other points and so there is not much space for the fitting to be adjusted for the less well constrained points. This event is typically indicated by a very large log         likelihood absolute value (anything greater than ~ 5000 is cause for suspicion), therefore when the code encounters a case of these very large log likelihood values, it will 
      plot the data points in question and the best fit it could reach for the user to see, the user then has the option to keep this fit in the final plot of all epochs (usually       for when the fit still reasonably resembles the shape of the data points) or to neglect the fit from the final plot (for where the fit doesn't resemble the shape of the data       points). In either case the results of this fit are still stored in the txt files.
 - In UV, user can choose to:
    - fit SED over all wavelengths with single blackbody
    - fit separate blackbodies to optical and UV (if UV data exist).
        Optical fit gives better temperature estimate than single BB.
         UV fit used only to extrapolate flux for bolometric luminosity.
    - use a simple prescription for line blanketing at UV wavelengths,
        defined as L_uv(lambda < cutoff) = L_bb(lambda)*(lambda/cutoff)^x, where x is chosen by user.
        Cutoff is either set to bluest available band, or if bluest band is >3000A, cutoff = 3000A
- Numerically integrate observed SEDs, and account for missing UV and NIR flux using blackbody extrapolations.
    NIR is easy, UV uses options described above
    
Outputs
------
- interpolated_lcs_SN_filters.txt
    - multicolour light curves mapped to common times.
    - Footer gives methods of interpolation and extrapolation.
    - If file exists, can be read in future to skip interpolating next time.
- bol_SN_filters.txt
    - main output.
    - Contains pseudobolometric light curve, integrated trapezoidally,
    and bolometric light curve including the additional BB corrections, and errors on each.
    - Footer gives filters and method of UV fitting.
- logL_obs_SN_filters.txt
    - same pseudobolometric (observed) light curve, in convenient log form
- logL_obs_SN_filters.txt
    - light curve with the BB corrections, in convenient log form
- BB_params_SN_filters.txt
    - fit parameters for blackbodies: T, R and inferred L from Stefan-Boltzmann law (can compare with direct integration method).
    - If separate optical/UV fit, gives both T_bb (fit to all data) and T_opt (fit only to data >3000 A)
- Corner_plots_SN
    - A directory filled with corner plots of the sampled parameter space for the temperature and radius for every each epoch.
