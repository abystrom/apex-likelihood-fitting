import numpy as np
import scipy
import numdifftools
import astropy.table as atpy
import astropy.coordinates as acoord
import astropy.units as auni 
from math import pi

def loglikelihood(params, *args):
    """This function defines the loglikelihood as given in Eq. (8), where we replace the mean, mu, 
    with the predicted velocity for a given position on the sky as given in Eq. (10).
    The function returns the negative loglikelihood, as that is used with scipy's minimisation routine (as it does not
    contain a maximisation routine)"""

    # below are the parameters we want to get from the fitting of Eq. (10)
    v_compr, v_apex, l_apex, b_apex, sigma = params

    # the data is given as a tuple; let's break it into the components
    data_l = np.deg2rad(args[0])
    data_b = np.deg2rad(args[1])
    data_v = args[2]
    data_v_error = args[3]

    theta = acoord.angular_separation(data_l, data_b, l_apex, b_apex) # the angular separation between the stars' coordinates and the apex position
    mu = v_compr + v_apex*np.cos(theta) # in Eq. (8), we replace the general mu with Eq. (10)

    # the likelihood L as defined in Eq. (8)
    L = -(1/2)*np.sum((np.log(2*pi*(sigma**2 + data_v_error**2)) + (data_v - mu)**2/(sigma**2 + data_v_error**2)))

    return -L # note that we return the negative L; because scipy does not have a maximisation function, only a minimisation one

def loglikelihood_minimisation(data, initial_guess, bounds): 
    """A function that minimises the loglikelihood of our data using maximum likelihood estimation. As scipy does not have a maximisation
    routine, we use the negative loglikelihood and minimise it instead."""
    
    minimisation_result = scipy.optimize.minimize(loglikelihood, initial_guess, args=data, method='Nelder-Mead', options={'maxiter':10000}, bounds=bounds)
    
    if minimisation_result.success == True:
        
        # if successful, first return the parameter values
        v_compr = minimisation_result.x[0]
        v_apex  = minimisation_result.x[1]
        l_apex  = np.rad2deg(minimisation_result.x[2])
        b_apex  = np.rad2deg(minimisation_result.x[3])
        sigma   = minimisation_result.x[4]

        # then also compute the associated errors, as given by Eq. (9)
        def loglikelihood_just_params(params): # used in numdifftools.Hessian which expects a function of the form f(params)
            return loglikelihood(params, *data)
        Hessian_function = numdifftools.Hessian(loglikelihood_just_params)
        Hessian_matrix = Hessian_function(minimisation_result.x)
        covariance_matrix = np.linalg.inv(Hessian_matrix) 
        errors = np.sqrt(np.diag(covariance_matrix))
        
        v_compr_error = errors[0]
        v_apex_error  = errors[1]
        l_apex_error  = np.rad2deg(errors[2])
        b_apex_error  = np.rad2deg(errors[3])
        sigma_error   = errors[4]
        
        return v_compr, v_apex, l_apex, b_apex, sigma, v_compr_error, v_apex_error, l_apex_error, b_apex_error, sigma_error
    else:
        print('No fit success')