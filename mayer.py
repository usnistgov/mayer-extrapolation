"""
Perform a Mayer sampling Monte Carlo simulation
https://doi.org/10.1103/PhysRevLett.92.220601
"""

import math
import random
import copy
from itertools import islice, count
import pandas
import accumulator

def origin_sq_distance(pos_vec):
    """ Return the squared distance from the origin """
    rsqsum = 0.
    for dummy, item in enumerate(pos_vec):
        rsqsum += item**2
    return rsqsum
assert math.fabs(origin_sq_distance([math.sqrt(7)]) - 7.) < 1e-12

def ulj(rsq, alpha=6.):
    """ Return the potential energy of a Lennard Jones particle
        rsq: squared separation distance """
    return 4.*(rsq**(-alpha) - rsq**(-0.5*alpha))
assert math.fabs(ulj(2**(2./6.)) + 1) < 1e-12

def uhs(rsq):
    """ Return the potential energy of a hard sphere particle
        rsq: squared separation distance """
    if rsq < 1:
        return 1e100
    return 0.
assert uhs(1.001) == 0.

def second_virial_coefficient(beta=1., num_tune=int(1e6), freq_tune=int(1e4), num_prod=int(1e7),
                              freq_print=int(1e6)):
    """ Return dictionary of meta data and pandas data frame with taylor series
        beta: inverse temperature
        num_tune: number of trials to tune max move
        freq_tune: number of trials between each tune
        num_prod: total number of trials
        freq_print: number of trials before printing status"""
    metadata = {"beta" : beta, "num_tune" : num_tune, "freq_tune" : freq_tune,
                "num_prod" : num_prod, "freq_print" : freq_print}
    random.seed()
    pos_vec = [1.1, 0, 0]
    pe_old = ulj(origin_sq_distance(pos_vec))
    beta = metadata["beta"]
    f12old = math.exp(-beta*pe_old) - 1
    f12ref = math.exp(-beta*uhs(origin_sq_distance(pos_vec))) - 1
    max_disp = 0.1        # maximum 1D distance of displacement
    num_disp_accept = 0   # number of accepted displacements
    num_disp_attempt = 0  # number of attempted displacements
    targ_accept = 0.25    # target acceptance for displacements
    num_order = 20        # maximum order of extrapolation
    taylor_coeffs = pandas.DataFrame(index=range(num_order+1))
    taylor_coeffs["b2"] = 0.
    mayer_moments = list()
    for order in range(num_order):
        mayer_moments.append(accumulator.Accumulator())
    mayer = accumulator.Accumulator()
    mayer_ref = accumulator.Accumulator()

    # begin "num_prod" Mayer sampling Monte Carlo simulation trials
    for itrial in islice(count(1), metadata["num_prod"] - 1):
        # randomly displace the second particle
        pos_vec_old = copy.deepcopy(pos_vec)   # store old position
        num_disp_attempt += 1   # count displacement attempts
        for index, dummy in enumerate(pos_vec):
            pos_vec[index] += max_disp*random.uniform(-1., 1.)

        # compute the energy and determine if move is accepted
        pe_attempted = ulj(origin_sq_distance(pos_vec))
        f12 = math.exp(-beta*pe_attempted) - 1.
        if random.random() < math.fabs(f12)/math.fabs(f12old):
            # accept trial
            num_disp_accept += 1
            f12old = f12
            pe_old = pe_attempted
            pe_ref = uhs(origin_sq_distance(pos_vec))
            f12ref = math.exp(-beta*pe_ref) - 1.
        else:
            # reject trial
            pos_vec = copy.deepcopy(pos_vec_old)

        # tune maximum trial displacement
        if (itrial < metadata["num_tune"]) and (itrial % metadata["freq_tune"] == 0):
            if float(num_disp_accept) > targ_accept*float(num_disp_attempt):
                max_disp *= 1. + 0.05
            else:
                max_disp *= 1. - 0.05
            num_disp_accept = num_disp_attempt = 0
            if itrial == metadata["num_tune"]:
                print("# max_disp ", max_disp)

        # average mayer and derivatives only after tuning is complete
        if itrial > metadata["num_tune"]:
            if f12old < 0:
                mayer.accumulate(-1)
            else:
                mayer.accumulate(1)
            mayer_ref.accumulate(f12ref/math.fabs(f12old))
            uebu = math.exp(-beta*pe_old)/math.fabs(f12old)
            for order in range(num_order):
                uebu *= -pe_old
                mayer_moments[order].accumulate(uebu)

            # store and print Taylor coefficients
            if ((itrial % metadata["freq_print"] == 0) or
                    (itrial == metadata["num_prod"] - 1)):
                reffac = 2.*math.pi/3./mayer_ref.average()
                taylor_coeffs["b2"][0] = reffac*mayer.average()
                for order in range(num_order):
                    taylor_coeffs["b2"][order + 1] = (reffac*mayer_moments[order].average()
                                                      /math.factorial(order + 1))
                print(taylor_coeffs)
    return metadata, taylor_coeffs
