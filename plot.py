"""
Plot extrapolation after either running a Mayer-sampling simulation, or
reading the final result of a simulation from a file.

The mayer sampling simulations result in a table where the indicies are the
extrapolation order (about "beta0") and the values are the coefficients of
the Tayler series.

For example, the 0th index, ~ -5.2829 is the zeroth order extrapolation,
which is the actual virial coefficient value computed at "beta0."
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import pade
import numpy as np
import pandas as pd

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--series_filename', '-s', default="taylor_series.csv",
    type=str, help="csv format file containing taylor series")
PARSER.add_argument('--metadata_filename', '-m', default="metadata.json",
    type=str, help="json format file containing mayer sampling MC parameters")
PARSER.add_argument('--simulate', action='store_true', default=False,
    help="perform Mayer-sampling Monte Carlo to generate taylor series")
PARSER.add_argument('--series_type', default="pade", type=str,
    help="Pade approximate (pade) or Taylor series (taylor)")
PARSER.add_argument('--beta0', '-b', default=1., type=float,
    help="inverse temperature at which to perform Mayer-sampling Monte Carlo")
PARSER.add_argument('--num_tune', default=int(1e6), type=int,
                    help="number of trials to tune max move")
PARSER.add_argument('--num_prod', default=int(1e7), type=int,
                    help="number of total trials in Mayer-sampling Monte Carlo")
PARSER.add_argument('--freq_tune', default=int(1e4), type=int,
                    help="number of trials between each tune")
PARSER.add_argument('--freq_print', default=int(1e6), type=int,
                    help="number of trials before printing status")
ARGS = PARSER.parse_args()
print(ARGS)

if ARGS.simulate:
    import mayer
    METADATA, TAYLOR_SERIES = mayer.second_virial_coefficient(
        beta=ARGS.beta0,
        num_tune=ARGS.num_tune, freq_tune=ARGS.freq_tune,
        num_prod=ARGS.num_prod, freq_print=ARGS.freq_print)

    # write metadata and taylor series to file
    with open(ARGS.metadata_filename, 'w') as f:
        json.dump(METADATA, f)
    TAYLOR_SERIES.to_csv(ARGS.series_filename)
else:
    # read metadata and taylor series from file
    with open(ARGS.metadata_filename, "r") as f:
        METADATA = json.load(f)
    TAYLOR_SERIES = pd.read_csv(ARGS.series_filename, comment="#")

def series_to_num_order(series):
    """ Return the highest order of extrapolation """
    return len(series) - 1
assert series_to_num_order(TAYLOR_SERIES) == 20

def series_evaluate(series, deta, order=1, series_type="taylor"):
    """ Return evaluated polynomial using the data frame """
    assert order <= series_to_num_order(series)
    coeffs = series["b2"].values[:(order+1)]
    if series_type == "pade" or series_type == "q" or series_type == "p":
        p_pade, q_pade = pade(coeffs, 1)
        if series_type == "pade":
            return p_pade(deta)/q_pade(deta)
        if series_type == "q":
            return q_pade(deta)
        return p_pade(deta)
    poly = np.poly1d(coeffs[::-1])
    return poly(deta)
assert series_evaluate(TAYLOR_SERIES, deta=[0], order=20) == TAYLOR_SERIES["b2"][0]

def val2map(values, cmname="jet"):
    """ Given list of values, return scalar_map for colors """
    jet = plt.get_cmap(cmname)
    c_norm = colors.Normalize(vmin=values[0], vmax=values[-1])
    return cmx.ScalarMappable(norm=c_norm, cmap=jet)

def series_plot(series, eta0, xrnge, series_type="taylor", num_order=None):
    """ Plot the extrapolations for various orders """
    if num_order is None:
        num_order = series_to_num_order(series)
    empty_plot = plt.contourf([[0, 0], [0, 0]], np.arange(1, num_order + 0.01, 0.01), cmap="jet_r")
    values = range(1, num_order+1)
    scalar_map = val2map(values, cmname="jet_r")
    for dummy, order in enumerate(values):
        yvals = series_evaluate(series, xrnge - eta0, order=order, series_type=series_type)
        color_val = scalar_map.to_rgba(order)
        plt.plot(xrnge, yvals, color=color_val)
    bound = range(1, num_order + 1)
    if num_order > 15:
        bound = range(1, num_order + 1, 2)
    cbar = plt.colorbar(empty_plot, ticks=bound)
    cbar.set_label("extrapolation order", fontsize=14)

# Plot the series
series_plot(TAYLOR_SERIES, ARGS.beta0, xrnge=np.arange(0.01, 3.5, 0.1),
            series_type=ARGS.series_type, num_order=20)
plt.xlim([0, 3.0])
plt.ylim([-50, 10])
plt.xlabel(r'$\beta\epsilon$', fontsize=16)
plt.ylabel(r'$B_2/\sigma^3$', fontsize=16)
plt.savefig("plot.png", transparent=True, bbox_inches="tight")
plt.show()
