# from dlr_process_clean import get_possible_hosts_loop
import dlr_functions as fnc
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import argparse

# Basic imports
import os
import sys
import random
import time


# Plots
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.image as mpimg
import matplotlib.colors as mcolors

# Astropy
from astropy.table import Table
from astropy import coordinates as co, units as u
from astropy.io import fits

# Legacy Survey catalogs
import pyvo

# Legacy survey images
import urllib
from PIL import Image
import requests

# Debugging
import ipdb  # *ipdb.set_trace()

filename_GAMA = "data_files/gkvScienceCatv02.fits"
SN_df = pd.read_csv("data_files/tns_SNIa_20240424_copy.csv")

outfilename = "test_script.csv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Crossmatch SN with galaxies using DLR"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="run test suit only",
    )

    args = parser.parse_args()  # Parse the arguments

    if args.testing:
        id_sn_test = [
            "2021dnl",
            "2023hvr",
            "2020aeth",
            "2021ais",
            "2018jky",
            "2020yhn",
            "2022ej",
            "2021jnf",
            "2018jnf",
            "2019awd",
        ]
        ind_sn = SN_df[SN_df["name"].isin(id_sn_test)]  # Filter rows with .isin()
    else:
        ind_sn = SN_df  # Use the full dataset

    ###### INPUTS ######
    ra_sn = ind_sn["ra"].values
    dec_sn = ind_sn["declination"].values
    id_sn = ind_sn["name"].values

    possible_host_gama = fnc.get_possible_hosts_loop(
        filename_GAMA,
        outfilename,
        id_sn,
        ra_sn,
        dec_sn,
        catalogue="GAMA",
        radius_deg=1 / 60,
        dDLR_cut=4,
        type_dlr="classic",
        verbose=False,
        overwrite=True,
    )

    possible_host_legacy = fnc.get_possible_hosts_loop(
        None,
        outfilename,
        id_sn,
        ra_sn,
        dec_sn,
        catalogue="lsdr10",
        radius_deg=1 / 60,
        dDLR_cut=4,
        type_dlr="classic",
        verbose=False,
        overwrite=True,
    )


"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SNN trained models performance")

    parser.add_argument(
        "--path_models26X",
        default="./../snndump_26XBOOSTEDDES",
        type=str,
        help="Path to SNN models trained with 26 realisations of DES 5-year",
    )

parser.add_argument(
        "--testing", action="store_true", help="run test suit only",
    )
if args.testing:
    ids = ....
"""
