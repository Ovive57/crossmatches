# from dlr_process_clean import get_possible_hosts_loop
import dlr_functions as fnc

# Importing libraries
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


# Debugging
import ipdb  # *ipdb.set_trace()

# To test GAMA too uncomment this :
# filename_GAMA = "data_files/gkvScienceCatv02.fits"

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

    ###### If you want to test GAMA too, follow the README steps and uncomment this part: ########
    # possible_host_gama = fnc.get_possible_hosts_loop(
    #    filename_GAMA,
    #    outfilename,
    #    id_sn,
    #    ra_sn,
    #    dec_sn,
    #    catalogue="GAMA",
    #    radius_deg=1 / 60,
    #    dDLR_cut=4,
    #    type_dlr="classic",
    #    verbose=False,
    #    overwrite=True,
    # )

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
