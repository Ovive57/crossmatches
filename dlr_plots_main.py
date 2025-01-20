import sys

import pandas as pd
import numpy as np
from astropy.table import Table
import argparse


import matplotlib.pyplot as plt

import dlr_functions as fnc
import dlr_plots_functions as plt_fnc

import style

plt.style.use(style.style1)


filename_GAMA = "data_files/gkvScienceCatv02.fits"
filename_sn = "data_files/tns_SNIa_20240424_copy.csv"

SN_df = pd.read_csv(filename_sn)

outfilename = "test_script.csv"

catalogue = "lsdr10"

radius_deg = 0.5 / 60
#! The field of view is selectable between either 15.3×28.3 arcsec with 0.7′′ spatial sampling or 27.4×50.6 arcsec with 1.25′′ sampling.
scale_factor = [0, 4]
# filename = (
#     "output_files/lsdr10/possible_hosts/lsdr10_after_cuts_tns_SNIa_7041_mt_2matches.csv"
# )
# filename = (
#     "output_files/lsdr10/possible_hosts/lsdr10_after_cuts_tns_SNIa_7041_no_host_ac.csv"
# )
# filename = (
#     "output_files/lsdr10/possible_hosts/lsdr10_after_cuts_tns_SNIa_7041_2matches.csv"
# )
# filename = None

# filename = "output_files/lsdr10/possible_hosts/lsdr10_after_cuts_tns_SNIa_7041_2matches_bt05.csv"
filename = "follow_up/lsdr10_after_cuts_tns_SNIa_priorities.csv"
filename = None
filename_host_legacy = (
    "output_files/lsdr10/possible_hosts/lsdr10_after_cuts_tns_SNIa_7041.csv"
)
filename_host_GAMA = (
    "output_files/GAMA/possible_hosts/GAMA_after_cuts_tns_SNIa_7041.csv"
)

outdir = "plots/dDLR_first_try/"
outdir_test = "plots/test/"
save = True
colour = ["red", "red"]

rect_width = 27.4 / 3600
rect_height = 50.6 / 3600
radius_deg = 0.5 / 60

catalogue = "lsdr10"

# filename = None
# if filename != None:
#     file_pd = pd.read_csv(filename)
#     id_sn = file_pd["sn_name"].unique()

# else:
#     id_sn = ["2021dnl"]

file_pd = pd.read_csv(filename_sn)
id_sn = file_pd["name"].unique()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crossmatch SN with galaxies using DLR"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="run test suit only",
    )
    parser.add_argument(
        "--testing_host", action="store_true", help="run test suit only"
    )

    parser.add_argument(
        "--testing_redshift", action="store_true", help="run test suit only"
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
            "2018jnf",
            "2019awd",
        ]
        ind_sn = SN_df[SN_df["name"].isin(id_sn_test)]  # Filter rows with .isin()
        radius_deg = 0.5 / 60
        outdir = outdir_test

    elif args.testing_host:
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
        plt_fnc.plot_host(
            id_sn=id_sn_test,
            catalogue="lsdr10",
            radius_deg=radius_deg,
            filename=filename_host_legacy,
            save=True,
            out_dir=outdir_test + "hosts/",
        )
        plt_fnc.plot_host(
            id_sn=id_sn_test,
            catalogue="GAMA",
            radius_deg=radius_deg,
            filename=filename_host_GAMA,
            save=True,
            out_dir=outdir_test + "hosts/",
        )
        sys.exit()
    elif args.testing_redshift:
        plt_fnc.redshift_dif(
            filename_sn=filename_sn, filename_gal=filename_host_GAMA, catalogue="GAMA"
        )
        sys.exit()
    else:
        ind_sn = SN_df[SN_df["name"].isin(id_sn)]  # Use the full dataset

    ###### INPUTS ######
    ra_sn = ind_sn["ra"].values
    dec_sn = ind_sn["declination"].values
    id_sn = ind_sn["name"].values

    plt_fnc.loop_ellipses_with_image(
        id_sn,
        ra_sn,
        dec_sn,
        catalogue=catalogue,
        radius_deg=radius_deg,
        dDLR_bar=True,
        filename_GAMA=filename_GAMA,
        filename_legacy=filename_host_legacy,
        save=save,
        overwrite=False,
        out_dir=outdir,
    )

    plt_fnc.loop_ellipses_with_image(
        id_sn,
        ra_sn,
        dec_sn,
        catalogue="GAMA",
        radius_deg=radius_deg,
        rectangle=False,
        rect_width=rect_width,
        scale_factor=scale_factor,
        colours=colour,
        dDLR_bar=True,
        filename_GAMA=filename_GAMA,
        save=save,
        overwrite=False,
        out_dir=outdir,
    )
