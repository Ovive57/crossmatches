# Importing libraries
import argparse
import pandas as pd


# Debugging
import ipdb  # *ipdb.set_trace()

# Importing functions
import dlr_functions as fnc

# To test GAMA too uncomment this :
filename_GAMA = "data_files/gkvScienceCatv02.fits"

SN_df_testing = pd.read_csv("data_files/tns_SNIa_20240424_copy.csv")
SN_df = pd.read_csv("data_files/tns_SNIa_20240424_copy.csv")
# SN_df = pd.read_csv("data_files/to_crossmatch_host.csv")

radius_deg = 0.5 / 60
dDLR_cut = 4
# outfilename = f"test_script_{int(radius_deg*2*60)}arcmin.csv"
outfilename = "tns_SNIa_7041.csv"
# outfilename = "ZTF_SNIa_paperAn.csv"
out_dir = "output_files/"

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
        ind_sn = SN_df_testing[
            SN_df_testing["name"].isin(id_sn_test)
        ]  # Filter rows with .isin()
        outfilename = f"test_script_{int(radius_deg*2*60)}arcmin.csv"
        ###### INPUTS ######
        ra_sn = ind_sn["ra"].values
        dec_sn = ind_sn["declination"].values
        id_sn = ind_sn["name"].values
    else:
        ind_sn = SN_df  # Use the full dataset

        ###### INPUTS ###### #! Change the column names to match the ones in your dataset
        ra_sn = ind_sn["ra"].values
        dec_sn = ind_sn["declination"].values
        id_sn = ind_sn["name"].values

    ###### If you want to test GAMA too, follow the README steps and uncomment this part: ########
    # possible_host_gama = fnc.get_possible_hosts_loop(
    #     filename_GAMA,
    #     outfilename,
    #     id_sn,
    #     ra_sn,
    #     dec_sn,
    #     catalogue="GAMA",
    #     out_dir=out_dir,
    #     radius_deg=radius_deg,
    #     dDLR_cut=dDLR_cut,
    #     type_dlr="classic",
    #     save=True,
    #     verbose=False,
    #     overwrite=False,
    # )

    # possible_host_legacy = fnc.get_possible_hosts_loop(
    #     None,
    #     outfilename,
    #     id_sn,
    #     ra_sn,
    #     dec_sn,
    #     catalogue="lsdr10",
    #     out_dir=out_dir,
    #     radius_deg=radius_deg,
    #     dDLR_cut=dDLR_cut,
    #     type_dlr="classic",
    #     save=True,
    #     verbose=False,
    #     overwrite=False,
    # )
    possible_host_GAMA_after_cuts = fnc.get_galaxies_after_cuts(
        filename_GAMA,
        outfilename,
        id_sn,
        ra_sn,
        dec_sn,
        catalogue="GAMA",
        out_dir=out_dir,
        radius_deg=radius_deg,
        dDLR_cut=dDLR_cut,
        type_dlr="classic",
        save=True,
        verbose=False,
        overwrite=False,
    )

    possible_host_legacy_after_cuts = fnc.get_galaxies_after_cuts(
        None,
        outfilename,
        id_sn,
        ra_sn,
        dec_sn,
        catalogue="lsdr10",
        out_dir=out_dir,
        radius_deg=radius_deg,
        dDLR_cut=dDLR_cut,
        type_dlr="classic",
        save=True,
        verbose=False,
        overwrite=False,
    )
