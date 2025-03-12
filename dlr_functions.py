# Basic imports
import os
import sys

# Measure time
import time

# Import libraries
import numpy as np
import pandas as pd

# Astropy
from astropy.table import Table
from astropy import coordinates as co, units as u
from astropy.io import fits
import warnings

# Suppress warnings from Astropy
warnings.simplefilter("ignore", u.UnitsWarning)
#! Is this good practice? Another approach that I can do: # Define the custom units, u.def_unit('nanomaggy'),u.def_unit('1/nanomaggy^2'), u.def_unit('1/mas^2'), u.def_unit('1/arcsec^2')


# Legacy Survey catalogs
import pyvo


# Debugging
import ipdb  # *ipdb.set_trace()

############## TOOLBOX FUNCTIONS ################


def rotation(x, y, angle, rot_type="anticlockwise"):
    """Rotates a set of points by a given anglen in radians

    Args:
        x (array): x coordinates of the points
        y (array): y coordinates of the points
        angle (array): angle in radians to rotate the points
        rot_type (str, optional): type of rotation. Defaults to 'anticlockwise'.

    Returns:
        x_rotated, y_rotated (array): rotated x and y coordinates
    """
    # start = time.time()
    if rot_type == "anticlockwise":
        x_rotated = np.array(
            [x[i] * np.cos(angle[i]) + y[i] * np.sin(angle[i]) for i in range(len(x))]
        )
        y_rotated = np.array(
            [-x[i] * np.sin(angle[i]) + y[i] * np.cos(angle[i]) for i in range(len(x))]
        )
        return x_rotated, y_rotated

    elif rot_type == "clockwise":
        x_rotated = np.array(
            [x[i] * np.cos(angle[i]) - y[i] * np.sin(angle[i]) for i in range(len(x))]
        )
        y_rotated = np.array(
            [x[i] * np.sin(angle[i]) + y[i] * np.cos(angle[i]) for i in range(len(x))]
        )
        # end = time.time()
        # print(f"\nTime ROTATION function: {end-start} s\n")
        return x_rotated, y_rotated
    else:
        print("Rotation type not recognized, please use 'anticlockwise' or 'clockwise'")
        return None, None


def get_limits_region(ra, dec, radius_deg=1 / 60):
    """Get the limits of the region around a center point

    Args:
        ra (float64): ra of the galaxy
        dec (float64): dec of the galaxy
        radius_deg (float64): radius of the region in degrees
    Returns:
        ra_min, ra_max, dec_min, dec_max (float64): limits of the region
    """
    # start = time.time()
    cos_sn = 1  # np.cos(np.radians(dec))
    ra_min = ra - radius_deg / cos_sn
    ra_max = ra + radius_deg / cos_sn
    dec_min = dec - radius_deg
    dec_max = dec + radius_deg
    # end = time.time()
    # print(f"\nTime GET_LIMITS_REGION function: {end-start} s\n")
    return ra_min, ra_max, dec_min, dec_max


def get_limits_region_with_cos(ra, dec, radius_deg=1 / 60):
    """Get the limits of the region around a center point

    Args:
        ra (float64): ra of the galaxy
        dec (float64): dec of the galaxy
        radius_deg (float64): radius of the region in degrees
    Returns:
        ra_min, ra_max, dec_min, dec_max (float64): limits of the region
    """
    # start = time.time()
    cos_sn = np.cos(np.radians(dec))
    ra_min = ra - radius_deg / cos_sn
    ra_max = ra + radius_deg / cos_sn
    dec_min = dec - radius_deg
    dec_max = dec + radius_deg
    # end = time.time()
    # print(f"\nTime GET_LIMITS_REGION function: {end-start} s\n")
    return ra_min, ra_max, dec_min, dec_max


def get_sn_angle(
    ra_gal, dec_gal, major_axis, position_angle, ra_cen, dec_cen, verbose=False
):
    """Get the angle between the major axis of the galaxy and the SN. Needed to calculate the classic dDLR.

    Args:
        ra_gal (array): ra of the galaxy
        dec_gal (array): dec of the galaxy
        major_axis (array): major axis of the galaxy in arcsec
        position_angle (array): position angle of the galaxy in radians. In mathematicians convention, i.e. starting from the RA axis and going anti-clockwise
        ra_cen (float64): ra of the center of the region
        dec_cen (float64): dec of the center of the region
    Returns:
        angle_from_major (float64/array/pandas.core.series.Series): angle between the major axis of the galaxy and the SN
    """
    # start = time.time()
    # Convert the major axis to degrees
    major_axis = major_axis / 3600  # arcsec to deg

    # Separation in the 2D
    cos_dec_cen = np.cos(np.radians(dec_cen))
    # This is an approximation for small angles
    delta_ra = (ra_cen - ra_gal) * cos_dec_cen
    delta_dec = dec_cen - dec_gal

    angle_from_ra = np.arctan2(delta_dec, delta_ra)  # angle in radians already
    # To calculate the angle from the major axis, because of how angles move, instead of subtracting we add, they move in opposite directions.
    angle_from_major = position_angle + angle_from_ra

    # if verbose:
    #    print(f"Angles from major axis: {angle_from_major}")
    # end = time.time()
    # print(f"\nTime GET_SN_ANGLE function: {end-start} s\n")
    return angle_from_major


############## SELECTION FUNCTIONS ################


def get_index_galaxies_within_radius_GAMA(
    ra_gal, dec_gal, ra_sn, dec_sn, radius_deg=1 / 60, verbose=False
):
    """Get the indices of the galaxies within a certain radius around a supernova. The size of the region is default to 1 arcmin.

    Args:
        ra_gal (array/pandas.core.series.Series): Right ascension of the galaxies
        dec_gal (array/pandas.core.series.Series): Declination of the galaxies
        ra_sn (array len 1): Right ascension of the supernova
        dec_sn (array len 1): Declination of the supernova
        radius_deg (float64, optional): Radius in degrees around the supernova. Defaults to 1/60.
        verbose (bool, optional): Print or not. Defaults to False.

    Returns:
        ind (array): indices of the galaxies within the radius
    """
    # start = time.time()
    # Define the supernova's position
    sn_coord = co.SkyCoord(ra=ra_sn * u.deg, dec=dec_sn * u.deg)

    # Define the galaxy positions as a SkyCoord object
    gal_coords = co.SkyCoord(ra=ra_gal * u.deg, dec=dec_gal * u.deg)

    # Calculate the separation between the supernova and each galaxy
    separations = sn_coord.separation(gal_coords)
    sep = separations.degree  # Convert to degrees

    # Find indices of galaxies within the specified radius
    within_radius = sep < radius_deg

    ind = np.where(within_radius)[0]  # Get indices of the galaxies within radius

    # Count galaxies found
    num_galaxies = len(ind)
    if verbose:
        print(
            f"\nNumber of galaxies within a radius of {radius_deg*60} arcmin around the SN: {num_galaxies}"
        )

    if num_galaxies == 0:
        if verbose:
            print("No galaxies in the region")
        # end = time.time()
        # print(f"\nTime GET_INDEX_GALAXIES_WITHIN_RADIUS_GAMA function: {end-start} s\n")
        return None
    else:
        # end = time.time()
        # print(f"\nTime GET_INDEX_GALAXIES_WITHIN_RADIUS_GAMA function: {end-start} s\n")
        return ind


def table_GAMA_galaxies_within_radius(
    identifier,
    ra_cen,
    dec_cen,
    galaxy_catalogue,
    radius_deg=1 / 60,
    save=True,
    overwrite=False,
    verbose=False,
):
    """Save the galaxies within a certain radius around a supernova in a file if the user wants to.
    Return a table with all the galaxies in the region around the center.

    Args:
        identifier (str): Identifier of the supernova for the file name
        ra_cen (float64): Right ascension of the supernova or the center of the region of interest
        dec_cen (float64): Declination of the supernova or the center of the region of interest
        galaxy_catalogue (pandas.core.frame.DataFrame): DataFrame with the galaxy catalogue
        radius_deg (float64, optional): Radius in degrees around the supernova or the center of the region of interest. Defaults to 1/60.
        save(bool, optional): Saves the new file with the subsample of galaxies around the center of the region of interes. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it exists. Defaults to False.
        verbose (bool, optional): Print or not. Defaults to False.

    Returns:
        gal_table_region (astropy.table.table.Table): Table of the galaxies within the radius
    """
    # start = time.time()
    # How the files of the regions are called, depending on whether there are galaxies in it or not:
    cat_filename = f"{identifier}.gama.cat.fits"
    cat_filename_empty = f"{identifier}_EMPTY.gama.cat.fits"

    # If the file exists, just read the table:
    #! It is quicker to read the table or maybe do all the searching by index thing?
    # * Test this with timers
    if os.path.exists(cat_filename) and not overwrite:
        if verbose:
            print("reading from:", cat_filename)
        # end = time.time()
        # print(
        #    f"\nThis is the one to analyse: Time TABLE_GAMA_GALAXIES_WITHIN_RADIUS ((READING)) function: {end-start} s\n"
        # )
        return Table.read(cat_filename)
    elif os.path.exists(cat_filename_empty) and not overwrite:
        if verbose:
            print("reading from:", cat_filename_empty)
        # end = time.time()
        # print(
        #    f"\nThis is the one to analyse: Time TABLE_GAMA_GALAXIES_WITHIN_RADIUS ((EMPTY)) function: {end-start} s\n"
        # )
        return Table.read(cat_filename_empty)
    # If the file doesn't exist already, get the index of the galaxies in the region with the funtion get_index_galaxies_within_radius_GAMA
    else:
        ra_gal = np.asarray(galaxy_catalogue["RAcen"])
        dec_gal = np.asarray(galaxy_catalogue["Deccen"])

        index_region = get_index_galaxies_within_radius_GAMA(
            ra_gal, dec_gal, ra_cen, dec_cen, radius_deg=radius_deg, verbose=verbose
        )
        # If the index is None, there are no galaxies in the region, create an empty table to keep track after about the SN that doesn't have galaxies around
        if index_region is None:
            gal_table_region = Table(
                names=galaxy_catalogue.columns,
                dtype=[galaxy_catalogue[col].dtype for col in galaxy_catalogue.columns],
            )
            if save:
                gal_table_region.write(cat_filename_empty, overwrite=overwrite)
        # If there are galaxies in the region, locate them and create a Table with them
        else:
            gal_df_region = galaxy_catalogue.iloc[index_region]
            gal_table_region = Table.from_pandas(gal_df_region)
            if save:
                gal_table_region.write(cat_filename, overwrite=overwrite)
        if verbose:
            if save:
                print(f"Galaxies in the region saved to {identifier}.GAMA.fits")
                if index_region is None:
                    print(
                        "ATTENTION: empty table saved, there are no galaxies in the region"
                    )
        # end = time.time()
        # print(
        #    f"\nThis is the one to analyse: Time TABLE_GAMA_GALAXIES_WITHIN_RADIUS ((INDEX)) function: {end-start} s\n"
        # )
        return gal_table_region


def execute_query_with_retry(tap_service, ex_query, max_retries=5, wait_time=45):
    attempt = 0
    while attempt < max_retries:
        try:
            # Attempt to execute the query
            result = tap_service.search(ex_query)
            # Return the result if successful
            return result.to_table()

        except Exception as e:
            # If an error occurs, print the error and wait before retrying
            print(f"Error occurred: {e}. Retrying in {wait_time} seconds...")
            attempt += 1
            time.sleep(wait_time)  # Wait for the specified time before retrying

    # If all attempts fail, raise an error
    raise Exception(f"Failed to execute query after {max_retries} attempts.")


def table_galaxies_within_radius_Legacy(
    identifier, ra, dec, radius_deg=1 / 60, save=True, overwrite=False, verbose=False
):
    """Get the legacy galaxies within a certain radius around a supernova and save it in a file.
    Return a table with all the galaxies in the region around the center.

    Args:
        identifier (str): Identifier of the supernova for the file name
        ra (float64): Right ascension of the supernova or the center of the region of interest
        dec (float64): Declination of the supernova or the center of the region of interest
        radius_deg (float64): Radius in degrees around the supernova or the center of the region of interest
        save (bool, optional): Saves the new file with the subsample of galaxies around the center of the region of interes. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it exists. Defaults to False.
        verbose (bool, optional): Print or not. Defaults to False.

    Returns:
        tblResult (astropy.table.table.Table): Table of the galaxies within the radius
    """
    # start = time.time()
    # The file where the Table of the query will be saved:
    cat_filename = f"{identifier}.ls_dr10.cat.fits"  #! Automatise this better

    # If the file already exists, don't do the query, just read:
    if os.path.exists(cat_filename) and not overwrite:
        if verbose:
            print("reading from:", cat_filename)
        return Table.read(cat_filename)

    # If the file doesn't exist, do the query:
    else:
        tap_service = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")

        ex_query = f"""
                SELECT tr.ls_id, tr.ra, tr.dec,tr.objid, tr.ref_id, tr.gaia_phot_bp_rp_excess_factor, tr.parallax, tr.parallax_ivar,
                    tr.mag_g, tr.mag_r, tr.mag_i, tr.mag_z, tr.mag_w1, tr.mag_w2,
                    tr.flux_g, tr.flux_ivar_g, tr.flux_i, tr.flux_ivar_i, tr.flux_r, tr.flux_ivar_r,tr.flux_z, tr.flux_ivar_z, tr.flux_w1, tr.flux_ivar_w1, tr.flux_w2, tr.flux_ivar_w2,
                    tr.fracflux_g, tr.fracflux_i, tr.fracflux_r, tr.fracflux_z, tr.fracflux_w1, tr.fracflux_w2, tr.fracmasked_g, tr.fracmasked_i, tr.fracmasked_r, tr.fracmasked_z,
                    tr.sersic, tr.sersic_ivar,  tr.shape_e1, tr.shape_e1_ivar, tr.shape_e2, tr.shape_e2_ivar,
                    tr.shape_r,  tr.shape_r_ivar, tr.gaia_duplicated_source, zp.z_spec, zp.z_phot_mean, zp.z_phot_mean_i,
                    zp.z_phot_std, zp.z_phot_l68, zp.z_phot_u68, zp.z_phot_l95, zp.z_phot_u95
                FROM ls_dr10.tractor as tr LEFT JOIN ls_dr10.photo_z as zp ON tr.ls_id = zp.ls_id
                WHERE 't' = Q3C_RADIAL_QUERY(tr.ra,tr.dec,{ra:.6f},{dec:+.6f},{radius_deg:.6f})
                """
        # result = tap_service.search(ex_query)
        # Transform the result to a Table
        # tblResult = result.to_table()

        tblResult = execute_query_with_retry(tap_service, ex_query)

        if verbose:
            print(f"querying legacy with a search radius of {radius_deg*60} arcmin.")
            # print(f"query result has {len(tblResult)} rows.")
            if save:
                print("writing query result to:", cat_filename)
        if save:
            tblResult.write(cat_filename, overwrite=overwrite)
        # end = time.time()
        # print(f"\nTime TABLE_GALAXIES_WITHIN_RADIUS_LEGACY function: {end-start} s\n")
        return tblResult


############## GET DLR FUNCTIONS ################


def get_dDLR_rotated(
    ra_gal,
    dec_gal,
    galaxy_angle_rad,
    galaxy_major,
    galaxy_minor,
    ra_sn,
    dec_sn,
    catalogue,
    verbose=False,
):
    """Calculates the dDLR between a galaxy and a SN without worrying about the angle between the galaxy and the SN

    Args:
        ra_gal (array): Right ascension of the galaxy
        dec_gal (array): Declination of the galaxy
        galaxy_angle_rad (array): angle of the galaxy in radians
        galaxy_major (array): semi-major axis of the galaxy
        galaxy_minor (array): semi-minor axis of the galaxy
        ra_sn (float64): Right ascension of the SN
        dec_sn (float64): Declination of the SN
        catalogue (str): catalogue of the galaxy. Either 'GAMA' or 'lsdr10'
        verbose (bool, optional): Print or not. Defaults to False.

    Returns:
        dDLR (array): dDLR between the galaxy and the SN calculated with the rotated method
    """
    # start = time.time()

    # Define the coordinates in SkyCoord to use Astropy functions:
    gal_radec = co.SkyCoord(ra_gal, dec_gal, unit="deg")
    sn_radec = co.SkyCoord(ra_sn, dec_sn, unit="deg")

    # Distance between the two sets of coordinates. Distance in the coordinates of RA and Dec
    alpha, delta = sn_radec.spherical_offsets_to(gal_radec)

    # Choose to work in units of arcsec for convenience
    dx, dy = alpha.arcsec, delta.arcsec

    # Rotate these dx/dy to match the galaxy angle. Distance in the coordinates of the semi-major and semi-minor axis of the galaxy:
    if catalogue == "GAMA":
        rot_type = "clockwise"
    elif catalogue == "lsdr10":
        rot_type = "clockwise"
    else:
        print("Catalogue not recognised, please use 'GAMA' or 'lsdr10'")
        sys.exit()

    da, db = rotation(dx, dy, galaxy_angle_rad, rot_type=rot_type)

    # da and db are separations in arcsec in the coord system of the semi-major and semi-minor axes.

    dDLR = np.sqrt((da / galaxy_major) ** 2.0 + (db / galaxy_minor) ** 2.0)

    # if verbose:
    #    print(f"Rotated dDLR: {dDLR}")
    # end = time.time()
    # print(f"\nTime GET_DDLR_ROTATED function: {end-start} s\n")
    return dDLR


def get_dDLR_classic(
    ra_gal,
    dec_gal,
    position_angle_gal,
    major_gal,
    minor_gal,
    ra_sn,
    dec_sn,
    verbose=False,
):
    """Get the dDLR with the classic method

    Args:
        ra_gal (array): Right ascension of the galaxy
        dec_gal (array): Declination of the galaxy
        position_angle_gal (array): position angle of the galaxy in radians. In mathematicians convention, i.e. starting from the RA axis and going anti-clockwise
        major_gal (array): semi-major axis of the galaxy
        minor_gal (array): semi-minor axis of the galaxy
        ra_sn (float64): Right ascension of the SN
        dec_sn (float64): Declination of the SN
        verbose (bool, optional): Print or not. Defaults to False.

    Returns:
        dDLR (array): dDLR between the galaxy and the SN calculated with the classic method
    """
    # start = time.time()
    # Define the coordinates in SkyCoord to use Astropy functions:
    gal_radec = co.SkyCoord(ra_gal, dec_gal, unit="deg")
    sn_radec = co.SkyCoord(ra_sn, dec_sn, unit="deg")

    # Separation between the galaxy and the SN
    sep = gal_radec.separation(sn_radec).arcsec

    # Get the angle between the major axis of the galaxy and the SN
    phi = get_sn_angle(
        ra_gal,
        dec_gal,
        major_gal,
        position_angle_gal,
        ra_sn,
        dec_sn,
        verbose=verbose,
    )

    # Calculate the dDLR with the classic dDLR formula
    DLR = (major_gal * minor_gal) / (
        np.sqrt((major_gal * np.sin(phi)) ** 2 + (minor_gal * np.cos(phi)) ** 2)
    )

    dDLR = sep / DLR

    # if verbose:
    #    print(f"Classic dDLR: {dDLR}")
    # end = time.time()
    # print(f"\nTime GET_DDLR_CLASSIC function: {end-start} s\n")
    return dDLR


############## GET THE POSSIBLE HOSTS ################


def get_possible_hosts(
    ra_gal,
    dec_gal,
    major_gal,
    minor_gal,
    position_angle_gal,
    ra_sn,
    dec_sn,
    catalogue,
    dDLR_cut=4,
    type_dlr="classic",
    verbose=False,
):
    """Get the possible hosts of a SN

    Args:
        ra_gal (array): Right ascension of the galaxy
        dec_gal (array): Declination of the galaxy
        major_gal (array): semi-major axis of the galaxy
        minor_gal (array): semi-minor axis of the galaxy
        position_angle_gal (array): position angle of the galaxy in radians. In mathematicians convention, i.e. starting from the RA axis and going anti-clockwise
        ra_sn (float64): Right ascension of the SN
        dec_sn (float64): Declination of the SN
        catalogue (str): catalogue of the galaxy. Either 'GAMA' or 'lsdr10'
        dDLR_cut (int, optional): dDLR cut to consider a galaxy a possible host. Defaults to 4.
        type_dlr (str, optional): type of dDLR calculation. Either 'classic' or 'rotated'. Defaults to 'classic'.
        verbose (bool, optional): Print or not. Defaults to False.
    """
    # start = time.time()
    # Calculate the dDLR depending on the method chosen
    if type_dlr == "classic":
        dDLR = get_dDLR_classic(
            ra_gal,
            dec_gal,
            position_angle_gal,
            major_gal,
            minor_gal,
            ra_sn,
            dec_sn,
            verbose=verbose,
        )
    elif type_dlr == "rotated":
        dDLR = get_dDLR_rotated(
            ra_gal,
            dec_gal,
            position_angle_gal,
            major_gal,
            minor_gal,
            ra_sn,
            dec_sn,
            catalogue=catalogue,
            verbose=verbose,
        )
    else:
        print("Type of dDLR not recognised, please use 'classic' or 'rotated'")
        sys.exit()

    # Get the indices of the possible hosts and return them and their dDLR
    possible_hosts = np.where(dDLR < dDLR_cut)[0]

    if len(possible_hosts) == 0:
        #! If I want to keep track of the SN that doesn't have a galaxy with dDLR<4 but have a closest galaxy in the region
        # if len(dDLR) == 0:
        #     # print("No galaxies in the region")
        #     return possible_hosts, -999
        # else:
        #     print(dDLR)
        #     closest_galaxy = np.argmin(dDLR)
        #     print(closest_galaxy, dDLR[closest_galaxy])
        #     return [closest_galaxy], dDLR[closest_galaxy]
        # return closest_galaxy, dDLR[closest_galaxy]
        #! #########################################################
        # if verbose:
        #    print("No possible hosts")
        # end = time.time()
        # print(f"\nTime GET_POSSIBLE_HOSTS function: {end-start} s\n")
        return possible_hosts, -999
    else:
        # end = time.time()
        # print(f"\nTime GET_POSSIBLE_HOSTS function: {end-start} s\n")
        return possible_hosts, dDLR[possible_hosts]


########## BIG LOOP ###########


def get_possible_hosts_loop(
    filename,
    outfilename,
    id_sn,
    ra_sn,
    dec_sn,
    catalogue,
    out_dir="output_files/",
    radius_deg=1 / 60,
    dDLR_cut=4,
    type_dlr="classic",
    save=True,
    verbose=False,
    overwrite=False,
):
    """Get all the possible hosts of a list of supernovae and save them to a file

    Args:
        filename (path): path of the input file with the galaxy catalogue. For the moment only needed for GAMA, for legacy it is not needed, put None
        outfilename (str): name of the output file
        id_sn (array): ID of the supernovae
        ra_sn (array): Right ascension of the supernovae
        dec_sn (array): Declination of the supernovae
        catalogue (str): catalogue of the galaxy. Either 'GAMA' or 'lsdr10'
        radius_deg (float64, optional): radius in degrees around the supernova. Defaults to 1/60, i.e. 1 arcmin.
        dDLR_cut (int, optional): dDLR cut to consider a galaxy a possible host. Defaults to 4.
        type_dlr (str, optional): type of dDLR calculation. Either 'classic' or 'rotated'. Defaults to 'classic'.
        save (bool, optional): whether to save the output to a file. Defaults to True.
        verbose (bool, optional): Print or not. Defaults to False.
        overwrite (bool, optional): Overwrite the file if it exists. Defaults to False.

    Returns:
        big_df (pandas.core.frame.DataFrame): DataFrame with all the possible hosts of the supernovae
    """
    # start = time.time()
    # Create an empty list to append the DataFrames of the possible hosts
    data_hosts = []

    dirout = f"{out_dir}/{catalogue}/possible_hosts/"
    if not os.path.exists(dirout):  # Check if the directory does not exist
        os.makedirs(dirout, exist_ok=True)
        print(f"Output directories created: {dirout}")
    else:
        print(f"Output directories already exists: {dirout}")

    reg_arcmin = int(radius_deg * 2 * 60)
    if save:
        # Check and create directories
        dir_path = f"{out_dir}/{catalogue}/{reg_arcmin}arcmin_regions/"

        if not os.path.exists(dir_path):  # Check if the directory does not exist
            os.makedirs(dir_path, exist_ok=True)
            print(f"Output directories for subcatalogues created: {dir_path}")
        else:
            print(f"Output directories for subcatalogues already exists: {dir_path}")

    # The path of the files with the regions of the supernovae
    ident = [f"{out_dir}/{catalogue}/{reg_arcmin}arcmin_regions/{sn}" for sn in id_sn]

    # The path of the output file
    foutput = f"{out_dir}/{catalogue}/possible_hosts/{catalogue}_{outfilename}"

    # Check if the output file already exists
    if os.path.exists(foutput) and not overwrite:
        print(
            f"Output file {foutput} already exists. Skipping calculations and loading the existing file."
        )
        # Load the existing file and return it as a DataFrame
        existing_table = Table.read(foutput)
        return existing_table.to_pandas()

    # Load the catalogue ONCE outside the loop
    if catalogue == "GAMA":
        # start_reading = time.time()
        columns_to_read = [
            "uberID",
            "RAcen",
            "Deccen",
            "R50",
            "axrat",
            "ang",
            "duplicate",
            "starmask",
            "CATAID",
            "Z",
        ]
        # Open the FITS file and access the data
        with fits.open(filename) as hdul:
            file = hdul[1].data  # Assuming the table is in the first extension
            selected_data = {col: file[col] for col in columns_to_read}

        # Convert the selected columns to an Astropy Table
        galaxy_catalogue = Table(selected_data)

        # Convert to pandas DataFrame if needed
        gal_df = galaxy_catalogue.to_pandas()
        # galaxy_catalogue = Table.read(filename)
        # gal_df = galaxy_catalogue.to_pandas()
        # end_reading = time.time()
        # print(
        #    f"\nTime READING THE GAMA CATALOGUE ONCE: {end_reading-start_reading} s\n"
        # )
    # Loop over the supernovae
    for i, ira_sn in enumerate(pd.unique(ra_sn)):
        # Get the information of the catalogues in the appropiate format
        # Needed: ra_gal, dec_gal, major_gal, minor_gal, position_angle_gal
        if catalogue == "GAMA":
            # start_gama = time.time()
            table_gal = table_GAMA_galaxies_within_radius(
                ident[i],
                ira_sn,
                dec_sn[i],
                galaxy_catalogue=gal_df,
                radius_deg=radius_deg,
                save=save,
                overwrite=False,
                verbose=verbose,
            )
            table_df = table_gal.to_pandas()
            table_gal_df = table_df[table_df["duplicate"] == 0]
            ra_gal = np.asarray(table_gal_df["RAcen"])
            dec_gal = np.asarray(table_gal_df["Deccen"])
            major_gal = np.asarray(table_gal_df["R50"])  # In arcsec
            axrat = np.asarray(table_gal_df["axrat"])
            minor_gal = major_gal * axrat
            position_angle = table_gal_df["ang"]
            position_angle_gal = np.radians(np.asarray(position_angle) - 90)
            # end_gama = time.time()
            # print("Time GAMA parameters:", end_gama - start_gama)

        elif catalogue == "lsdr10":
            # start_legacy = time.time()
            table_gal = table_galaxies_within_radius_Legacy(
                ident[i],
                ira_sn,
                dec_sn[i],
                radius_deg=radius_deg,
                save=save,
                overwrite=False,
                verbose=verbose,
            )
            table_gal_df = table_gal.to_pandas()
            ra_gal = np.asarray(table_gal_df["ra"])
            dec_gal = np.asarray(table_gal_df["dec"])
            shape_r = np.asarray(table_gal_df["shape_r"])
            shape_e1 = table_gal_df["shape_e1"]
            shape_e2 = table_gal_df["shape_e2"]

            ellipticity = np.sqrt(shape_e1**2 + shape_e2**2)
            position_angle = 0.5 * np.arctan2(shape_e2, shape_e1)
            position_angle_gal = np.asarray(position_angle) - np.pi / 2
            axrat = (1 - ellipticity) / (1 + ellipticity)
            major_gal = shape_r
            minor_gal = major_gal * axrat
            # end_legacy = time.time()
            # print("Time LEGACY parameters:", end_legacy - start_legacy)

        else:
            print("Catalogue not recognised, please use 'GAMA' or 'lsdr10'")
            sys.exit()
        # Get the index of the possible hosts and their dDLR
        ind_possible_hosts, dDLR_hosts = get_possible_hosts(
            ra_gal,
            dec_gal,
            major_gal,
            minor_gal,
            position_angle_gal,
            ra_sn[i],
            dec_sn[i],
            catalogue,
            dDLR_cut=dDLR_cut,
            type_dlr=type_dlr,
            verbose=verbose,
        )

        # Create indice with the galaxies in the region
        galaxies_in_region = len(ra_gal) > 0

        # If there are no possible hosts, create a DataFrame with the information of the SN
        if len(ind_possible_hosts) == 0:
            # Create a DataFrame with one row, populated only with `ra_sn` and `dec_sn`
            df_possible_hosts = pd.DataFrame(
                [[-999] * len(table_gal_df.columns)], columns=table_gal_df.columns
            )
            df_possible_hosts["sn_name"] = id_sn[i]
            df_possible_hosts["ra_sn"] = ra_sn[i]
            df_possible_hosts["dec_sn"] = dec_sn[i]
            df_possible_hosts["dDLR"] = dDLR_hosts
            df_possible_hosts["galaxies_in_region"] = galaxies_in_region
            df_possible_hosts["multiple_matches"] = False
            df_possible_hosts["top_match_bc"] = False  # No match, so no top match

        # If there is only one possible host, create a DataFrame with the information of the galaxy near the SN
        elif len(ind_possible_hosts) == 1:
            # Extract possible hosts
            df_possible_hosts = table_gal_df.iloc[ind_possible_hosts].copy()
            df_possible_hosts["sn_name"] = id_sn[i]
            df_possible_hosts["ra_sn"] = ra_sn[i]
            df_possible_hosts["dec_sn"] = dec_sn[i]
            df_possible_hosts["dDLR"] = dDLR_hosts
            df_possible_hosts["galaxies_in_region"] = galaxies_in_region
            df_possible_hosts["multiple_matches"] = False
            df_possible_hosts["top_match_bc"] = (
                True  # Only one match, so it is the top match
            )
        # If there are multiple possible hosts, create a DataFrame with the information of all the galaxies near the SN
        else:
            # Extract possible hosts
            df_possible_hosts = table_gal_df.iloc[ind_possible_hosts].copy()
            df_possible_hosts["sn_name"] = [id_sn[i]] * len(df_possible_hosts)
            df_possible_hosts["ra_sn"] = [ra_sn[i]] * len(df_possible_hosts)
            df_possible_hosts["dec_sn"] = [dec_sn[i]] * len(df_possible_hosts)
            df_possible_hosts["dDLR"] = dDLR_hosts
            df_possible_hosts["galaxies_in_region"] = [galaxies_in_region] * len(
                df_possible_hosts
            )
            df_possible_hosts["multiple_matches"] = True
            # Set the top match
            min_dDLR = min(dDLR_hosts)
            df_possible_hosts["top_match_bc"] = df_possible_hosts["dDLR"] == min_dDLR

        # Reorder columns to place the data of the SN, the dDLR and the flags regarding the matches at the beginning
        columns_order = [
            "sn_name",
            "ra_sn",
            "dec_sn",
            "galaxies_in_region",
            "dDLR",
            "multiple_matches",
            "top_match_bc",
        ] + [
            col
            for col in df_possible_hosts.columns
            if col
            not in [
                "sn_name",
                "ra_sn",
                "dec_sn",
                "galaxies_in_region",
                "dDLR",
                "multiple_matches",
                "top_match_bc",
            ]
        ]

        df_possible_hosts = df_possible_hosts[columns_order]
        # Append to the data list
        data_hosts.append(df_possible_hosts)

    # Concatenate all the DataFrames in the list to create a big DataFrame
    big_df = pd.concat(data_hosts, ignore_index=True)

    # Save the big DataFrame to a file that can be .fits or .csv

    big_table = Table.from_pandas(big_df)
    big_table.write(foutput, overwrite=overwrite)
    print(f"Possible hosts saved to {foutput}")
    # end = time.time()
    # print(f"\nTime GET_POSSIBLE_HOSTS_LOOP function: {end-start} s\n")
    return big_df


def get_galaxies_after_cuts(
    filename,
    outfilename,
    id_sn,
    ra_sn,
    dec_sn,
    catalogue,
    out_dir="output_files/",
    radius_deg=1 / 60,
    dDLR_cut=4,
    type_dlr="classic",
    save=True,
    verbose=False,
    overwrite=False,
):
    """Get all the possible hosts of a list of supernovae and save them to a file

    Args:
        filename (path): path of the input file with the galaxy catalogue. For the moment only needed for GAMA, for legacy it is not needed, put None
        id_sn (array): ID of the supernovae
        ra_sn (array): Right ascension of the supernovae
        dec_sn (array): Declination of the supernovae
        catalogue (str): catalogue of the galaxy. Either 'GAMA' or 'lsdr10'
        radius_deg (float64, optional): radius in degrees around the supernova. Defaults to 1/60, i.e. 1 arcmin, 2 arcmin region.
        dDLR_cut (int, optional): dDLR cut to consider a galaxy a possible host. Defaults to 4.
        type_dlr (str, optional): type of dDLR calculation. Either 'classic' or 'rotated'. Defaults to 'classic'.
        save (bool, optional): whether to save the output to a file. Defaults to True.
        verbose (bool, optional): Print or not. Defaults to False.
        overwrite (bool, optional): Overwrite the file if it exists. Defaults to False.

    Returns:
        big_df (pandas.core.frame.DataFrame): DataFrame with all the possible hosts of the supernovae
    """

    gal_df = get_possible_hosts_loop(
        filename=filename,
        outfilename=outfilename,
        id_sn=id_sn,
        ra_sn=ra_sn,
        dec_sn=dec_sn,
        catalogue=catalogue,
        out_dir=out_dir,
        radius_deg=radius_deg,
        dDLR_cut=dDLR_cut,
        type_dlr=type_dlr,
        save=save,
        verbose=verbose,
        overwrite=overwrite,
    )

    # multiple_hosts = gal_df[gal_df["multiple_matches"] == "True"].copy()

    if catalogue == "GAMA":
        gal_df["cut_match"] = False
        # Start with the same values before cuts and after cuts
        gal_df["top_match_ac"] = gal_df["top_match_bc"]

        # If there are multiple matches and the galaxy has the starmask cut the galaxy:
        gal_df.loc[
            (gal_df["multiple_matches"] == "True") & (gal_df["starmask"] == 1),
            "cut_match",
        ] = True

        # If there are multiple matches and the galaxy is a duplicate cut the galaxy:
        gal_df.loc[
            (gal_df["multiple_matches"] == "True") & (gal_df["duplicate"] == 1),
            "cut_match",
        ] = True

        # If we decided to cut the galaxy, set top_match_ac to False
        gal_df.loc[
            (gal_df["cut_match"] == True) & (gal_df["top_match_bc"] == "True"),
            "top_match_ac",
        ] = False

        filtered = gal_df[(gal_df["cut_match"] == False) & (gal_df["dDLR"] > 0)]

        # Find the index of the rows with the minimum dDLR for each `sn_name`
        min_indices = filtered.loc[filtered.groupby("sn_name")["dDLR"].idxmin()].index

        # Create the `top_match_ac` column
        gal_df["top_match_ac"] = gal_df.index.isin(min_indices)

        # Reorder columns to place the data of the SN, the dDLR and the flags regarding the matches at the beginning
        columns_order = [
            "sn_name",
            "ra_sn",
            "dec_sn",
            "galaxies_in_region",
            "dDLR",
            "multiple_matches",
            "top_match_bc",
            "cut_match",
            "top_match_ac",
        ] + [
            col
            for col in gal_df.columns
            if col
            not in [
                "sn_name",
                "ra_sn",
                "dec_sn",
                "galaxies_in_region",
                "dDLR",
                "multiple_matches",
                "top_match_bc",
                "cut_match",
                "top_match_ac",
            ]
        ]
        gal_df = gal_df[columns_order]

    elif catalogue == "lsdr10":
        gal_df["cut_match_photo"] = False
        gal_df["cut_match_fracflux"] = True
        gal_df["top_match_ac"] = gal_df["top_match_bc"]

        # Photometry cuts:

        # If there are multiple matches and the galaxy has the excess_factor between 0 and 2.5 cut the galaxy:
        # gal_df.loc[
        #     (gal_df["multiple_matches"] == "True")
        #     & (
        #         (gal_df["gaia_phot_bp_rp_excess_factor"] < 0)
        #         | (
        #             (gal_df["gaia_phot_bp_rp_excess_factor"] > 0)
        #             & (gal_df["gaia_phot_bp_rp_excess_factor"] < 2.5)
        #         )
        #     ),
        #     "cut_match_photo",
        # ] = True
        gal_df.loc[
            gal_df["gaia_phot_bp_rp_excess_factor"].between(0, 2.5, inclusive="right")
            | (gal_df["gaia_phot_bp_rp_excess_factor"] < 0),
            "cut_match_photo",
        ] = True

        # We decided not to cut in parallax since it is already included IN GENERAL in the excess factor cut
        # error_parallax = 1 / np.sqrt(gal_df["parallax_ivar"])
        # SNR_parallax = gal_df["parallax"] / error_parallax
        # # If there are multiple matches and the galaxy has the parallax cut the galaxy:
        # gal_df.loc[
        #     (gal_df["multiple_matches"] == "True")
        #     & (SNR_parallax > 0)
        #     & (SNR_parallax < 3),
        #     "cut_match_photo",
        # ] = True

        # Define fracflux_cols and apply the condition
        fracflux_cols = ["fracflux_g", "fracflux_i", "fracflux_r", "fracflux_z"]
        # fracflux_not_cut = gal_df[fracflux_cols].apply(
        #     lambda row: any((row > 0) & (row < 1)), axis=1
        # )
        #! I just changed this, I hope I don't regret it!!
        fracflux_not_cut = gal_df[fracflux_cols].apply(
            lambda row: any((row > 0) & (row < 1)) or all(row <= 0), axis=1
        )

        # Update cut_match_fracflux based on the condition
        gal_df.loc[fracflux_not_cut, "cut_match_fracflux"] = False

        # Not cutting in SNR_flux, since we saw it is not necessary

        # gal_df["cut_SNR"] = True

        # # Define flux and flux_ivar columns
        # flux_cols = ["flux_g", "flux_i", "flux_r", "flux_z"]
        # flux_ivar_cols = ["flux_ivar_g", "flux_ivar_i", "flux_ivar_r", "flux_ivar_z"]
        # SNR_g = gal_df["flux_g"] * (gal_df["flux_ivar_g"] ** 0.5)
        # SNR_i = gal_df["flux_i"] * (gal_df["flux_ivar_i"] ** 0.5)
        # SNR_r = gal_df["flux_r"] * (gal_df["flux_ivar_r"] ** 0.5)
        # SNR_z = gal_df["flux_z"] * (gal_df["flux_ivar_z"] ** 0.5)

        # # Calculate SNR and apply the condition SNR > 4
        # snr_not_cut = gal_df.apply(
        #     lambda row: any(
        #         (
        #             (row[flux] / (row[ivar] ** -0.5) > 4) if row[ivar] > 0 else True
        #         )  # Avoid division by zero
        #         for flux, ivar in zip(flux_cols, flux_ivar_cols)
        #     ),
        #     axis=1,
        # )

        # # Update `cut_SNR` column based on the condition
        # gal_df.loc[snr_not_cut, "cut_SNR"] = False

        # Filter rows where both `cut_match_photo` and `cut_match_fracflux` are False
        filtered = gal_df[
            (gal_df["cut_match_photo"] == False)
            & (gal_df["cut_match_fracflux"] == False)
            & (gal_df["dDLR"] > 0)
        ]

        # Find the index of the rows with the minimum dDLR for each `sn_name`
        min_indices = filtered.loc[filtered.groupby("sn_name")["dDLR"].idxmin()].index

        # Create the `top_match_ac` column
        gal_df["top_match_ac"] = gal_df.index.isin(min_indices)

        # Reorder columns to place the data of the SN, the dDLR and the flags regarding the matches at the beginning
        columns_order = [
            "sn_name",
            "ra_sn",
            "dec_sn",
            "galaxies_in_region",
            "dDLR",
            "multiple_matches",
            "top_match_bc",
            "cut_match_photo",
            "cut_match_fracflux",
            "top_match_ac",
        ] + [
            col
            for col in gal_df.columns
            if col
            not in [
                "sn_name",
                "ra_sn",
                "dec_sn",
                "galaxies_in_region",
                "dDLR",
                "multiple_matches",
                "top_match_bc",
                "cut_match_photo",
                "cut_match_fracflux",
                "top_match_ac",
            ]
        ]
        gal_df = gal_df[columns_order]

    # Save the final table
    gal_table = Table.from_pandas(gal_df)
    gal_table.write(
        f"{out_dir}/{catalogue}/possible_hosts/{catalogue}_after_cuts_{outfilename}",
        overwrite=True,
    )

    return gal_table.to_pandas()
