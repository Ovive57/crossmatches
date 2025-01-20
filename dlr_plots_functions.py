# Legacy survey images
import urllib
from PIL import Image
import requests

# Basic imports
import numpy as np
import random
from pathlib import Path
import sys

# Astropy imports
from astropy.table import Table
import pandas as pd
import astropy.coordinates as co


# Plots
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm


## DLR imports

import dlr_functions as fnc  # rotation, get_limits_region


def draw_ellipses(
    ra_gal,
    dec_gal,
    ra_sn,
    dec_sn,
    major_axis,
    minor_axis,
    position_angle,
    ax,
    catalogue,
    scale_factor=[1, 4],
    rot_type="anticlockwise",
    colours=["red", "blue"],
    dDLR_bar=False,
    label=None,
):
    """Draws an ellipse with the DLR of a galaxy

    Args:
        scale_factor (list shape (2,)): Scale factor of the 2 ellipses in an array, how big they are going to be #! make a default value later
        major_axis (float64): Major axis of the galaxy. In arcsec
        minor_axis (float64): Minor axis of the galaxy. In arcsec
        position_angle (float32): Position angle of the galaxy in radians in mathematicians convention, i.e. starting from the RA axis and going anti-clockwise
        ra_gal (float64): ra of the galaxy
        dec_gal (float64): dec of the galaxy
    Returns:
        ax: the axis of the plot
    """

    xi_deg = np.linspace(0, 360, 1000)
    xi_rad = np.radians(xi_deg)

    # Convert to array:
    major_axis_arr = np.array(major_axis) / 3600  # arcsec to deg
    minor_axis_arr = np.array(minor_axis) / 3600  # arcsec to deg

    if len(major_axis_arr) == 0:
        print(f"No major axis, no ellipse to plot")
        return None

    if len(ra_gal) == 1:
        if ra_gal.iloc[0] == -999.0:
            print(f"No galaxy found, no ellipse to plot")
            return None

    if catalogue == "GAMA":
        position_angle_arr = np.radians(np.array(position_angle) - 90)
    elif catalogue == "lsdr10":
        position_angle_arr = np.array(position_angle) - np.pi / 2
    else:
        position_angle_arr = np.array(position_angle)
    ra_gal_arr = np.array(ra_gal)
    dec_gal_arr = np.array(dec_gal)

    ### PLOTTING WITH THE DLR DIRECTLY
    DLR_ellipse = [
        major_axis_arr[i]
        * minor_axis_arr[i]
        / np.sqrt(
            (major_axis_arr[i] * np.sin(xi_rad)) ** 2
            + (minor_axis_arr[i] * np.cos(xi_rad)) ** 2
        )
        for i in range(len(major_axis_arr))
    ]

    dDLR = fnc.get_dDLR_classic(
        ra_gal,
        dec_gal,
        position_angle,
        major_axis,
        minor_axis,
        ra_sn,
        dec_sn,
    )

    # Adjust the radius for d = 1 and d = 4
    r_1 = [scale_factor[0] * DLR_ellipse[i] for i in range(len(DLR_ellipse))]
    r_2 = [scale_factor[1] * DLR_ellipse[i] for i in range(len(DLR_ellipse))]

    x_1 = r_1 * np.cos(xi_rad)
    y_1 = r_1 * np.sin(xi_rad)

    x_2 = r_2 * np.cos(xi_rad)
    y_2 = r_2 * np.sin(xi_rad)

    x1_rotated, y1_rotated = fnc.rotation(
        x_1, y_1, position_angle_arr, rot_type=rot_type
    )
    x1_rotated = np.array(
        [x1_rotated[i] + ra_gal_arr[i] for i in range(len(x1_rotated))]
    )
    y1_rotated = np.array(
        [y1_rotated[i] + dec_gal_arr[i] for i in range(len(y1_rotated))]
    )

    x2_rotated, y2_rotated = fnc.rotation(
        x_2, y_2, position_angle_arr, rot_type=rot_type
    )
    x2_rotated = np.array(
        [x2_rotated[i] + ra_gal_arr[i] for i in range(len(x2_rotated))]
    )
    y2_rotated = np.array(
        [y2_rotated[i] + dec_gal_arr[i] for i in range(len(y2_rotated))]
    )

    if dDLR_bar:
        # Filter indices where dDLR < 4
        # dDLR = dDLR.reset_index(drop=True)
        indices = [i for i in range(len(dDLR)) if dDLR.iloc[i] < 4]

        # Normalize dDLR for the colormap with range [0, 4]
        norm = plt.Normalize(vmin=0, vmax=4)
        cmap = cm.ScalarMappable(norm=norm, cmap="rainbow")
        [
            ax.plot(
                x2_rotated[i],
                y2_rotated[i],
                color=cmap.to_rgba(dDLR.iloc[i]),
                linestyle="--",
                label=label,
                zorder=100,
            )
            for i in indices
        ]
        # Add the colorbar
        cbar = ax.figure.colorbar(cmap, ax=ax, orientation="vertical")
        cbar.set_label("dDLR", fontsize=12)  # Label for the colorbar
    else:
        [
            ax.plot(
                x1_rotated[i],
                y1_rotated[i],
                color=colours[0],
                linestyle="-",
                label=label,
                zorder=100,
            )
            for i in range(len(x1_rotated))
        ]
        [
            ax.plot(
                x2_rotated[i],
                y2_rotated[i],
                color=colours[1],
                linestyle="--",
                label=label,
                zorder=100,
            )
            for i in range(len(x2_rotated))
        ]

    return ax


def get_jpeg_cutout(size_arcmin, name, ra, dec, pixscale=0.262, save=False):
    #! pixscale = 0.262 # this is the LS DR10 standard; i.e. no resampling
    filename = f"{name}.jpeg"
    num_pixels = int((size_arcmin * 60.0) / pixscale) + 1
    # imagequery = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&size={num_pixels}&pixscale={pixscale}&layer=ls-dr10"
    imagequery = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&size={num_pixels}&pixscale={pixscale}&layer=ls-dr10"
    # imagequery = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer=ls-dr9&pixscale={pixscale}&width={num_pixels}&height={num_pixels}"

    # and here is actually issuing the request and opening image
    im = Image.open(requests.get(imagequery, stream=True).raw)

    # NB. this does not actually save the image to disk
    # but this does!
    if save:
        im.save(filename)
    return im


def plot_jpeg_cutout(filename, ra, dec, radius_deg, ax):
    filename_jpeg = f"{filename}.jpeg"
    if Path(filename_jpeg).exists():
        img = mpimg.imread(filename_jpeg)
    else:
        img = get_jpeg_cutout(2 * radius_deg * 60, filename, ra, dec)

    min_RA, max_RA, min_Dec, max_Dec = fnc.get_limits_region(
        ra, dec, radius_deg=radius_deg
    )
    ax.imshow(img, extent=[max_RA, min_RA, max_Dec, min_Dec])
    ax.plot(
        ra,
        dec,
        "+",
        color="red",
        markersize=10,
        markeredgewidth=3,
        label=f"SN {filename}",
    )
    return ax


def loop_ellipses_with_image(
    id_sn,
    ra_sn,
    dec_sn,
    catalogue,
    radius_deg,
    rectangle=False,
    rect_width=15.3 / 3600,
    scale_factor=[1, 4],
    colours=["red", "blue"],
    dDLR_bar=False,
    rot_type="anticlockwise",
    filename_GAMA=None,
    filename_legacy=None,
    save=False,
    overwrite=False,
    out_dir=None,
):

    if catalogue == "GAMA":
        galaxies_table_GAMA = Table.read(filename_GAMA)
        galaxies_GAMA_df = galaxies_table_GAMA.to_pandas()
    elif catalogue == "lsdr10":
        if filename_legacy != None:
            galaxies_table_legacy = Table.read(filename_legacy)
            galaxies_legacy_df_all = galaxies_table_legacy.to_pandas()

    for i, sn_name in enumerate(id_sn):
        fig_path = (
            f"{out_dir}/{id_sn[i]}_{catalogue}_ellipses_{radius_deg*60}arcmin.jpeg"
        )
        if Path(fig_path).exists() and not overwrite:
            print(f"File already exists in {fig_path}")
            continue
        min_RA, max_RA, min_Dec, max_Dec = fnc.get_limits_region(
            ra_sn[i], dec_sn[i], radius_deg
        )
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set(
            xlabel=("RA (deg)"),
            ylabel=("Dec (deg)"),
            xlim=(max_RA, min_RA),
            ylim=(min_Dec, max_Dec),
        )

        if catalogue == "GAMA":
            region_GAMA_table = fnc.table_GAMA_galaxies_within_radius(
                id_sn[i],
                ra_sn[i],
                dec_sn[i],
                galaxies_GAMA_df,
                radius_deg=radius_deg,
                save=False,
                overwrite=False,
                verbose=True,
            )

            galaxies_GAMA_region_df = region_GAMA_table.to_pandas()
            ra_gal = galaxies_GAMA_region_df["RAcen"]
            dec_gal = galaxies_GAMA_region_df["Deccen"]

            axrat = galaxies_GAMA_region_df["axrat"]

            major_axis = galaxies_GAMA_region_df["R50"]  # In arcsec
            minor_axis = major_axis * axrat
            position_angle = galaxies_GAMA_region_df["ang"]

        elif catalogue == "lsdr10":
            if filename_legacy == None:
                galaxies_table_legacy = fnc.table_galaxies_within_radius_Legacy(
                    id_sn[i],
                    ra_sn[i],
                    dec_sn[i],
                    radius_deg,
                    save=False,
                    overwrite=False,
                    verbose=True,
                )
                galaxies_legacy_df = galaxies_table_legacy.to_pandas()

            else:
                galaxies_legacy_df = galaxies_legacy_df_all[
                    galaxies_legacy_df_all["sn_name"] == sn_name
                ]

            ra_gal = galaxies_legacy_df["ra"]
            dec_gal = galaxies_legacy_df["dec"]

            shape_r = galaxies_legacy_df["shape_r"]
            shape_e1 = galaxies_legacy_df["shape_e1"]
            shape_e2 = galaxies_legacy_df["shape_e2"]
            ellipticity = np.sqrt(shape_e1**2 + shape_e2**2)
            axrat = (1 - ellipticity) / (1 + ellipticity)

            major_axis = shape_r
            minor_axis = major_axis * axrat
            position_angle = 0.5 * np.arctan2(shape_e2, shape_e1)

        else:
            print("Catalogue not recognised, please use 'GAMA' or 'lsdr10'")
            sys.exit()
        if len(ra_gal) == 1 and ra_gal.iloc[0] == -999.0:
            print(f"No legacy image downloaded")
        else:
            plot_jpeg_cutout(
                id_sn[i],
                ra_sn[i],
                dec_sn[i],
                radius_deg,
                ax,
            )

            draw_ellipses(
                ra_gal,
                dec_gal,
                ra_sn[i],
                dec_sn[i],
                major_axis,
                minor_axis,
                position_angle,
                ax,
                catalogue,
                scale_factor=scale_factor,
                rot_type=rot_type,
                colours=colours,
                dDLR_bar=dDLR_bar,
                label=None,
            )
            if rectangle:
                # Draw rectangle boundaries
                # Vertical lines (left and right for the rectangle width)
                ax.axvline(x=ra_sn[i] - rect_width / 2, color="red", linestyle="--")
                ax.axvline(x=ra_sn[i] + rect_width / 2, color="red", linestyle="--")

                # Horizontal lines (top and bottom for the rectangle height)
                ax.axhline(y=dec_sn[i] - rect_width / 2, color="blue", linestyle="--")
                ax.axhline(y=dec_sn[i] + rect_width / 2, color="blue", linestyle="--")
            plt.legend()

        if save:
            if len(ra_gal) == 1 and ra_gal.iloc[0] == -999.0:
                print("Not saving plot, no host galaxy found")
            else:
                if out_dir is None:
                    out_dir = "plots/"
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    f"{out_dir}/{id_sn[i]}_{catalogue}_ellipses_{radius_deg*60}arcmin.jpeg"
                )
                print(
                    f"Saving plot to {out_dir}/{id_sn[i]}_{catalogue}_ellipses_{radius_deg*60}arcmin.jpeg"
                )
        else:
            plt.show()
        plt.close()
    return ax


def plot_major_axis(ra_gal, dec_gal, major_axis, position_angle, ax):

    major_x1 = ra_gal - major_axis * np.cos(position_angle)
    major_y1 = dec_gal + major_axis * np.sin(position_angle)
    major_x2 = ra_gal + major_axis * np.cos(position_angle)
    major_y2 = dec_gal - major_axis * np.sin(position_angle)
    ax.plot([major_x1, major_x2], [major_y1, major_y2], "-", color="red", zorder=100)


def plot_host(id_sn, catalogue, radius_deg, filename=None, save=False, out_dir=None):

    gal_df = pd.read_csv(filename)

    for i, sn_name in enumerate(id_sn):
        ra_sn = gal_df.loc[gal_df["sn_name"] == sn_name, "ra_sn"].unique()
        dec_sn = gal_df.loc[gal_df["sn_name"] == sn_name, "dec_sn"].unique()
        if len(ra_sn) == 0:
            print(f"No SN found for {sn_name}")
            continue
        min_RA, max_RA, min_Dec, max_Dec = fnc.get_limits_region(
            ra_sn[0], dec_sn[0], radius_deg
        )
        if catalogue == "GAMA":

            ra_gal = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                "RAcen",
            ]
            dec_gal = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                "Deccen",
            ]

            axrat = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                "axrat",
            ]

            major_axis = (
                gal_df.loc[
                    (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                    "R50",
                ]
                / 3600
            )
            minor_axis = major_axis * axrat

            position_angle = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name), "ang"
            ]
            position_angle_axis = np.radians(position_angle - 90)

        elif catalogue == "lsdr10":
            ra_gal = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name), "ra"
            ]
            dec_gal = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name), "dec"
            ]

            shape_r = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                "shape_r",
            ]
            shape_e1 = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                "shape_e1",
            ]
            shape_e2 = gal_df.loc[
                (gal_df["top_match_ac"] == True) & (gal_df["sn_name"] == sn_name),
                "shape_e2",
            ]
            ellipticity = np.sqrt(shape_e1**2 + shape_e2**2)
            axrat = (1 - ellipticity) / (1 + ellipticity)

            major_axis = shape_r / 3600
            minor_axis = major_axis * axrat
            position_angle = 0.5 * np.arctan2(shape_e2, shape_e1)
            position_angle_axis = position_angle + np.pi / 2

        else:
            print("Catalogue not recognised, please use 'GAMA' or 'lsdr10'")
            sys.exit()
        if ra_gal.empty:
            print(f"No host galaxy found for {sn_name} in the {catalogue} catalogue")
            continue

        else:
            fig, ax = plt.subplots()
            ax.set_title("SN and host galaxy")
            ax.set(xlabel=("RA (deg)"), ylabel=("Dec (deg)"))
            ax.set_aspect("equal")
            ax.set(
                xlabel=("RA (deg)"),
                ylabel=("Dec (deg)"),
                xlim=(max_RA, min_RA),
                ylim=(min_Dec, max_Dec),
            )
            plot_jpeg_cutout(
                sn_name,
                ra_sn[0],
                dec_sn[0],
                radius_deg,
                ax,
            )
            draw_ellipses(
                ra_gal,
                dec_gal,
                ra_sn,
                dec_sn,
                major_axis * 3600,
                minor_axis * 3600,
                position_angle,
                ax,
                catalogue,
                scale_factor=[1, 4],
                rot_type="anticlockwise",
                colours=["red", "blue"],
                label=None,
            )
            # plt.legend()
            plot_major_axis(ra_gal, dec_gal, major_axis, position_angle_axis, ax)

            if save:
                if out_dir is None:
                    out_dir = "plots/hosts"
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{out_dir}/{sn_name}_{catalogue}_host.jpeg")
                print(f"Saving plot to {out_dir}/{sn_name}_{catalogue}_host.jpeg")

        # angle_from_major = fnc.get_sn_angle_from_major(
        #     ra_sn[i], dec_sn[i], ra_gal, dec_gal, position_angle
        # )


def redshift_dif(filename_sn, filename_gal, catalogue):
    sn_df = pd.read_csv(filename_sn)

    if filename_gal.endswith(".fits"):
        gal_df = Table.read(filename_gal).to_pandas()
    else:
        gal_df = pd.read_csv(filename_gal)

    if catalogue == "GAMA":
        z_sn = sn_df["redshift"]
        z_gal = gal_df["Z"]
        host = gal_df["top_match_ac"] == True
        z_gal = z_gal[host]
        id_sn = gal_df["sn_name"][host]
        z_sn = z_sn[sn_df["name"].isin(id_sn)]

    else:
        print(
            "Catalogue not recognised, please use 'GAMA'. If you used 'lsdr10, please note that legacy survey catalogues doesn't provide redshifts"
        )

    fig1, ax1 = plt.subplots()
    ax1.scatter(z_sn, z_gal, color="blue")
    ax1.set_title("Redshift of galaxies vs redshift of SN")
    ax1.set(xlabel=("Redshift of SN"), ylabel=("Redshift of galaxy"))

    dif_z = np.abs(np.array(z_sn) - np.array(z_gal))
    fig2, ax2 = plt.subplots()
    ax2.scatter(
        z_gal, dif_z, color="red"
    )  #! interesting? See if the difference is correlated with the redshift of the galaxy
    ax2.set_title("Redshift difference vs redshift of galaxy")
    ax2.set(xlabel=("Redshift of galaxy"), ylabel=("Redshift difference"))

    mean = np.mean(dif_z)
    median = np.median(dif_z)
    std_dev = np.std(dif_z)
    fig3, ax3 = plt.subplots()
    ind = np.where(np.abs(dif_z) < 1)
    ax3.hist(dif_z, bins=100)
    plt.legend([f"Mean: {mean:9f}\nMedian: {median:9f}\nStd Dev: {std_dev:9f}"])
    ax3.set_title("Histogram of redshift differences")
    ax3.set(xlabel=("Redshift difference"), ylabel=("Counts"))

    plt.show()
