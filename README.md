# crossmatches

## To test only with legacy

Step 1: Run python dlr_main.py --testing

## To test GAMA too

Step 1: Download from GAMA the gkvScienceCatv02 (It is to heavy to update it in GitHub)

<https://www.gama-survey.org/dr4/schema/dmu.php?id=1004>

Step 2: Put this catalogue in the data_files/ folder

Step 3: Uncomment the GAMA part in the main function in dlr_main.py

Step 4: Run python dlr_main.py --testing

# About the files

## Output file

The final output file will be saved in f"{out_dir}/{catalogue}/possible_hosts/{catalogue}_after_cuts_{outfilename}".

If you decide to save the data per region for each SN, it will be saved in f"{out_dir}/{catalogue}/{reg_arcmin}arcmin_regions/{sn_name}"

## Notes about the output file

- galaxies_in_region: Flag that will be True if the SN has galaxies in the region of the size that you decided or False if there are no galaxies in that region.
- dDLR: Will have the value of the dDLR between the SN and that galaxy. Exceptions: It will be -999 in two different cases:
    1. There are no galaxies in the region. This is ony to keep track of how many SN are outside the footprint, if galaxies_in_region = False, all the galaxy information will be -999, since there are no galaxies.
    2. There are galaxies in the region, but none of them has a dDLR < 4, so there is not a possible match. These are the hostless SN, so again, all the galaxy information will be -999. **It will be useful in this case to keep the dDLR of the closest galaxy even if the dDLR > 4, because in some cases, we lost the host because of this, especially at low redshift. We would need to count how many we lose to think about the dDLR > 4 cut**
- multiple_matches: Flag that will be True if several of those galaxies in the region have a dDLR < 4 with the SN, and False if only one galaxy has dDLR < 4. Exceptions: This will be False also in the case in which there are no galaxies in the region as well as in the case in which the galaxies are no close enough to the SN.
-
