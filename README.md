# crossmatches

## To test only with legacy

Step 1: Run python dlr_main.py --testing

## To test GAMA too

Step 1: Download from GAMA the gkvScienceCatv02 (It is to heavy to update it in GitHub)

<https://www.gama-survey.org/dr4/schema/dmu.php?id=1004>

Step 2: Put this catalogue in the data_files/ folder

Step 3: Uncomment the GAMA part in the main function in dlr_main.py

Step 4: Run python dlr_main.py --testing
