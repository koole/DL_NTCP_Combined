"""
This script:
    - Load overview_rtdose.csv and check which patients have invalid dose values
    - Load segmentation_fraction_outside_ct.csv and check which patients have too large segmentation fraction outside CT
    - Load endpoints.csv and check which patients have no endpoint (i.e. for which df_endpoints[endpoint].isna()).

We do not check the UID match between CT and RTSTRUCT, because those (should) have DLC segmentation
"""
import os
import time
import numpy as np
import pandas as pd
import data_preproc_config as cfg
from data_preproc_functions import Logger


def main():
    """
    Tip: disable the usage of cfg.test_patients_list in the following files to make sure that exclude_patients.csv
        will be created for all patients:
        - data_folder_types.py (function: main())
        - data_collection.py (function: main_match_ct_rtdose_rtstruct())
        - check_data_preproc_ct_rtdose.py (function: main())
        - check_data_preproc.py (function: main_segmentation_outside_ct())

    Search/Add `# TODO: temporary: consider all patients, for exclude_patients.csv` in those files above.

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    logger = Logger(output_filename=None)
    rtdose_lower_limit = cfg.rtdose_lower_limit
    rtdose_upper_limit = cfg.rtdose_upper_limit
    max_outside_fraction = cfg.max_outside_fraction
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py (data_folder_types.py)
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            # df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
            #                               for x in df_data_folder_types.index.values]

        # Check which patients have no match between CT UID and RTDOSE UID (data_collection.py)
        # Note: We do not consider patients without matching UID CT and RTSTRUCT, because those (should) have DLC
        # segmentation
        df_invalid_uid = pd.read_csv(os.path.join(save_root_dir, cfg.filename_invalid_uid_csv), sep=';', index_col=0)
        if len(df_invalid_uid) > 0:
            df_invalid_uid = df_invalid_uid[df_invalid_uid['No Frame of Reference UID match CT RTDOSE']][
                'No Frame of Reference UID match CT RTDOSE']

        # Check which patients have invalid dose values (check_data_preproc_ct_rtdose.py)
        rtdose = pd.read_csv(os.path.join(save_root_dir, cfg.filename_overview_rtdose_csv), sep=';', index_col=0)
        # Check which patients have too large segmentation fraction outside CT (check_data_preproc.py)
        df_fraction_outside = pd.read_csv(os.path.join(save_root_dir, cfg.filename_segmentation_fraction_outside_ct_csv),
                                          sep=';', index_col=0, dtype={0:object})

        logger.my_print('RTDOSE in [{}, {}] allowed'.format(rtdose_lower_limit, rtdose_upper_limit))
        logger.my_print('Maximum segmentation fraction outside CT allowed: {}'.format(max_outside_fraction))

        # List of patients
        patients_list = next(os.walk(cfg.save_dir))[1]
        # df_endpoints = pd.read_csv(os.path.join(save_root_dir, cfg.filename_endpoints_csv), sep=';')
        # patients_list = df_endpoints[cfg.patient_id_col]

        df = pd.DataFrame(columns=['Patient_id'])
        for patient_id in patients_list:
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            """
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''
            """

            invalid_dict_i = dict()

            # Too low or too large max RTDOSE value
            try:
                rtdose_i = rtdose.loc[[int(patient_id)]]     # DANIEL: made patient_id an int(), so that it actually indexes the dataframe properly (dataframe index isn't a string padded with zeros)s
                if not (rtdose_lower_limit < rtdose_i['max'].values < rtdose_upper_limit):
                    invalid_dict_i['Patient_id'] = patient_id
                    invalid_dict_i['Max RTDOSE'] = rtdose_i['max'].values  # DANIEL: added ".values"
            except:
                pass

            # Segmentation fraction outside CT
            fraction_outside_i = df_fraction_outside.loc[[int(patient_id)]]   # DANIEL: made patient_id an int(), so that it actually indexes the dataframe properly (dataframe index isn't a string padded with zeros)
            for structure in cfg.parotis_structures + cfg.submand_structures:
                if fraction_outside_i[structure].values > max_outside_fraction:
                    invalid_dict_i['Patient_id'] = patient_id
                    invalid_dict_i['Fraction {} outside CT'.format(structure)] = fraction_outside_i[structure].values   # DANIEL: added ".values" 

            # Add patients to be excluded and the reason
            if len(invalid_dict_i) > 0:
                df_i = pd.DataFrame(invalid_dict_i, index=[patient_id])
                df = pd.concat([df, df_i], axis=0)

        # Set column = 'patient_id' as index
        df = df.set_index(df.iloc[:, 0].values)
        del df['Patient_id']

        # Add patients with no match between CT and RTDOSE UID
        if len(df_invalid_uid) > 0:
            df = pd.concat([df, df_invalid_uid], axis=1)

        # Sort columns based on their names, and save df to csv file
        logger.my_print('Number of patients to be excluded: {}'.format(len(df)))
        df = df.sort_index(axis=1)
        df.to_csv(os.path.join(save_root_dir, cfg.filename_exclude_patients_csv), sep=';')
    print("Hello")
    find_all_patients_with_missing_structures()

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def find_all_patients_with_missing_structures():
    """
    Reads the cfg.filename_overview_structures_count_csv file, which contains the pixel counts for each patient's 16 structures
    Saves a new file that just shows the patients that are missing (at least 1) structure, and which structure(s) are missing
    """
    path_to_structure_count_csv = os.path.join(cfg.save_root_dir, cfg.filename_overview_structures_count_csv)
    df = pd.read_csv(path_to_structure_count_csv, delimiter=';', index_col=0)
    # Set non-zero values to None
    df[df != 0] = None

    # Remove rows where all values are None (i.e. keep only patients that have a struct=0)
    df = df.dropna(axis=0, how='all')
    df[df == 0] = 1

    # save the csv
    save_path = os.path.join(cfg.save_root_dir, cfg.filename_patients_incorrect_structure_count_csv)
    df.to_csv(save_path, sep=';')



if __name__ == '__main__':
    main()
    print("hi")
    


