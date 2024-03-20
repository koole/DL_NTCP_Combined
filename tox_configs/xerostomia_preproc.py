# Preprocessing config
preproc_endpoint = "HN35_Xerostomia_M12_class"
preproc_baseline_col = "HN35_Xerostomia_W01_class"
preproc_submodels_features = [
    [
        "Submandibular_meandose",
        "HN35_Xerostomia_W01_little",
        "HN35_Xerostomia_W01_moderate_to_severe",
    ],
    [
        "Parotid_meandose_adj",
        "HN35_Xerostomia_W01_little",
        "HN35_Xerostomia_W01_moderate_to_severe",
    ],
]  # Features of submodels. Should be a list of lists. len(submodels_features) = nr_of_submodels. If None, then
# no fitting of submodels.
preproc_features = [
    "Submandibular_meandose",
    "Parotid_meandose_adj",
    "HN35_Xerostomia_W01_little",
    "HN35_Xerostomia_W01_moderate_to_severe",
]  # Features of final model. Elements in submodels_features should
# be a subset of `features`, i.e. submodels_features can have more distinct features than the final model.
preproc_lr_coefficients = None  # [-2.9032, 0.0193, 0.1054, 0.5234, 1.2763]  # Values starting with coefficient for `intercept`,
# followed by coefficients of `features` (in the same order). If None, then no predefined coefficients will be used.
preproc_ext_features = [
    "HN35_Xerostomia_W01_not_at_all",
    "CT+C_available",
    "CT_Artefact",
    "Photons",
    "Loctum2_v2",
    "Split",
    "Gender",
    "Age",
]