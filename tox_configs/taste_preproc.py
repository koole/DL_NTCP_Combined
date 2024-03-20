# Preprocessing config
preproc_endpoint = 'Taste_M06'
preproc_baseline_col = 'Taste_W01'
preproc_submodels_features = [
    ['Parotid_meandose', 'OralCavity_Ext_meandose', 'LEEFTIJD'],
]  # Features of submodels. Should be a list of lists. len(submodels_features) = nr_of_submodels. If None, then
# no fitting of submodels.
preproc_features = ['Parotid_meandose', 'OralCavity_Ext_meandose', 'LEEFTIJD']  # Features of final model. Elements in submodels_features should
# be a subset of `features`, i.e. submodels_features can have more distinct features than the final model.
#####
preproc_lr_coefficients = None  # [-2.9032, 0.0193, 0.1054, 0.5234, 1.2763]  # Values starting with coefficient for `intercept`,
# followed by coefficients of `features` (in the same order). If None, then no predefined coefficients will be used.
preproc_ext_features = ['Artefact', 'LEEFTIJD']
