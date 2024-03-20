import os

toxicity = os.getenv("TOX")

if toxicity is None:
    toxicity = "xerostomia"

if toxicity == "xerostomia":
    from tox_configs.xerostomia_preproc import *
elif toxicity == "taste":
    from tox_configs.taste_preproc import *
elif toxicity == "dysphagia":
    from tox_configs.dysphagia_preproc import *