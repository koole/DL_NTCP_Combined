# "xerostomia", "taste", "dysphagia"
toxicity = "xerostomia"
# Whether to perform quick run for checking workability of code or not
perform_test_run = True


from tox_configs.base import *

if toxicity == "xerostomia":
    from tox_configs.xerostomia import *
elif toxicity == "taste":
    from tox_configs.taste import *
elif toxicity == "dysphagia":
    from tox_configs.dysphagia import *

if perform_test_run:
    lr_finder_num_iter = 0
    n_samples = 50
    nr_runs = 1
    max_epochs = 2
    train_frac = 0.33
    val_frac = 0.33
    cv_folds = 4
    batch_size = 2
    num_workers = 0
    pin_memory = False
    plot_interval = 1
    max_nr_images_per_interval = 5