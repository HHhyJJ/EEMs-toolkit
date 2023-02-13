# -*- coding = utf-8 -*-
# @TIME : 2023/02/07 18:25
# @File : const.py
# @Software : PyCharm

# Outlier Test
# ot = {
#     'starts': 2,
#     'tol': 1e-2,
#     'n_iter_max': 1000,
#     'init': 'svd'
# }
# Modeling
md = {
    'starts': 20,
    'tol': 1e-6,
    'n_iter_max': 2500,
    'init': 'random'
}

# Modeling and Split Half Analysis
# The number of times that the model is randomly initialized.
starts = 20

# Convergence criterion
tol = 1e-6

# Maximum number of iterations
max_iter = 2500

# If you want the same splits, change it to a fixed value.
random_state = None
init = 'random'
