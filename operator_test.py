from deap_gp import cpu_ops
from deap_gp import gpu_ops

import numpy as np

ma = np.random.random((1200,3000))
mb = np.random.random((1200,3000))

print(cpu_ops.protected_div(ma,mb))
