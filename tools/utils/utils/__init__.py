from . import optimize
from .optimize import newton_root_vec, newton_root_scal, newton_optimize_vec, newton_optimize_scal

from . import diff
from .diff import grad, hessian, jacobian, deriv

from . import matrix
from .matrix import seidel, simple_iterative

from . import ode
from .ode import rk4_nsteps, rk4_step, eyler_adaptive

from . import vm_models
