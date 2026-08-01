"""Run a legacy #1226 probe under an fp64 precision policy (skill Rule 1c).

The band probes predate the policy fix and build their grids at the DEFAULT
fp32 control dtype, so their recorded residuals may be dtype rather than
physics. This wrapper changes ONE variable: the precision policy.
"""
import sys, os, runpy
from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
if os.environ.get("FP64", "1") == "1":
    set_policy(PrecisionPolicy.fp64())
print(f"[run_fp64] control dtype = {get_policy().control}", flush=True)
sys.argv = sys.argv[1:]
# probes import sibling helpers from their own directory
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
