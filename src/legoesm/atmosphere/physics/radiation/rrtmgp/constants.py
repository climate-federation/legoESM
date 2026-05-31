# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Commonly used physical constants.

All values are re-exports from the central ``legoesm.constants`` module to
guarantee bit-identical agreement in heating rates and column density
calculations between RRTMGP and the host model.  No values are defined
locally — any drift would silently bias radiative transfer.
"""

from legoesm.constants import (
    g as G,
    R_d as R_D,
    R_v as R_V,
    c_pd as CP_D,
    M_dry as DRY_AIR_MOL_MASS,
    M_h2o as WATER_MOL_MASS,
    N_A as AVOGADRO,
)
# iter-47: dropped ``c_vd as CV_D`` and ``c_pv as CP_V`` re-exports.
# Both were imported here but referenced nowhere in the rrtmgp
# package after the iter-22 swirl_jatmos compute_heating_rate purge.
# RRTMGP only needs CP_D (heating-rate conversion) and the molar
# constants; CV_D / CP_V are for moist-thermo paths in the dycore.
