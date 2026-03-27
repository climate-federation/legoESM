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

Thermodynamic constants are imported from the central ``legoesm.constants``
module to guarantee consistency in heating rates and column density
calculations between RRTMGP and the host model.
"""

from legoesm.constants import (
    g as G,
    R_d as R_D,
    R_v as R_V,
    c_pd as CP_D,
    c_vd as CV_D,
    c_pv as CP_V,
)

# Molecular constants specific to RRTMGP spectral calculations.
DRY_AIR_MOL_MASS = 0.0289647  # The molecular mass of dry air (kg/mol).
WATER_MOL_MASS = 0.0180153  # The molecular mass of water (kg/mol).
AVOGADRO = 6.022e23  # Avogadro's number.
