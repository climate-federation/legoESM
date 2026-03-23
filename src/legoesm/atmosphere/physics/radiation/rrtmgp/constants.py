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

Values are aligned with legoesm.constants to ensure consistent heating rates
and column density calculations between RRTMGP and the host model.
"""

# Use the same values as legoesm.constants for consistency in heating rate
# computation (g/c_pd) and column density (dp/g). The original swirl_jatmos
# values were G=9.81, R_D=286.69 which differ by ~0.1% and cause a small
# but unnecessary inconsistency in radiation-dynamics coupling.
G = 9.80616  # Gravitational acceleration [m/s^2].
R_D = 287.05  # The gas constant for dry air [J/kg/K].
R_V = 461.51  # The gas constant for water vapor [J/kg/K].
CP_D = 1004.64  # Constant-pressure heat capacity of dry air [J/kg/K].
CV_D = 717.56  # Constant-volume heat capacity of dry air [J/kg/K].
CP_V = 1846.0  # Constant-pressure specific heat of water vapor [J/kg/K].
DRY_AIR_MOL_MASS = 0.0289647  # The molecular mass of dry air (kg/mol).
WATER_MOL_MASS = 0.0180153  # The molecular mass of water (kg/mol).
AVOGADRO = 6.022e23  # Avogadro's number.
