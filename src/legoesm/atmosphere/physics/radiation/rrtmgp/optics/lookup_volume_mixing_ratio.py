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

"""A base Dataclass for RRTMGP lookup tables."""

import dataclasses
from typing import TypeAlias

import jax

Array: TypeAlias = jax.Array

@dataclasses.dataclass(frozen=True, kw_only=True)
class LookupVolumeMixingRatio:
  """Lookup table of volume mixing ratio profiles of atmospheric gases."""

  # Volume mixing ratio (vmr) global mean of predominant atmospheric gas
  # species, keyed by chemical formula.
  global_means: dict[str, float]
  # Volume mixing ratio profiles, keyed by chemical formula.
  profiles: dict[str, Array] | None = None
