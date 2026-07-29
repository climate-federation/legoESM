packages/land/legoesm/land/restart.py:160:    data = np.load(str(path), allow_pickle=False)
scripts/validate/inspect_checkpoint_realism.py:146:    z = np.load(ckpt, allow_pickle=True)
scripts/validate/fv3_native/gen_dsw1_duo_oracle.py:60:    z = np.load(IN_NPZ, allow_pickle=True)
scripts/validate/fv3_native/w2_duo_oracle_gate.py:71:    z = np.load(npz_path, allow_pickle=True)
packages/coupler/legoesm/driver/model_driver.py:701:                # would pickle a 0-d object array into the npz and crash the
packages/coupler/legoesm/driver/model_driver.py:4308:                        # (allow_pickle=False).  Skip by NAME so BOTH None and
packages/coupler/legoesm/driver/model_driver.py:4383:                    # is an unloadable object array under allow_pickle=False).
packages/coupler/legoesm/driver/model_driver.py:5674:        # THREE batched buffer allreduces instead of eight scalar pickle
packages/coupler/legoesm/driver/distributed_checkpoint.py:216:    data = np.load(str(rank_path), allow_pickle=True)
packages/coupler/legoesm/driver/diagnostics.py:1797:        with np.load(str(path), allow_pickle=False) as npz:
packages/ml/legoesm/ml/s2s/neuralgcm_slab/neuralgcm_backend.py:12:import pickle
packages/ml/legoesm/ml/s2s/neuralgcm_slab/neuralgcm_backend.py:65:def _load_pickle(checkpoint: str) -> Any:
packages/ml/legoesm/ml/s2s/neuralgcm_slab/neuralgcm_backend.py:76:                return pickle.load(handle)
packages/ml/legoesm/ml/s2s/neuralgcm_slab/neuralgcm_backend.py:79:            return pickle.load(handle)
packages/ml/legoesm/ml/s2s/neuralgcm_slab/neuralgcm_backend.py:81:        return pickle.load(handle)
packages/ml/legoesm/ml/s2s/neuralgcm_slab/neuralgcm_backend.py:155:        checkpoint = _load_pickle(self.config.checkpoint)
packages/coupler/legoesm/driver/config.py:1042:    # (morrison_phase_aware_sat_adj / cloud_rh_ice_crit / cloud_rh_ice_sat
packages/ml/legoesm/tuning.py:808:    # (cloud_rh_ice_crit / cloud_rh_ice_sat catalog entries DELETED
tests/unit/test_cmor_experiments_restart.py:971:        # (cloud_rh_ice_crit/sat) advertised a CloudConfig RH_i cirrus ramp

codex
VERDICT: SHIP
tokens used
105,995
VERDICT: SHIP
