tests/unit/test_cmor_experiments_restart.py-833-        """Different config gives different hash."""
--
tests/unit/test_cmor_experiments_restart.py-835-        from legoesm.forcing.amip_config import AMIPExperimentConfig
tests/unit/test_cmor_experiments_restart.py-836-
tests/unit/test_cmor_experiments_restart.py-837-        cfg1 = AMIPExperimentConfig(resolution=16)
tests/unit/test_cmor_experiments_restart.py-838-        cfg2 = AMIPExperimentConfig(resolution=48)
tests/unit/test_cmor_experiments_restart.py:839:        self.assertNotEqual(compute_config_hash(cfg1), compute_config_hash(cfg2))
tests/unit/test_cmor_experiments_restart.py-840-
tests/unit/test_cmor_experiments_restart.py-841-
tests/unit/test_cmor_experiments_restart.py-842-class TestSaveLoadRestart(unittest.TestCase):
tests/unit/test_cmor_experiments_restart.py-843-    """Save and load restart with metadata."""

codex
(4) RESOLVED — [config.py:2153](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2153) now matches resolver tolerance at [physics_pipeline.py:3171](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3171).

(2) RESOLVED — [config.py:2980](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2980) stamps v2; [config.py:2926](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2926) drops it before construction, while [config.py:2942](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2942) gates migration and [config.py:2993](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2993) remains marker-independent. No consumer schema rejects the extra key.

VERDICT: SHIP
tokens used
75,215
(4) RESOLVED — [config.py:2153](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2153) now matches resolver tolerance at [physics_pipeline.py:3171](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3171).

(2) RESOLVED — [config.py:2980](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2980) stamps v2; [config.py:2926](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2926) drops it before construction, while [config.py:2942](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2942) gates migration and [config.py:2993](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2993) remains marker-independent. No consumer schema rejects the extra key.

VERDICT: SHIP
