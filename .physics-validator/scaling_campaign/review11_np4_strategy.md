1. **F — static guard on `n_devices == 4` *and* the known edge-leaf shape, barrier only on the tendency tail:** likely preserves the measured **−17.6% np4**; zero intended np2/np8 impact; jit/scan/grad-safe; tightly local to `sharded_dynamics.py`.  
2. **B — `n_devices == 4` static Python guard around the tendency-output barrier:** likely **−17.6% np4**; avoids known np2 +10.7%, but risks other 4-device resolutions; AD-safe; small shared-file change.  
3. **E — document and report upstream:** **0% np4**; no performance/AD regression risk; no code blast radius.  
4. **D — separate jitted edge-array axpy:** uncertain np4 benefit; extra call boundary may hurt other shard counts; generally AD-safe but scan/jit composition risk; moderate local complexity.  
5. **C — per-leaf axpy plus donation hints:** uncertain and donation is awkward for values needed by RK/AD; risk across np2–np8; AD/liveness risk; affects pytree execution behavior.  
6. **A — barrier inside `pytree_axpy`:** could help np4, but risks all integrators and all device counts; AD-safe in principle; largest blast radius.

Recommend **F**: narrowly gate the tendency-tail barrier to the proven np4 shape/signature, keeping the workaround out of generic `pytree_axpy` and avoiding the demonstrated np2 regression.