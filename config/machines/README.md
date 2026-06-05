# Machine profiles

Per-host configuration consumed by the experiment harness
(`scripts/experiment/{lego_detect_machine,init_experiment,fetch_data}.py`). Each
file is one profile under a `machine:` key:

```yaml
machine:
  name: macbook
  hostnames: ["*MacBook*", "*.local"]   # fnmatch patterns vs socket.gethostname()
  scheduler: none                       # none | slurm
  jax_platforms: cpu                    # value for JAX_PLATFORMS
  precision: float64                    # default dynamics precision (overridable with -o)
  venv_activate: "source .venv/bin/activate"
  data_root: ./data                     # where fetch_data.py stages datasets
  slurm: {}                             # partition/nodes/gpus/time/account when scheduler=slurm
```

Resolution: `lego_detect_machine.resolve_machine()` picks the first non-`default`
profile whose `hostnames` patterns match the host, else `default`. `default.yaml`
has empty `hostnames` so it is only ever the fallback (deterministic regardless
of file order).

```bash
python scripts/experiment/lego_detect_machine.py            # -> detected profile name
python scripts/experiment/lego_detect_machine.py --dump     # full resolved profile
python scripts/experiment/lego_detect_machine.py --host levante1   # test a hostname
```

Shipped: `default` (CPU fallback), `macbook` (Apple-Silicon → CPU; Metal backend
broken), `levante-gpu` (DKRZ Levante A100, SLURM). Add a profile by dropping a
new `<name>.yaml` here.
