# Halo Geometry Mismatch (C36)

Distance between target ghost-cell centers and source neighbor boundary-cell centers used by current `pad_halo` copy mapping.

| Face | Edge | Neighbor | Neighbor edge | reversed | mean mismatch (deg) | max mismatch (deg) |
|---:|---|---:|---|:---:|---:|---:|
| 0 | WEST | 3 | EAST | False | 0.648 | 1.161 |
| 0 | EAST | 1 | WEST | False | 0.648 | 1.161 |
| 0 | SOUTH | 5 | NORTH | False | 0.648 | 1.161 |
| 0 | NORTH | 4 | SOUTH | False | 0.648 | 1.161 |
| 1 | WEST | 0 | EAST | False | 0.648 | 1.161 |
| 1 | EAST | 2 | WEST | False | 0.648 | 1.161 |
| 1 | SOUTH | 5 | EAST | True | 0.648 | 1.161 |
| 1 | NORTH | 4 | EAST | False | 0.648 | 1.161 |
| 2 | WEST | 1 | EAST | False | 0.648 | 1.161 |
| 2 | EAST | 3 | WEST | False | 0.648 | 1.161 |
| 2 | SOUTH | 5 | SOUTH | True | 0.648 | 1.161 |
| 2 | NORTH | 4 | NORTH | True | 0.648 | 1.161 |
| 3 | WEST | 2 | EAST | False | 0.648 | 1.161 |
| 3 | EAST | 0 | WEST | False | 0.648 | 1.161 |
| 3 | SOUTH | 5 | WEST | False | 0.648 | 1.161 |
| 3 | NORTH | 4 | WEST | True | 0.648 | 1.161 |
| 4 | WEST | 3 | NORTH | True | 0.648 | 1.161 |
| 4 | EAST | 1 | NORTH | False | 0.648 | 1.161 |
| 4 | SOUTH | 0 | NORTH | False | 0.648 | 1.161 |
| 4 | NORTH | 2 | NORTH | True | 0.648 | 1.161 |
| 5 | WEST | 3 | SOUTH | False | 0.648 | 1.161 |
| 5 | EAST | 1 | SOUTH | True | 0.648 | 1.161 |
| 5 | SOUTH | 2 | SOUTH | True | 0.648 | 1.161 |
| 5 | NORTH | 0 | SOUTH | False | 0.648 | 1.161 |

- Global mean mismatch: **0.648 deg**
- Global max mismatch: **1.161 deg**
