# Installation

## Install from GitHub

The package can be installed directly via pip

```bash
python -m pip install qafm
```

or from Github:

```bash
python -m pip install git+https://github.com/MolQS/qafm.git
```

Check the installation with:

```bash
python -c "import qafm; print(qafm.__file__)"
```

## Test the installation

Create a small Python file, for example `test_qafm.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
import qafm

z = np.linspace(0.01e-9, 10e-9, 10000)
vdw_force = qafm.interactions.vdw.cone_sphere_force(z)

plt.figure(figsize=(10, 6))
plt.plot(z * 1e9, vdw_force * 1e12, label="vdW Force")
plt.xlabel("z (nm)")
plt.ylabel("Force (pN)")
plt.legend()
plt.show()
```

## Update an existing installation

To update `qafm` use:

```bash
python -m pip install --upgrade qafm
```

