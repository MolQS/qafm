# qafm - A python package for Quantitative Atomic Force Microscopy

Atomic force microscopy (AFM) probes surfaces and nanoscale interactions by monitoring the motion of a mechanical resonator carrying a sharp tip. In dynamic, non-contact AFM the resonator oscillates close to its resonance frequency while tip-sample forces modify its static deflection, frequency, amplitude, phase, and energy dissipation. Quantitative interpretation of these observables requires a consistent description of the resonator dynamics, the spatial averaging caused by its finite oscillation amplitude, and the reconstruction of the underlying interaction forces.

The **AFM Toolbox** provides the mathematical and numerical tools needed for this analysis. It supports the forward problem-calculating AFM observables from model interactions-as well as the inverse problem of reconstructing conservative forces and interaction potentials from measured frequency-shift data. The theoretical framework follows H. Söngen et al., *Quantitative atomic force microscopy*, J. Phys. Condens. Matter 29, 274001 (2017) ([DOI: 10.1088/1361-648X/aa6f8b](https://doi.org/10.1088/1361-648X/aa6f8b)). Force deconvolution is based on J. E. Sader and S. P. Jarvis, *Accurate formulas for interaction force and energy in frequency modulation force spectroscopy*, Appl. Phys. Lett. 84, 1801 (2004) ([DOI: 10.1063/1.1667267](https://doi.org/10.1063/1.1667267)). Furhtermore, the toolbox implements tip-sample **Interaction Laws** from well established theoretical forces in the submodule `qafm.interactions`.

![Layout of the AFM toolbox](afm-toolbox/package_layout.png)

*Figure: Structure of the AFM Toolbox and its main functional areas.*

## Wiki overview

```{toctree}
:maxdepth: 2
:caption: Contents

installation
interactions-laws
qafm
example
api_reference
```

The documentation is organized into the following chapters:

- [Interaction Laws](#interaction-laws) introduces analytical force and force-gradient models, predefined interaction combinations, and the flexible construction of combined force models.
- [AFM theory](#afm-theory) derives the equations that connect the tip-sample interaction to the motion and experimentally accessible signals of the resonator.
- [Harmonic Oscillator](#harmonic-oscillator) describes the resonator transfer function, amplitude and phase response, resonance frequency, and fitting routines.
- [Coordinate Systems](#coordinate-systems) defines the tip-sample, piezo, and analysis axes required for an unambiguous interpretation of AFM data.
- [Averaging functions](#averaging-functions) introduces the cup and cap averages that account for sampling of the interaction over one oscillation cycle.
- [AFM Observables and Solvers](#afm-observables-and-solvers) relates physical interaction parameters, measured observables, and sensor parameters, and presents the available numerical solvers.
- [Force Deconvolution](#force-deconvolution) explains how conservative forces and interaction potentials are reconstructed from frequency-shift or cap-averaged force-gradient data.
- [Helper Functions](#helper-functions) summarizes frequently used conversions and utilities for AFM calculations.

## Package overview

The package should be used via:

```python
import qafm
```

The `qafm` package is structured as visualized below

```text
qafm/
├── __init__.py
├── averaging.py
├── oscillator.py
├── parameters.py
│
├── config/
│   ├── __init__.py
│   └── defaultParameterSet.py
│
├── fm/
│   ├── __init__.py
│   ├── conversions.py
│   ├── inversion.py
│   └── observables.py 
│
├── interactions/
│   ├── __init__.py
│   ├── chemical.py
│   ├── combined.py
│   ├── custom.py
│   ├── electrostatic.py
│   └── vdw.py
│
└── numerics/
    ├── __init__.py
    ├── differentiation.py
    ├── filters.py
    └── utils.py
```

