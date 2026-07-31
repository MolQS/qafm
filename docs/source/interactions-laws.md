

<a id="interaction-laws"></a>

## Interaction Laws

Representative force and force-gradient models are organized by their physical origin. In the legacy implementation, they are defined in `ForceModels.py`; the current Python API exposes them through `qafm.interactions`.

The force laws and their gradients are defined analytically. Legacy functions expect the $z_{\text{ts}}$ axis (see [Coordinate Systems](#coordinate-systems)), the optional velocity $\dot{z}_{\text{ts}}$, and a `model_par` dictionary containing model and AFM sensor parameters. Default parameters for this interface are available through `ForceModels.DefaultParameters()`. The current API instead uses dedicated parameter classes, as demonstrated below.

The examples show how to calculate individual tip-sample force contributions, use predefined custom force combinations, and construct flexible combined force models from callable force and gradient functions. All models are evaluated on a tip-sample distance axis `z_ts`.

### Individual Forces

The basic force and force gradient models can be evaluated separately. The following examples include chemical interactions, electrostatic forces, and several van der Waals geometries. 

```python
import numpy as np
import qafm

z_ts = np.linspace(0.01e-9, 10e-9, 10_000)
```

```python
# chemical forces
morse_par = qafm.interactions.chemical.MorseParameters(
    kappa = 2.5*(1/1e-9), sigma0 = 0.85e-9, ebond = 4.6e-21)
morse_force = qafm.interactions.chemical.morse_force(
    z_axis=z_ts, model_par=morse_par)

lj_params = qafm.interactions.chemical.LennardJonesParameters(
    sigma_tip = 0.9e-9, sigma_sample = 0.9e-9,
    epsilon_tip = 4.6e-21, epsilon_sample = 4.6e-21)
lj_force = qafm.interactions.chemical.lennard_jones_force(
    z_axis=z_ts, model_par=lj_params)
```

```python
# electrostatic forces
elstatic_params = qafm.ParameterSet([
    qafm.interactions.electrostatic.ElectrostaticParameters(
        tip_area=5e-9 * 5e-9,           # m^2
        vbias=1.0,                      # V
        eps_r=1.0e-9                    # unitless
    ),
])
elstatic_parallel_plates = qafm.interactions.electrostatic.parallel_plates_force(
    z_axis=z_ts, model_par=elstatic_params)
elstatic_sphere_plane = qafm.interactions.electrostatic.sphere_plane_force(
    z_axis=z_ts)
```

```python
# van der Waals forces
params = qafm.interactions.vdw.VdwParameters(
    H=357.619e-21,
    Theta=29.7,       # degree
    R=5e-9,
    zoffset=583.04e-12,
)
forces = {    
    "cone": qafm.interactions.vdw.cone_force(z_ts, params),
    "cone sphere": qafm.interactions.vdw.cone_sphere_force(z_ts, params),
    "cone integrated": qafm.interactions.vdw.cone_integrated_force(z_ts, params),
    "truncated cone": qafm.interactions.vdw.truncated_cone_force(z_ts, params),
    "spherical cap cone": \
        qafm.interactions.vdw.spherical_cap_cone_force(z_ts, params),
    "spherical cap cone geom.": \
        qafm.interactions.vdw.spherical_cap_cone_geometric_force(z_ts, params),
}
```

<table>
<tr>
<td><img src="afm-toolbox/example-force-chemical.png" alt="Example interaction force: chemical interaction" /></td>
<td><img src="afm-toolbox/example-force-vdW.png" alt="Example interaction force: van der Waals interaction" /></td>
</tr>
</table>

*Figure: Examples of individual chemical and van der Waals interaction forces.*

| Chemical interaction | van der Waals interaction |
|---|---|
| ![Chemical interaction](afm-toolbox/example-force-chemical.png) | ![van der Waals interaction](afm-toolbox/example-force-vdW.png) |

### Custom Forces

Predefined custom force models combine multiple physical interactions in one model function. The following examples combine a Morse-type chemical interaction with van der Waals contributions for different tip geometries and calculate both the resulting force and its gradient.

```python
from qafm.interactions.custom import *

c1_force = morse_vdw_cone_force(
    z_axis=z_ts,
    morse_par=qafm.interactions.chemical.MorseParameters(),
    vdw_par=qafm.interactions.vdw.VdwParameters(),
)
c1_gradient = morse_vdw_cone_forcegradient(
    z_axis=z_ts,
    morse_par=qafm.interactions.chemical.MorseParameters(),
    vdw_par=qafm.interactions.vdw.VdwParameters(),
)
c2_force = morse_vdw_cone_sphere_force(
    z_axis=z_ts,
    morse_par=qafm.interactions.chemical.MorseParameters(),
    vdw_par=qafm.interactions.vdw.VdwParameters(),
)
c2_gradient = morse_vdw_cone_sphere_forcegradient(
    z_axis=z_ts,
    morse_par=qafm.interactions.chemical.MorseParameters(),
    vdw_par=qafm.interactions.vdw.VdwParameters(),
)
```

<table>
<tr>
<td><img src="afm-toolbox/example-force-custom-1.png" alt="Example interaction force: custom model 1" /></td>
<td><img src="afm-toolbox/example-force-custom-2.png" alt="Example interaction force: custom model 2" /></td>
</tr>
</table>

*Figure: Predefined custom force models combining chemical and van der Waals interactions.*

### Combined Forces

The `CombinedForce` interface provides a flexible alternative to predefined custom models. Individual force and gradient functions are supplied as callables and summed internally. Model-specific parameters can be attached to each function using `functools.partial`.

```python
# parameters for the combined forces
vdw_par = qafm.interactions.vdw.VdwParameters(R=20e-9)
morse_par = qafm.interactions.chemical.MorseParameters()

# single forces
vdw_force = qafm.interactions.vdw.cone_sphere_force(z_ts, vdw_par)
vdw_force_2 = qafm.interactions.vdw.cone_sphere_force(
    z_ts, model_par=vdw_par)

# combined force model with custom parameters
combined_model = qafm.interactions.combined.CombinedForce(
    forces=(
        partial(qafm.interactions.vdw.cone_sphere_force,
                model_par=vdw_par),
        partial(qafm.interactions.chemical.morse_force,
                model_par=morse_par),
    ),
    gradients=(
        partial(qafm.interactions.vdw.cone_sphere_forcegradient,
                model_par=vdw_par),
        partial(qafm.interactions.chemical.morse_forcegradient,
                model_par=morse_par),
    ),
    zero_for_negative_z=True,
)

combined_force = combined_model.force(z_ts)
combined_gradient = combined_model.gradient(z_ts)
```
