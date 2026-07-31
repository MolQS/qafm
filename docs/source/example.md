

<a id="force-deconvolution"></a>

## Force Deconvolution

The interaction force $F_{\text{even}}^\circ$ and the interaction potential $U_{\text{even}}^\circ$ can be reconstructed from the measured observable `kts_cap` under the following assumptions:

- Only even forces are present.
- $q_{\text{s}} = 0$.
- Force and potential are zero at the largest sampled value of $z_{\text{p}}$.

Suitable reconstruction algorithms are implemented in `qafm.inversion`. The Sader–Jarvis deconvolution approach is implemented in `Feven_deconv()`, and the matrix method is implemented in `Feven_matrix_deconv()`.

```text
Feven_deconv(zk, ktscap, A0, sgwin, sgdegree, tozero)
Feven_matrix_deconv(zk, ktscap, A0, tozero, spacing_tolerance)
Feven_deconv_iterative()  # not implemented yet
```

The following example in the Python code and the figure below demonstrates the convolution and deconvolution workflow in `qafm`.

```python
import qafm

# create a z-axis
z_axis = np.linspace(-0.01e-9, 30e-9, 5000)

# create sensor parameters
sensor = qafm.oscillator.OscillatorParameters(
    f0=300000.0,  # Hz
    k0=35.0,      # N/m
    Q0=20000,     # dimensionless
    A0=3e-9,      # m
)

# create a combined force model (e.g., vdW + Lennard-Jones)
model_force = qafm.interactions.combined.CombinedForce(
    forces=(
        qafm.interactions.vdw.cone_sphere_force,
        qafm.interactions.chemical.lennard_jones_force,
    ),
    gradients=(
        qafm.interactions.vdw.cone_sphere_forcegradient,
        qafm.interactions.chemical.lennard_jones_forcegradient,
    ),
    zero_for_negative_z=True,
).force(z_axis)

# example for convolution and deconvolution
zk, kts = qafm.numerics.differentiation.numdiff(
    xin=z_axis, yin=model_force, order=1, method="gradient")
    
zc, ktscap = qafm.averaging.wcap(zk, kts, A0=sensor.A0)

df = qafm.fm.conversions.ktscap_to_df_approx(
    ktscap=ktscap,model_par=sensor)

zFeven, Feven = qafm.fm.inversion.df_to_force(
    z=zc, df=df, sensor=sensor, tozero=False)

ktsevencap = qafm.fm.conversions.df_to_ktscap(df=df, model_par=sensor)

zFeven_matrix, Feven_matrix = qafm.fm.inversion.Feven_matrix_deconv(
    zk=zc, ktscap=ktsevencap, A0=sensor.A0)
```

The model tip-sample force from [Interaction Laws](#interaction-laws) is used, which combines a van der Waals and Lennard-Jones interaction. The force is numerically differentiated to obtain the force gradient, which is then cap-averaged to simulate the measured FM-AFM observable. From the resulting frequency shift, the conservative force is reconstructed by deconvolution and compared with the original model force.

|  | |
|---|---|
| ![1](afm-toolbox/example-convolution-model-force.png) | ![2](afm-toolbox/example-convolution-kts.png) |
| ![3](afm-toolbox/example-convolution-df.png) | ![4](afm-toolbox/example-convolution-calculated-force.png) |

*Figure: Example procedure for convolution and deconvolution with a model tip-sample force.*

The algorithms use Savitzky–Golay filters implemented in `savgolfilter.py` to calculate derivatives. For equidistant data along $z$, `scipy.signal.savgol_filter` is used. For non-uniform data along $z$, a custom implementation is used.