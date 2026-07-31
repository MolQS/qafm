# AFM theory
<!-- <a id="afm-theory"></a> -->

In an AFM experiment, the tip is attached to a mechanical resonator, such as a cantilever, tuning fork, or length-extension sensor. The resonator is modeled as a harmonic oscillator with effective mass $m$, spring constant $k$, and damping constant $\gamma$. Equivalently, it can be characterized by $k$, its eigenfrequency

$$
f_0
=
\frac{1}{2\pi}\sqrt{\frac{k}{m}},
$$

and its quality factor

$$
Q = \frac{\sqrt{km}}{\gamma}.
$$

The implementation of the harmonic oscillator in the AFM toolbox is described in [Harmonic Oscillator](#harmonic-oscillator). Within the harmonic approximation, the resonator deflection and tip-sample distance are

$$
\begin{aligned}
q(t)
&=
q_{\text{s}}
+
A\cos\left(2\pi f_{\text{exc}}t+\varphi\right), \\
z_{\text{ts}}(t)
&=
z_{\text{c}}
+
A\cos\left(2\pi f_{\text{exc}}t+\varphi\right),
\end{aligned}
$$

and the tip velocity is given by

$$
\dot{z}_{\text{ts}}
=
-2\pi f_{\text{exc}} A
\sin\left(2\pi f_{\text{exc}}t + \varphi\right).
$$

Here, $q_{\text{s}}$ is the static deflection, $A$ the oscillation amplitude, $f_{\text{exc}}$ the excitation frequency, and $\varphi$ the phase shift. The center position of the tip oscillation is denoted by $z_{\text{c}}$. A precise description of the tip oscillation in the context of AFM and the tip-sample interaction requires the definition of several $z$ axes, which are described in [Coordinate Systems](#coordinate-systems).

The tip motion is governed by the equation of motion of the driven damped harmonic oscillator:

<a id="oscillator-equation"></a>

$$
m\ddot{q}
=
F_{\text{ts}}
\left(
z_{\text{ts}},
\dot{z}_{\text{ts}}
\right)
+
F_0\cos\left(2\pi f_{\text{exc}}t\right)
-
kq
-
\gamma\dot{q}.
$$

Here, $F_{\text{ts}}$ denotes the force exerted by the sample on the tip. The tip-sample force can generally depend on both the tip-sample distance $z_{\text{ts}}$ and the tip velocity $\dot{z}_{\text{ts}}$, and can be classified as either even or odd with respect to the tip velocity:

$$
F_{\text{ts}}
\left(
z_{\text{ts}},
\dot{z}_{\text{ts}}
\right)
=
F_{\text{even}}
\left(
z_{\text{ts}},
\dot{z}_{\text{ts}}
\right)
+
F_{\text{odd}}
\left(
z_{\text{ts}},
\dot{z}_{\text{ts}}
\right).
$$

Even and odd forces satisfy the following symmetry relations:

$$
\begin{aligned}
F_{\text{even}}
\left(
z_{\text{ts}},
\dot{z}_{\text{ts}}
\right)
&=
F_{\text{even}}
\left(
z_{\text{ts}},
-\dot{z}_{\text{ts}}
\right), \\
F_{\text{odd}}
\left(
z_{\text{ts}},
\dot{z}_{\text{ts}}
\right)
&=
-
F_{\text{odd}}
\left(
z_{\text{ts}},
-\dot{z}_{\text{ts}}
\right).
\end{aligned}
$$

The even and odd force components contribute to different time-averaged quantities: the even component is related to the time-averaged kinetic energy, whereas the odd component determines the time-averaged power transfer. Conservative forces contribute exclusively to the even component, while non-conservative forces may, in general, contribute to both the even and odd components.

Introducing the tip-sample force gradient $k_{\text{ts}}$ and the tip-sample damping coefficient $\gamma_{\text{ts}}$ as

$$
\begin{aligned}
k_{\text{ts}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
&=
\frac{
\partial F_{\text{even}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
}{
\partial z_{\text{ts}}
}, \\
F_{\text{odd}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
&=
-\gamma_{\text{ts}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
\dot{z}_{\text{ts}},
\end{aligned}
$$

the experimentally accessible quantities within the harmonic approximation are weighted averages over the interval $[z_{\text{c}}-A,z_{\text{c}}+A]$, as further shown in [Averaging functions](#averaging-functions). The three AFM equations connect the measured quantities to the weighted cup and cap averages:

$$
\begin{aligned}
\left\langle
F_{\text{even}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
\right\rangle_{\cup}
&=
kq_{\text{s}}, \\
\left\langle
k_{\text{ts}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
\right\rangle_{\cap}
&=
k\left[
1-
\left(
\frac{f_{\text{exc}}}{f_0}
\right)^2
\right]
-
\frac{F_0}{A}\cos\varphi, \\
\left\langle
\gamma_{\text{ts}}
\left(z_{\text{ts}},\dot{z}_{\text{ts}}\right)
\right\rangle_{\cap}
&=
-\frac{k}{2\pi f_0Q}
-
\frac{F_0}{2\pi f_{\text{exc}}A}\sin\varphi.
\end{aligned}
$$

Quantitative AFM distinguishes between weighted averages of physical parameters, experimental observables, and sensor parameters. The conversion between the measured observables and the weighted averages of the physical parameters is described in [AFM Observables and Solvers](#afm-observables-and-solvers). The three AFM equations can be solved for the weighted averages of the physical parameters using different solvers, as described in that section. The interaction force $F_{\text{even}}$ and the interaction potential $U_{\text{even}}$ can be reconstructed from the measured observable $\langle k_{\text{ts}} \rangle_{\cap}$; suitable reconstruction algorithms are described in [Force Deconvolution](#force-deconvolution).

<a id="harmonic-oscillator"></a>

## Harmonic Oscillator

Dynamic atomic force microscopy is based on an externally driven, damped harmonic oscillator. The mechanical sensor such as a cantilever or tuning fork is characterized by its eigenfrequency, stiffness, and quality factor. Its response to a harmonic excitation determines the measured oscillation amplitude and phase.

The corresponding functions and data classes are implemented in `qafm.oscillator`.

The immutable `OscillatorParameters` data class stores the sensor parameters used throughout the module:

| Attribute | Physical quantity | Default value |
|---|---|---:|
| `f0` | Eigenfrequency | 277203 Hz |
| `k0` | Sensor stiffness | 18.58 N/m |
| `Q0` | Quality factor | 20000 |
| `A0` | Oscillation amplitude | 1 nm |

Within the harmonic approximation, the sensor deflection is modeled as a cosine oscillation around the static deflection $q_{{s}}$. The function

- `qafm.oscillator.q_t(t, qs, A, fexc, phi)`

evaluates this expression for one or more time values `t` and returns the corresponding sensor deflection as a NumPy array. The transfer function $G_{\text{ho}}$ determines the amplitude and phase response to a given harmonic excitation signal. It can be evaluated using the following functions:


- `qafm.oscillator.Gho(fexc, model_par=None)` returns the complex transfer function.
- `qafm.oscillator.Gho_A(fexc, model_par=None)` returns its magnitude, equivalent to `abs(Gho(...))`.
- `qafm.oscillator.Gho_phi(fexc, model_par=None)` returns its phase in radians.

For the transfer-function functions, `model_par=None` selects these default values. A compatible parameter container can also be supplied and is resolved internally by `qafm.parameters.resolve_params`. The amplitude resonance of the damped oscillator lies slightly below the eigenfrequency and is returned by

- `qafm.oscillator.f_resonance(model_par=None)`

A quick example of the usage of the harmonic oscillator is shown here:

```python
import numpy as np

from qafm.oscillator import (
    Gho,
    Gho_A,
    Gho_phi,
    OscillatorParameters,
    f_resonance,
)

sensor = OscillatorParameters()
frequencies = np.linspace(
    sensor.f0 - 50.0,
    sensor.f0 + 50.0,
    2_001,
)

transfer = Gho(frequencies, sensor)
amplitude = Gho_A(frequencies, sensor)
phase = Gho_phi(frequencies, sensor)
resonance_frequency = f_resonance(sensor)
```

A fit of the amplitude of $G_{ho}$ to measured amplitude-response data for determining the oscillation parameters can be performed using the function

- `qafm.oscillator.Gho_fit(f, Af, verbose=True)`

The input arrays contain the excitation frequencies `f` and the corresponding positive amplitudes `Af`. Note that the returned parameter `k0` is not necessarily the spring constant unless the system has been properly calibrated.

The function obtains initial estimates from the measured resonance curve:

- `f0` is initialized from the frequency at the maximum amplitude.
- `Q0` is estimated from the full width at half maximum (FWHM). If the width cannot be determined robustly, the fallback value is {math}`Q_0=10^4`.
- `k0` is estimated from the peak magnitude using the near-resonance relation {math}`\max|G_{\mathrm{ho}}|\approx Q_0/k_0`.

The nonlinear least-squares fit is performed with `scipy.optimize.curve_fit`. The inputs must be one-dimensional arrays of equal shape, contain at least five points, and all values of `Af` must be positive.

```python
from qafm.oscillator import Gho_fit

fit = Gho_fit(
    f=frequencies,
    Af=amplitude,
    verbose=False,
)
```

The returned `FitResult` contains:

| Attribute | Description |
|---|---|
| `f0` | Fitted eigenfrequency |
| `k0` | Fitted stiffness or amplitude-scale parameter |
| `Q0` | Fitted quality factor |
| `covariance` | 3x3 covariance matrix returned by `curve_fit` |
| `standard_deviation` | Standard deviations calculated from the diagonal of the covariance matrix |
| `parameters` | Convenience property returning the fitted values as `OscillatorParameters` |

The fit determines only `f0`, `k0`, and `Q0`. Consequently, `FitResult.parameters` uses the default value for `A0`. Moreover, the fitted `k0` represents the physical spring constant only when the measured response has been calibrated with the correct amplitude and excitation-force scale.

| amplitude response | phase response |
|---|---|
| ![Amplitude of the harmonic oscillator transfer function and fitted response](afm-toolbox/example-harmonic-oscillator-transfer-function.png) | ![Phase of the harmonic oscillator transfer function and fitted response](afm-toolbox/example-harmonic-oscillator-transfer-function-phi.png) |

*Figure: Amplitude and phase response of the harmonic oscillator.*

<a id="coordinate-systems"></a>

## Coordinate Systems

A precise description of AFM signals requires the definition of several axes:

| Symbol | Definition | Description |
|---|---|---|
| $z_{\text{ts}}$ | Tip-sample distance | Distance between tip and sample, referenced to the sample surface. |
| $z_{\text{p}}$ | Piezo position | Position of the piezo. Experimental data are acquired with respect to this axis. |
| $z_{\text{tip}}$ | Tip position used for data analysis | Data analysis axis that has an unknown offset $\delta z_0$ with respect to the tip-sample axis. |

*Table: Coordinate systems used in AFM signal analysis.*

An illustration of the different $z$ axes and coordinates is shown in the following figure:

![Coordinate systems used for AFM signal analysis](afm-toolbox/axis-system.png)

*Figure: AFM coordinate systems. Source: P. Rahe et al., Beil. J. Nanotech. 13, 610 (2022), [DOI: 10.3762/bjnano.13.53](https://doi.org/10.3762/bjnano.13.53).*

<a id="averaging-functions"></a>

## Averaging functions

Within the harmonic approximation, the tip samples the force field along one oscillation path. Therefore, the experimentally accessible quantities are weighted averages over the interval $[z_{\text{c}}-A,z_{\text{c}}+A]$.

The cup average is defined as

$$
\left\langle f^\circ \right\rangle_{\cup}(z_{\text{c}})
=
\int_{-A}^{+A}
f^\circ(z_{\text{c}}+z)
w_{\cup}(z)
\,\mathrm{d}z,
$$

with

$$
w_{\cup}(z)
=
\frac{1}{\pi\sqrt{A^2-z^2}}.
$$

The cap average is defined as

$$
\left\langle f^\circ \right\rangle_{\cap}(z_{\text{c}})
=
\int_{-A}^{+A}
f^\circ(z_{\text{c}}+z)
w_{\cap}(z)
\,\mathrm{d}z,
$$

with

$$
w_{\cap}(z)
=
\frac{2}{\pi A^2}
\sqrt{A^2-z^2}.
$$

<table>
<tr>
<td><img src="afm-toolbox/example-cup-average-function.png" alt="Cup averaging function for AFM data" /></td>
<td><img src="afm-toolbox/example-cap-average-function.png" alt="Cap averaging function for AFM data" /></td>
</tr>
</table>

*Figure: Averaging functions for AFM data.*

<a id="afm-observables-and-solvers"></a>

## AFM Observables and Solvers

**Quantitative AFM** distinguishes between weighted averages of physical parameters, experimental observables, and sensor parameters:

| Weighted averages of physical parameters | Experimental observables | Sensor parameters |
|---|---|---|
| $\langle F_{\text{even}} \rangle$ | $q_{\text{s}}$ | $k_0$ |
| $\langle F_{\text{odd}} \rangle$ | $A$ | $Q_0$ |
|  | $f_{\text{exc}}$ | $f_0$ |
|  | $F_0$ |  |
|  | $\varphi$ |  |

*Table: Main quantities used in quantitative AFM.*

Dynamic AFM in the harmonic approximation can determine three quantities of the tip-sample interaction:

- $F_{\text{even}}^\circ$
- $k_{\text{even}}^\circ$
- $\gamma_{\text{ts}}^\circ$

The key measurement observables are the cup- and cap-averaged quantities:

- $\left\langle F_{\text{even}}^\circ \right\rangle_{\cup}$
- $\left\langle k_{\text{even}}^\circ \right\rangle_{\cap}$
- $\left\langle \gamma_{\text{ts}}^\circ \right\rangle_{\cap}$

For efficiency and flexibility, several versions of the averaging functions are implemented. All versions are accessible via the central functions:

- `q_afm.wcup(z_axis, feq, A0)`
- `q_afm.wcap(z_axis, feq, A0)`

Depending on the properties of `z_axis` and the type of `feq`, different implementations are called internally:

- `wcup_callable(z_axis, feq, A0)` and `wcap_callable(z_axis, feq, A0)` are used if `feq` is a callable object, for example a function. Integration is performed using `scipy.integrate.quad`.
- `wcup_equiz(z_axis, feq, A0)` and `wcap_equiz(z_axis, feq, A0)` are used if `feq` is a NumPy array and `z_axis` is equidistant.
- `wcup_nonequiz(z_axis, feq, A0)` and `wcap_nonequiz(z_axis, feq, A0)` are used if `feq` is a NumPy array and `z_axis` is not equidistant. This implementation is slower than the equidistant variant.

The averaging functions are tested using the `ForceModels.Fts_Morse_vdWF3` and `ForceModels.kts_Morse_vdWF3` equations. The results can be plotted by calling `q_afm.example_plots_wcupwcap()`.

Different **solvers** are implemented to calculate the measurement observables from the force interactions for different modes and levels of approximation:

- `fmmode_smallA(z_p, ktsfunc, model_par)`: small-amplitude approximation. This solver does not perform convolution and assumes $q_{\text{s}} = 0$. It uses $\Delta f \approx -\frac{f_0}{2k_0}k_{\text{ts}}(z)$.
- `fmmode_approx(z_p, ktsfunc, model_par)`: assumes $q_{\text{s}} = 0$ and performs a convolution to calculate $\langle k_{\text{ts}} \rangle_{\cap}$.
- `fmmode_itersolve(z_p, Fts, kts, gammats, model_par, maxIter)`: iterative solution within the harmonic approximation, including the static deflection.
- `fmmode_fsolve(z_p, Fts, kts, gammats, model_par, debug)`: uses `scipy.optimize.fsolve` to solve the AFM equations.