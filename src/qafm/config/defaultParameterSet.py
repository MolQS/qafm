# defaultParameterSet.py

# ---------------------------------------------------------------------------
# Sensor parameters
# ---------------------------------------------------------------------------

f0 = 300000.0          # Hz
k0 = 35.0             # N/m
Q0 = 20000            # dimensionless
A0 = 5e-9             # m


# ---------------------------------------------------------------------------
# van der Waals parameters
# ---------------------------------------------------------------------------

H = 357.619e-21   # J
Theta = 29.7      # degree
R = 15e-9         # m
zoffset = 583.04e-12  # m

# optional F6-type geometry parameters
default_h1 = None         # m
default_L = None          # m


# ---------------------------------------------------------------------------
# Morse parameters
# ---------------------------------------------------------------------------

kappa = 4.25e9      # 1/m
sigma0 = 0.235e-9   # m
ebond = 0.371e-18   # J


# ---------------------------------------------------------------------------
# Lennard-Jones parameters
# ---------------------------------------------------------------------------

sigma_tip: float = 0.3e-9          # m
sigma_sample: float = 0.3e-9       # m
epsilon_tip: float = 1.0e-21       # J
epsilon_sample: float = 1.0e-21    # J
r_offset: float | None = None      # m
r_min: float = 1.0e-12             # m


# ---------------------------------------------------------------------------
# Electrostatic parallel-plate parameters
# ---------------------------------------------------------------------------

tip_area: float = 5e-9 * 5e-9  # m^2
vbias: float = 1.0             # V
eps_r: float = 1.0             # unitless


# ---------------------------------------------------------------------------
# Electrostatic sphere-plane parameters
# ---------------------------------------------------------------------------

radius: float = 5e-9        # m
vbias: float = 1.0          # V
eps_tip: float = 1.0        # unitless
eps_sample: float = 1.0     # unitless
z_min: float = 1e-12        # m
