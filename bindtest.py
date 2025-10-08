# %%
import uaibot as ub
import numpy as np
import sys
from adaptive_cpp import AdaptiveController, ControlLoop
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
from uaibot.utils import Utils
from scipy.linalg import block_diag
from vfutils import vector_field_plot

# Sytem
rho = 8050.0  # Density of steel in kg/m^3
r = 15e-2  # Radius of the cylinder in meters
h = 10e-2  # Height of the cylinder in meters
r = 0.25
h = 1.0
m = rho * np.pi * r**2 * h  # Mass of the cylinder in Kg
print(m)


rng = np.random.default_rng(42)
r_p = np.array([0.0, 0.0, h / 2.0])  # Measurement point
N = 6  # Number of agents
# Initial positions of the agents
r_i = np.array([[0, 0, h / 2], [0, 0, -h / 2], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]])
# Radially distributed around the object
angle_dist = np.linspace(0, 2 * np.pi, N, endpoint=False)
# r_i = np.array([[r * np.cos(angle), r * np.sin(angle), h / 2.0] for angle in angle_dist])
print(f"Agents distributed as: {r_i}")

# Inertia tensor
I_cm = (1.0 / 12.0) * np.eye(3)
I_cm[0, 0] *= m * (3 * r**2 + h**2)
I_cm[1, 1] *= m * (3 * r**2 + h**2)
I_cm[2, 2] *= 6 * m * r**2
I_p = np.array(I_cm - m * Utils.S(r_p) @ Utils.S(r_p))

mean_a, std_a = 0.0, 1.0
o_hat = [rng.normal(mean_a, std_a, (10, )) for _ in range(N)]
o_true = np.array(
    [
        m,
        0,
        0,
        0,  # m * r_p
        I_p[0, 0],
        I_p[0, 1],
        I_p[0, 2],
        I_p[1, 1],
        I_p[1, 2],
        I_p[2, 2],
    ]
)
o_true[1:4] = m * r_p
o_i = o_true / N

mean_r, std_r = mean_a, 2 * std_a
r_hat = [rng.normal(mean_r, std_r, (3, )) for _ in range(N)]

k1= 35e-1
# k1 = 35e-3
K_adap = k1 * block_diag(20e3 / N * np.eye(3), 25e3 / N * np.eye(3))


abs_o_i = np.abs(o_i) + 1e-2 * np.ones(o_i.shape)
kgo, kgr = 3e1, 3e3
factor_ = 1.0
Gamma_o_inv = factor_ * kgo * np.linalg.inv(np.diag(abs_o_i.flatten()))  # 3e1
Gamma_r_inv = factor_ * kgr * np.eye(3)  # 3e3

print(type(Gamma_o_inv), Gamma_o_inv.shape)
print(type(Gamma_r_inv), Gamma_r_inv.shape)
print(type(K_adap), K_adap.shape)
print(type(o_i), o_i.shape)
print(type(r_i), len(r_i), len(r_i[0]))
print(type(I_p), I_p.shape)
print(type(m), type(r_hat))
print(type(r_hat), len(r_hat), r_hat[0].shape)
adaptiveSys = AdaptiveController(
    Gamma_o_inv,
    Gamma_r_inv,
    K_adap,
    o_i,
    r_i,
    I_p,
    m,
    N,
    r_p,
)

# %%
# Kinematic Controller
def hd(s, r=1, b=1, d=0.2):
    """Curve parametrization used in paper. This is based on the hyperbolic
    paraboloid.

    Parameters
    ----------
    s : float
        Parameter of the curve. It must be in the interval [0, 1].
    r : float, optional
        Radius of the curve in XY plane. The default is 1.
    b : float, optional
        Height of the curve. The default is 1.
    d : float, optional
        Curvature of the curve. The default is 0.2.

    Returns
    -------
    hds : np.array
        Homogeneous transformation matrix of the curve evaluated at parameter s.
        This is a 'list' of elements of the SE(3) group.
    """
    theta = 2 * np.pi * s
    hds = np.identity(4)  # initialize the homogeneous transformation matrix
    # position = [
    #     r * np.cos(theta),
    #     r * np.sin(theta),
    #     b + d * r**2 * (np.cos(theta) ** 2 - np.sin(theta) ** 2),
    # ]
    position = [
        r * (np.sin(theta) + 2 * np.sin(2 * theta)),
        r * (np.cos(theta) - 2 * np.cos(2 * theta)),
        b + r * (-np.sin(3 * theta)),
    ]

    hds[:3, 3] = np.array(position)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # skew_mat = SmapSO3(axis)
    # orientation = expSO3(theta * skew_mat)
    # orientation = np.eye(3)
    hds[:3, :3] = orientation
    return hds


def hd_derivative(s, r=1, b=1, d=0.2):
    theta = 2 * np.pi * s
    dhds = np.zeros((4, 4))
    # dposition_ds = [
    #     -r * 2 * np.pi * np.sin(theta),
    #     r * 2 * np.pi * np.cos(theta),
    #     d
    #     * r**2
    #     * 2
    #     * (-2 * np.cos(theta) * np.sin(theta) - 2 * np.sin(theta) * np.cos(theta))
    #     * 2
    #     * np.pi,
    # ]

    dposition_ds = [
        r * 2 * np.pi * (np.cos(theta) + 2 * 2 * np.cos(2 * theta)),
        r * 2 * np.pi * (-np.sin(theta) + 2 * 2 * np.sin(2 * theta)),
        r * 2 * np.pi * (-3 * np.cos(3 * theta)),
    ]

    dhds[:3, 3] = np.array(dposition_ds)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    chain = np.pi / 6 * 2 * np.pi * np.cos(2 * np.pi * s)
    # chain = 2 * np.pi
    dorientation_ds = chain * SmapSO3(np.array([1, 0, 0])) @ orientation
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # dorientation_ds = 2 * np.pi * SmapSO3(axis * theta)
    dhds[:3, :3] = dorientation_ds

    # dhds[:3, :3] = 2 * np.pi * np.array(
    #     [
    #         [0, 0, 0],
    #         [0, -np.sin(theta), np.cos(theta)],
    #         [0, -np.cos(theta), -np.sin(theta)],
    #     ]
    # )
    return dhds


def precomputed_hd(curve_fun, n_points, *args, **kwargs):
    """Function that precomputes the curve for each parameter s.

    Parameters
    ----------
    curve_fun : function
        Function that computes the curve. It must be a function that takes as
        first argument the parameter s, and returns a homogeneous transformation
        matrix.
    n_points : int
        Number of points in the curve.
    *args : list
        Arguments of the curve function.
    **kwargs : dict
        Keyword arguments of the curve function.

    Returns
    -------
    precomputed : np.array
        Array with the precomputed curve. The shape is (n_points, 4, 4).
    """
    s = np.linspace(0, 1, num=n_points)
    precomputed = []
    for si in s:
        precomputed.append(curve_fun(si, *args, **kwargs))
    # precomputed = np.array(precomputed)
    return precomputed


def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def progress_bar(i, imax):
    """Prints a progress bar in the terminal.

    Parameters
    ----------
    i : int
        Current iteration.
    imax : int
        Maximum number of iterations.
    """
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()


# Initial conditions
n_points = 3000
n_points = 2000
r, b, d = 0.35, 1, 0.2
r, b = 0.7, 0.4
curve = precomputed_hd(hd, n_points, r, b, d)
curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)
p0 = np.array([-0.1, 0, 0.2]).reshape(-1, 1)
# p0 = curve[0, :3, 3].reshape(-1, 1)  # Start at beginning of curve
p = p0.copy()
# R0 = curve[0, :3, :3]  # Start with orientation of curve
R0 = np.eye(3)  # Start with no rotation
R = R0.copy()
htm0 = pose2htm(p0, R0)
htm = htm0.copy()

xi = np.zeros((6, 1))
v, omega = xi[0:3], xi[3:6]

kn1, kn2 = 1.0, 10.0
kt1, kt2, kt3 = 0.2, 1, 1.0

# %%
# Simulation
dt = 1e-4
T = 30.0
delta = 1e-3
ds = 1e-3
deadband = 0.01
n_steps = int(T / dt)

loop = ControlLoop(htm0, xi, xi.copy(), curve, curve_derivative, adaptiveSys,
                   o_hat, r_hat, kt1, kt2, kt3, kn1, kn2, delta, ds, deadband, dt, T)
loop.simulate()
