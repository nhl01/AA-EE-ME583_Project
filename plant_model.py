import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
mQ = 0.21    # quadrotor mass [kg]
mL = 0.06    # load mass [kg]
l = 0.6      # cable length [m]
g = 9.81     # gravity [m/s^2]
M = mQ + mL  # total mass

e3 = np.array([0.0, 0.0, 1.0])   # z axis (points DOWN in the inertial frame)


def S(x):
    return np.array([[0.0, -x[2], x[1]],[x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


# State vector x has 21 numbers:
#   x[0:3]   = pL  (load position)
#   x[3:6]   = vL  (load velocity)
#   x[6:9]   = q   (cable direction, unit vector)
#   x[9:12]  = w   (cable angular velocity)
#   x[12:21] = R   (quadrotor rotation matrix, stored as 9 numbers)
#
# Input u has 4 numbers:
#   u[0]   = T      (total thrust)
#   u[1:4] = Omega  (body angular velocity)

def dynamics(x, u):
    # unpack the state
    pL = x[0:3]
    vL = x[3:6]
    q = x[6:9]
    w = x[9:12]
    R = x[12:21].reshape(3, 3)

    # unpack the input
    T = u[0]
    Omega = u[1:4]

    # thrust force vector in the inertial frame
    F = -T * R @ e3

    # part of F that points along the cable:  P_q(F) = q q^T F
    PqF = q * (q @ F)

    # equations from paper
    pL_dot = vL
    vL_dot = PqF / M - (mQ * l / M) * (w @ w) * q + g * e3
    q_dot = -S(q) @ w
    w_dot = -S(q) @ F / (mQ * l)
    R_dot = R @ S(Omega)

    # pack everything back into one vector
    return np.concatenate([pL_dot, vL_dot, q_dot, w_dot, R_dot.ravel()])


def control_input(t, x):
    # returns u = [T, Omega_x, Omega_y, Omega_z] at time t.
    # Here: hover thrust, plus a roll-rate kick between 0.2 s and 0.3 s.
    # (Later, the backstepping controller goes in here and can use the state x.)
    T_hover = M * g
    u = np.array([T_hover, 0.0, 0.0, 0.0])
    if 0.2 <= t < 0.3:
        u[1] = 0.5
    return u


def rhs(t, x):
    # the function solve_ivp calls: it must return dx/dt given (t, x)
    u = control_input(t, x)
    return dynamics(x, u)


def cable_tension(x, u):
    # tension in the cable from eq. (4). It must stay > 0 or the cable goes slack.
    q = x[6:9]
    vL_dot = dynamics(x, u)[3:6]
    return q @ (mL * (g * e3 - vL_dot))


# Test: hover, then give a short roll-rate kick
if __name__ == "__main__":
    t_end = 10.0

    # initial state: load 1 m above the origin, cable hanging straight down,
    x0 = np.zeros(21)
    x0[0:3] = [0, 0, -1.0]        # pL  (z is down, so -1 means 1 m up)
    x0[6:9] = [0, 0, 1.0]         # q
    x0[12:21] = np.eye(3).ravel() # R

    # solve_ivp does all the integrating.
    t_log = np.linspace(0, t_end, 5001)
    sol = solve_ivp(rhs, (0, t_end), x0, method="DOP853",
                    t_eval=t_log, rtol=1e-9, atol=1e-12, max_step=0.01)

    X = sol.y.T                   # one row per time step, 21 columns
    pL_log = X[:, 0:3]
    q_log = X[:, 6:9]

    # cable tension at every saved time
    TL_log = np.zeros(len(t_log))
    for i in range(len(t_log)):
        u = control_input(t_log[i], X[i])
        TL_log[i] = cable_tension(X[i], u)

    print("solver success:", sol.success)
    print("min cable tension =", TL_log.min(), "N (must be > 0)")

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t_log, pL_log)
    ax[0].set_ylabel("load position pL [m]")
    ax[0].legend(["x", "y", "z"])
    ax[1].plot(t_log, q_log)
    ax[1].set_ylabel("cable direction q")
    ax[1].legend(["qx", "qy", "qz"])
    ax[2].plot(t_log, TL_log)
    ax[2].set_ylabel("cable tension [N]")
    ax[2].set_xlabel("time [s]")
    for a in ax:
        a.grid(True)

    fig.tight_layout()
    plt.show()

    plt.plot(t_log, pL_log)
    plt.xlabel("time [s]")
    plt.ylabel("load position pL [m]")
    plt.legend(["x", "y", "z"])
    plt.grid(True)
    plt.show()
