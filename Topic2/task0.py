from typing import Iterable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp


def sezr_model_gamma(
    t: float, y: Iterable, beta_0: float, lam: float, sigma: float, gamma: float, N: int
) -> Iterable:
    """
    A function for creating a SEZR model for the Ebola outbreaks.

    :param float t: Time
    :param float beta_0: Initial infection rate
    :param float lam: Rate of exponential decay
    :param float sigma: Rate from exposed to infectious
    :param float gamma: Rate from infectious to recovered
    :param int N: Total population size
    :return: Numpy Array of derivatives [dS/dt, dE/dt, dZ/dt, dR/dt, dC/dt] representing the
          rate of change for each compartment.
    """
    S, E, Z, R = y

    beta = beta_0 * np.exp(-lam * t)

    dS = -beta * (S * Z / N)
    dE = beta * (S * Z / N) - sigma * E
    dZ = sigma * E - gamma * Z
    dR = gamma * Z

    return np.array([dS, dE, dZ, dR])


def step(dt, f, t, S, method, *args, **kwargs):
    """
    This function computes a single time step using either Euler's method or the second-order Runge-Kutta method (RK2).

    :Parameters:
    dt: float
        Time step
    f: Callable
        function defining the ODE
    S: float
        Susceptible population at time t
    t: float
        Current time
    method: str
        'Euler' or 'RK2'

    :raises ValueError: if method is not implemented

    :Returns:
    float
        The estimated change in the susceptible population over the timestep dt
    """
    if method == "Euler":
        return dt * f(t, S, *args, **kwargs)
    elif method == "RK2":
        k1 = np.array(dt * f(t, S, *args, **kwargs))
        return dt * f(t + dt * 0.5, S + 0.5 * k1, *args, **kwargs)
    else:
        raise ValueError(method + ": not implemented")


def ode_solver(
    h: float,
    method: str,
    t0: float,
    t_end: float,
    y0: float,
    f,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function uses Eulers method with richardson extrapolation to solve ODEs with constant or adaptive step size.

    :params:
    h: float
        step size (if h=0, use richardson extrapolation)
    method: str
        "Euler" or "RK2"
    t0: float
        starting time
    t_end: float
        end time
    y0: float
        initial condition
    f: ode function (rhs) f(t,c,tau)
    *args, **kwargs: additional arguments for f

    :raises ValueError: if step size h is less than zero

    :returns:
    tuple(y, t)
    """
    if h < 0:
        raise ValueError("Step size h must be greater than zero.")

    if method == "Euler":
        p = 1
    elif method == "RK2":
        p = 2
    else:
        raise ValueError(method + ": not implemented")

    y0 = np.asarray(y0)
    y = []
    y.append(y0.copy())

    if h > 0:
        t = np.arange(t0, t_end + h, h)  # Time array from 0 to t_end with step h
        for i in range(1, len(t)):
            S_new = step(h, f, t[i - 1], y[-1], method, *args, **kwargs)
            y.append(y[-1] + S_new)

    elif h == 0:  # richardson extrapolation
        eps = 1e-5
        t = []
        t.append(t0)
        dt_old = 1e-4
        while t[-1] < t_end:
            y_old = y[-1]
            eps_calc = 10 * eps  # just to enter while loop
            while eps_calc > eps:
                dt = dt_old
                y_long = y_old + step(dt, f, t[-1], y_old, method, *args, **kwargs)
                y_half = y_old + step(
                    0.5 * dt, f, t[-1], y_old, method, *args, **kwargs
                )
                y_two_half = y_half + step(
                    0.5 * dt, f, t[-1] + 0.5 * dt, y_half, method, *args, **kwargs
                )
                eps_calc = np.linalg.norm((y_long - y_two_half) / (2**p - 1))
                eps_calc = max(
                    eps_calc, 1e-15
                )  # Prevents the whole function from blowin up if we divide by zero in prev line, fixes the "perfect step" problem
                dt_old = dt * (eps / eps_calc) ** (1 / (p + 1))
            y.append((2**p * y_two_half - y_long) / (2**p - 1))
            t.append(t[-1] + dt)

    return np.array(y), np.array(t)


def simulate_and_plot(filename: str, countryName: str) -> None:
    """
    Function for simulating a fitted model and plotting it.

    :param: str filename: Name of file
    :param: str countryName: Name of country used for figure title

    :return: None, will plot.
    """

    data = pd.read_csv(filename, sep="\t")

    startIndex = 0
    t0 = 0

    if countryName.lower() == "liberia":
        startIndex = np.where(data["NumOutbreaks"] > 5)[0][0]
        t0 = data["Days"].iloc[startIndex]

    t = data["Days"].iloc[startIndex:] - t0
    new_cases_data = data["NumOutbreaks"].iloc[startIndex:]

    def non_linear_func(t, beta0, lam):
        """
        Simulates a time-decaying SEZR-like model and returns the interpolated
        cumulative cases at specified time points.

        :param array_like t: Time
        :param float beta_0: Initial infection rate
        :param float lam: Rate of exponential decay

        :return array_like: Interpolated cumulative new cases at time points t
        """
        N = 1e7
        sigma = 1 / 9
        gamma = 1 / 10
        S0, E0, Z0, R0 = N - 1, 0, 1, 0
        y0 = np.array([S0, E0, Z0, R0])

        y, t_sim = ode_solver(
            1, "Euler", 0, max(t), y0, sezr_model_gamma, beta0, lam, sigma, gamma, N
        )
        S, _, Z, _ = y.T

        beta_t = beta0 * np.exp(-lam * t_sim)
        new_cases = beta_t * (S * Z / N)

        return np.interp(t, t_sim, new_cases)

    p0 = [0.18, 0.00185]
    bds = ((0.0, 0.0), (np.inf, np.inf))
    popt, pcov = sp.optimize.curve_fit(
        non_linear_func, t, new_cases_data, p0=p0, bounds=bds
    )

    N = 1e7
    sigma = 1 / 9
    gamma = 1 / 10
    S0, E0, Z0, R0 = N - 1, 0, 1, 0
    y0 = np.array([S0, E0, Z0, R0])

    y_fit, t_sim = ode_solver(
        1, "Euler", 0, max(t), y0, sezr_model_gamma, popt[0], popt[1], sigma, gamma, N
    )

    S_fit, _, Z_fit, _ = y_fit.T

    beta_t_fit = popt[0] * np.exp(-popt[1] * t_sim)
    new_cases_fit = beta_t_fit * (S_fit * Z_fit / N)

    new_cases_fit_interp = np.interp(t, t_sim, new_cases_fit)

    _, ax1 = plt.subplots(figsize=(12, 6))

    ax1.scatter(
        t,
        new_cases_data,
        marker="o",
        linestyle="-",
        label="Real data",
    )
    ax1.plot(t, new_cases_fit_interp, color="orange", label="Model")

    ax1.set_xlabel("Days since first outbreak")
    ax1.set_ylabel("Number of outbreaks")

    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=9)
    plt.title(f"Data vs Model: Ebola outbreaks in {countryName}")
    plt.show()
