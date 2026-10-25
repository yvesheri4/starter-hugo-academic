"""Differentiable, linear cold-fluid / Poisson example (periodic 1D).

Units use a FIXED reference density, never the inferred density.
eta = (n_e-n0)/n0; q = n0/n_ref; x in L/(2*pi); t in 1/omega_ref.
eta_t = -u_x, u_t = -E, phi_xx = q*eta, E = -phi_x.
Centered finite differences and an FFT inversion of the discrete Laplacian;
classical RK4 in a differentiable lax.scan. This is not a kinetic model.
"""
from functools import partial
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

N_REF = 1e14  # m^-3; held fixed throughout inference
LENGTH = 0.01  # m
E_CHARGE = 1.602176634e-19
M_E = 9.1093837139e-31
EPS0 = 8.8541878188e-12
OMEGA_REF = (N_REF * E_CHARGE**2 / (M_E * EPS0))**0.5
X_REF = LENGTH / (2 * jnp.pi)
E_REF = M_E * OMEGA_REF**2 * X_REF / E_CHARGE


def derivative(a):
    dx = 2 * jnp.pi / a.shape[-1]
    return (jnp.roll(a, -1, axis=-1) - jnp.roll(a, 1, axis=-1)) / (2*dx)


def potential(eta, q):
    n = eta.shape[-1]
    dx = 2 * jnp.pi / n
    kdx = 2 * jnp.pi * jnp.fft.fftfreq(n)
    eigenvalues = -4 * jnp.sin(kdx/2)**2 / dx**2
    # Set the mean potential to zero without evaluating a division by zero.
    safe = eigenvalues.at[0].set(1.)
    coeff = (q * jnp.fft.fft(eta, axis=-1) / safe).at[..., 0].set(0.)
    return jnp.fft.ifft(coeff, axis=-1).real


def field(eta, q):
    return -derivative(potential(eta, q))


def rhs(state, q):
    eta, u = state
    return jnp.stack((-derivative(u), -field(eta, q)))


@partial(jax.jit, static_argnames=('n', 'steps', 'dt', 'amplitude', 'mode'))
def simulate(q, n=64, steps=720, dt=0.025, amplitude=0.02, mode=1):
    """Return [time, variable (eta,u), position], including initial state.

    Use positive q and resolved smooth modes, 0 < mode < n/2. Mean eta is
    zero (neutrality). Collocated differences cannot resolve Nyquist modes.
    """
    x = 2*jnp.pi*jnp.arange(n)/n
    state = jnp.stack((amplitude*jnp.cos(mode*x), jnp.zeros(n)))
    def step(y, _):
        a = rhs(y, q)
        b = rhs(y + dt*a/2, q)
        c = rhs(y + dt*b/2, q)
        d = rhs(y + dt*c, q)
        yn = y + dt*(a+2*b+2*c+d)/6
        return yn, yn
    _, states = jax.lax.scan(step, state, None, length=steps)
    return jnp.concatenate((state[None], states), axis=0)


def probe(q, n=64, steps=720, dt=0.025, amplitude=0.02):
    """Electric field at x=L/4; uniform sample times independent of q."""
    return field(simulate(q, n=n, steps=steps, dt=dt, amplitude=amplitude)[:, 0], q)[:, n//4]


def analytical_probe(q, times, amplitude=0.02):
    """Independent continuum solution for the mode-one initial condition."""
    return -q * amplitude * jnp.cos(jnp.sqrt(q)*times)


def energy(states, q):
    """Conserved quadratic discrete energy, with edge-centered field energy.

    The forward potential difference is the edge field; the dynamics uses
    its centered average. This energy matches the discrete Poisson operator.
    """
    phi = potential(states[:, 0], q)
    edge_field = -(jnp.roll(phi, -1, axis=-1)-phi)/(2*jnp.pi/phi.shape[-1])
    return .5*jnp.mean(q*states[:, 1]**2 + edge_field**2, axis=-1)
