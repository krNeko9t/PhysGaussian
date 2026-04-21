"""Material preset factories.

Each function returns a plain ``dict`` that is stored as
``PartConfig.material``.  The function signature exposes commonly
tuned parameters for IDE auto-complete; ``**kw`` passes through any
backend-specific extras.
"""


def rubber(
    density: float = 1100.0,
    mu: float = 0.5,
    E: float = 1e6,
    nu: float = 0.45,
    **kw,
) -> dict:
    return dict(density=density, mu=mu, E=E, nu=nu, **kw)


def sand(
    density: float = 2000.0,
    E: float = 5e7,
    nu: float = 0.3,
    friction_angle: float = 30.0,
    **kw,
) -> dict:
    return dict(density=density, E=E, nu=nu, friction_angle=friction_angle, **kw)


def jelly(
    density: float = 200.0,
    E: float = 1e5,
    nu: float = 0.4,
    **kw,
) -> dict:
    return dict(density=density, E=E, nu=nu, **kw)


def soft_body(
    density: float = 300.0,
    k_mu: float = 1e5,
    k_lambda: float = 1e5,
    k_damp: float = 1e-3,
    **kw,
) -> dict:
    return dict(physics="soft", density=density, k_mu=k_mu,
                k_lambda=k_lambda, k_damp=k_damp, **kw)


def rigid_body(
    density: float = 1000.0,
    mu: float = 0.5,
    collision_geometry: str = "convex_hull",
    **kw,
) -> dict:
    return dict(physics="rigid", density=density, mu=mu,
                collision_geometry=collision_geometry, **kw)
