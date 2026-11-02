# Plate motion (Text S2): GAD direction in a rock frame rotating about an Euler pole.
# speed = surface speed at 90 deg from the pole; rotation referred to the MRS midpoint.
import math
import numpy as np

EARTH_RADIUS_KM = 6371.0
REFERENCE_AGE_MA = 0.5 * (485.0 + 466.1)
PALEOLATITUDES_DEG = np.array([-75., -60., -45., -30., -15., 0., 15., 30., 45., 60., 75.])


def fibonacci_axes(n):
    i = np.arange(n, dtype=float)
    ga = math.pi * (3 - math.sqrt(5))
    z = 1 - 2 * (i + .5) / n
    r = np.sqrt(np.maximum(0, 1 - z * z))
    p = ga * i
    return np.c_[r * np.cos(p), r * np.sin(p), z]


def rodrigues(axis, angles, v):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    a = np.asarray(angles, float); v = np.asarray(v, float)
    if v.ndim == 1:
        v = np.broadcast_to(v, (a.size, 3))
    c = np.cos(a)[:, None]; s = np.sin(a)[:, None]
    cross = np.cross(np.broadcast_to(axis, v.shape), v)
    dot = (v @ axis)[:, None]
    return v * c + cross * s + axis[None, :] * dot * (1 - c)


def ref_geom(lat_deg):
    la = np.deg2rad(lat_deg)
    r = np.array([np.cos(la), 0, np.sin(la)])
    n = np.array([-np.sin(la), 0, np.cos(la)]); e = np.array([0, 1, 0]); d = -r
    return r, np.column_stack((n, e, d))


def gad_response(ages, lat_deg, axis, speed):
    # unit field direction (N, E, D) in the rock frame at each age
    r0, E0 = ref_geom(lat_deg)
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    omega = (speed * 10.0) / EARTH_RADIUS_KM
    ang = omega * (np.asarray(ages) - REFERENCE_AGE_MA)
    q = rodrigues(axis, -ang, np.array([0., 0., 1.]))      # spin axis in the rock frame
    b = q - 3.0 * (q @ r0)[:, None] * r0[None, :]
    b /= np.linalg.norm(b, axis=1)[:, None]
    m = b @ E0
    m /= np.linalg.norm(m, axis=1)[:, None]
    return m
