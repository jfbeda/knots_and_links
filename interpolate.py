
import numpy as np
import sympy as sp


def smooth_and_resample_3d(coords, n_points, smoothness):
    coords = np.array(coords, dtype=float)

    if not np.allclose(coords[0],coords[-1]):
        coords = np.vstack((coords,coords[0]))

    deltas = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1)) # list of lengths of edges
    cumdist = np.insert(np.cumsum(seg_lengths), 0, 0)

    #Interpolate to create equally spaced points 
    total_len = cumdist[-1]
    uniform_d = np.linspace(0, total_len, n_points)
    x_uniform = np.interp(uniform_d, cumdist, coords[:, 0])
    y_uniform = np.interp(uniform_d, cumdist, coords[:, 1])
    #z_uniform = np.interp(uniform_d, cumdist, coords[:, 2])
    scale_factor = 3 
    z_uniform = np.interp(uniform_d, cumdist, coords[:, 2]) * scale_factor

    def neighbor_average(arr, iterations):
        arr = arr.copy()
        n = len(arr)
        for _ in range(iterations):
            arr_new = arr.copy()
            for i in range(n):
                arr_new[i] = (arr[(i-1)%n] + arr[i] + arr[(i+1)%n]) / 3
            arr = arr_new
        return arr

    if smoothness > 1:
        x_smooth = neighbor_average(x_uniform, smoothness)
        y_smooth = neighbor_average(y_uniform, smoothness)
        z_smooth = neighbor_average(z_uniform, smoothness)
    else:
        x_smooth, y_smooth, z_smooth = x_uniform, y_uniform, z_uniform

    return np.column_stack((x_smooth, y_smooth, z_smooth))

def resample(coords, n_points):
    c_open = np.asarray(coords, float)
    assert len(c_open) > 2, "coords must be longer than 2"
    assert not np.allclose(c_open[0], c_open[-1]), "coords must be an open list"

    c = np.vstack([c_open, c_open[0]]) # close the input coordinate list
    d = np.diff(c, axis=0)
    seg = np.sqrt((d*d).sum(1))
    cum = np.insert(np.cumsum(seg), 0, 0.0)
    total = cum[-1]
    assert total > 0, "curve has zero total length"

    u = np.linspace(0, total, n_points + 1)[:-1]  

    out = []
    for dist in u:
        i = np.searchsorted(cum, dist, side="right") - 1   # segment index
        t = (dist - cum[i]) / seg[i] if seg[i] > 0 else 0.0
        out.append(c[i] + t * d[i])                        # point along segment i

    return np.asarray(out)


def resample_symbolic(coords, m, prec=80):
    P = [sp.Matrix(p) for p in coords]
    assert len(P) > 2
    assert P[0] != P[-1]

    P = P + [P[0]]

    # segment vectors and exact segment lengths
    d   = [P[i+1] - P[i] for i in range(len(P)-1)]
    seg = [sp.sqrt(v.dot(v)) for v in d]

    # cumulative arclengths (exact expressions)
    s = [sp.Integer(0)]
    for Lk in seg:
        s.append(s[-1] + Lk)
    Ltot = s[-1]

    # target distances (exact multiples of total length)
    U = [sp.Rational(j, m) * Ltot for j in range(m)]

    # numeric versions for robust segment lookup
    sN = [sp.N(x, prec) for x in s]

    Q = []
    for u in U:
        uN = sp.N(u, prec)

        # find i with s[i] <= u < s[i+1] (using numeric comparisons)
        i = max(k for k in range(len(sN)-1) if sN[k] <= uN)

        # exact interpolation once i is chosen
        t = (u - s[i]) / seg[i] if seg[i] != 0 else 0
        q = P[i] + t * d[i]
        Q.append(sp.simplify(q))

    return Q

# def resample(coords, n_points):
#     c_open = np.asarray(coords, float)
#     assert len(c_open) > 2, "coords must be longer than 2"
#     assert not np.allclose(c_open[0], c_open[-1]), "coords must be an open list"

#     c = np.vstack([c_open, c_open[0]])          # close implicitly
#     d = np.diff(c, axis=0)
#     seg = np.sqrt((d*d).sum(1))
#     cum = np.insert(np.cumsum(seg), 0, 0.0)
#     total = cum[-1]
#     assert total > 0, "curve has zero total length"

#     u = np.linspace(0, total, n_points + 1)[:-1]  # n_points, no duplicated endpoint

#     out = []
#     for dist in u:
#         i = np.searchsorted(cum, dist, side="right") - 1   # segment index
#         t = (dist - cum[i]) / seg[i] if seg[i] > 0 else 0.0
#         out.append(c[i] + t * d[i])                        # point along segment i

#     return np.asarray(out)

# # define algebraic numbers
# sqrt3  = sp.sqrt(3)
# sqrt22 = sp.sqrt(22)

# # define points as SymPy column vectors
# A = sp.Matrix([-1, 0, 0])
# B = sp.Matrix([ 1, 0, 0])

# C = sp.Matrix([
#     (13 - 4*sqrt22)/25,
#     sp.Rational(6, 5),
#     -4*(sqrt22 - 1)*sqrt3/25
# ])

# D = sp.Matrix([
#     sp.Rational(-1, 2),
#     sp.Rational(3, 5),
#     sqrt3/2
# ])

# E = sp.Matrix([
#     sp.Rational(1, 2),
#     sp.Rational(3, 5),
#     -sqrt3/2
# ])

# F = sp.Matrix([
#     (-13 + 4*sqrt22)/25,
#     sp.Rational(6, 5),
#     4*(sqrt22 - 1)*sqrt3/25
# ])

# symbolic_output = resample_symbolic([A,B,C,D,E,F], 7)

# trefoil_points = np.array([
#     [-1.0, 0.0, 0.0],
#     [ 1.0, 0.0, 0.0],
#     [(13 - 4*np.sqrt(22))/25, 6/5, -(4/25)*(-1 + np.sqrt(22))*np.sqrt(3)],
#     [-1/2, 3/5,  (1/2)*np.sqrt(3)],
#     [ 1/2, 3/5, -(1/2)*np.sqrt(3)],
#     [(-13 + 4*np.sqrt(22))/25, 6/5, (4/25)*(-1 + np.sqrt(22))*np.sqrt(3)],
# ])
