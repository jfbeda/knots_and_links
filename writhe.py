import numpy as np
from numba import prange, njit
import matplotlib.pyplot as plt

def compute_writhe_matrix_slow(knot_coords_1, knot_coords_2):
    """
    Input: knot_coors_1/knot_coords_2 are a set of (N,3) arrays of coordinates where the knot has N points in it.
    """
    assert knot_coords_1.shape == knot_coords_2.shape, "Cannot compute writhe density between two knots of different shapes"
    N = len(knot_coords_1)

    # Extend the coordinate knots:
    r1 = np.vstack((knot_coords_1,knot_coords_1[0]))
    r2 = np.vstack((knot_coords_2,knot_coords_2[0]))

    dr1 = np.zeros_like(knot_coords_1)
    dr2 = np.zeros_like(knot_coords_2)

    for i in range(N):
        dr1[i] = r1[i+1]-r1[i]
        dr2[i] = r2[i+1]-r2[i]

    writhe_density = np.zeros((N,N))

    if np.allclose(r1,r2): # i.e. if the two knots are the same
        for i in range(N):
            for j in range(N):
                if i != j:
                    writhe_density[i,j] = (np.dot(np.cross(dr1[i],dr2[j]),(r1[i]-r2[j])))/(np.linalg.norm(r1[i]-r2[j])**3)
                    if i == j+1:
                        print(f"cross: {np.cross(dr1[i],dr2[j])}, difference {r1[i]-r2[j]}, norm {np.linalg.norm(r1[i]-r2[j])}")

    else:
        for i in range(N):
            for j in range(N):
                assert not np.isclose(np.linalg.norm(r1[i]-r2[j]),0), "Can't divide by zero. Dispite the two knots fed to writhe density being different, they seem to overlap at a point (There's no way this error message is actually gonna be called)"
                writhe_density[i,j] = (np.dot(np.cross(dr1[i],dr2[j]),(r1[i]-r2[j])))/(np.linalg.norm(r1[i]-r2[j])**3)

    return writhe_density

def writhe(knot_coords_1, knot_coords_2):
    return np.sum(compute_writhe_matrix(knot_coords_1, knot_coords_2))/(4*np.pi)

@njit()
def compute_writhe_matrix(ring1, ring2):
    '''
    General computation over all segments.

    Returns unnormalized_writhe_matrix (i..e doesn't divide by 4pi)
    '''

    matrix = np.zeros((ring1.shape[0],ring2.shape[0]))
    # Loop on the first ring
    for i in prange(ring1.shape[0]):
        # Loop on the second ring
        for j in prange(ring2.shape[0]):
            matrix[i,j] += compute_kernel_chord(ring1, ring2, i, j)

    
    return matrix

    
@njit()
def vec_cross(a, b):
    n = np.cross(a, b)
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-9:
        return n / norm_n
    return np.array((0.0, 0.0, 0.0), dtype=np.float64)

@njit()
def clip1n1(x):
    return np.minimum(1.0, np.maximum(-1.0, x))

@njit()
def compute_kernel_chord(ring1, ring2, i, j):
    '''
    Klenin computation of writhe
    '''

    P = ring1.shape[0]
    one = ring1[np.mod(i-1,ring1.shape[0]),:]
    three = ring2[np.mod(j-1,ring2.shape[0]),:]
    two = ring1[np.mod(i,ring1.shape[0]),:]
    four = ring2[np.mod(j,ring2.shape[0]),:]

    if i == j:
        if np.isclose(np.linalg.norm(ring1[i]-ring2[j]),0):
            return 0
        
    if np.abs(i-j) <= 1.1 or np.abs(i-j) == 50:
        return 0
    
    # if (j-i)%P in (1, P-1):
    #     # Build Fulton-Macpherson compactification on edges which share vertices.
    #     # https://www.jstor.org/stable/pdf/2946631.pdf
    #     one, two, three, four = FM_compactification(one, two, three, four)

    # Standard Klenin techniques
    r12=two-one
    r34=four-three
    r23=three-two
    r13=three-one
    r14=four-one
    r24=four-two

    n1 = vec_cross(r13, r14)
    n2 = vec_cross(r14, r24)
    n3 = vec_cross(r24, r23)
    n4 = vec_cross(r23, r13)

    n1n2=clip1n1(np.dot(n1,n2))
    n2n3=clip1n1(np.dot(n2,n3))
    n3n4=clip1n1(np.dot(n3,n4))
    n4n1=clip1n1(np.dot(n4,n1))

    triple = float(np.dot(np.cross(r34,r12),r13))
    sign = 0.0 if abs(triple) < 1e-18 else np.sign(triple)
    
    omega = (np.arcsin(n1n2) + np.arcsin(n2n3) + np.arcsin(n3n4) + np.arcsin(n4n1)) * sign
    return omega



def plot_writhe_matrix(W, title="Writhe Matrix"):
    """
    Plots the NxN writhe matrix using color to denote magnitude.

    Parameters
    ----------
    W : np.ndarray
        NxN matrix representing writhe values.
    title : str, optional
        Title for the plot.
    """
    if not isinstance(W, np.ndarray):
        raise TypeError("W must be a numpy array.")
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix (NxN).")
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(W, cmap='viridis', origin='lower')
    plt.colorbar(im, label="Writhe Magnitude")
    plt.title(title)
    plt.xlabel("Index i")
    plt.ylabel("Index j")
    plt.tight_layout()
    plt.show()