import numpy as np


def initialize_circle(N_particles, radius, N2):
    """
    Generates a set of N particles randomly along a circle of the given radius 
    
    Inputs:
        N_particles: the number of particles in the swarm
        radius: the radius of the circle from the center of the grid
        N2: the index of the center of the square grid
    
    Returns:
        P: a (2 x N_particles) array of particle locations as grid indices 
    """
    angles = 2 * np.pi * np.random.random(N_particles)
    X = radius * np.sin(angles) + N2
    Y = radius * np.cos(angles) + N2
    return np.stack([X,Y]).astype(np.int)



def random_step(P, N_particles, N2, full_idx):
    """
    Updates an array of particles by moving them by 1 grid cell in a random direction 
    
    Inputs:
        P: array of particle locations
        N_particles: the number of particles in the swarm
        N2: the index of the center of the square grid
        full_idx: the indices np.array(0,...,N_particles), passed to avoid regenerating
    
    Returns:
        No return variables (updates P in place)
    """
    
    step = np.random.randint(0,4,N_particles)
    dim = step // 2
    direction = 2 * ( step % 2 ) - 1
    P[dim,full_idx] += direction

def check_out_of_bounds(P, lower, upper, radius, N2):
    """
    Replaces any particles that leave a rectangular boundary with random new particles
    Improves performance since diffusion can carry particles too far away
    
    Inputs:
        P: an array of indices for all active diffusing particles
        lower: the minimum grid index allowed for either dimension
        upper: the maximum grid index allowed for either dimension
        radius: the distance from the center for any regenerated particles
        N2: the index of the center of the square grid
    
    Returns:
        N_replaced: the number of replaced particles 
    """
    idx  = (P[0,:] <= lower) + (P[0,:] >= upper ) + (P[1,:] <= lower) + (P[1,:] >= upper )
    idx = idx > 0
    N_replaced = np.sum(idx)
    if N_replaced > 0:
        P[:,idx] = initialize_circle(N_replaced, radius, N2)
    return N_replaced
    
    
def check_neighbors(grid, P, R, N2, diagonals, r_ratio):
    """
    Merges (and replaces) any diffusing particles that contact a grid particle
    
    Inputs:
        grid: the grid of static (aggregated) particles
        P: an array of indices for all active diffusing particles
        R: the radius of the circle from the center of the grid
        N2: the index of the center of the square grid
        diagonals: if True, uses 8-way nearest neighbors (otherwise 4)
        r_ratio: a scaling factor for the radius where particles are added
    
    Returns:
        N_matched: the number of particles added to the grid (and replaced)
        R: an updated radius based on the farthest particle in the cluster
    """
    I, J = P[0,:], P[1,:]
    nn = grid[I-1,J] + grid[I+1,J] + grid[I,J-1] + grid[I,J+1]
    if diagonals:
        nn += grid[I-1,J-1] + grid[I-1,J+1] + grid[I+1,J-1] + grid[I+1,J+1]
    nn = nn > 0
    N_matched = np.sum(nn)
    if N_matched > 0:
        grid[ I[nn], J[nn] ] = 1
        X, Y = (I[nn] - N2), (J[nn] - N2)
        R_matched = np.ceil( np.sqrt( X**2 + Y**2 ) ).astype(int)
        R = max( R, np.max(R_matched) )
        P[:,nn] = initialize_circle(N_matched, R*r_ratio, N2)
    return N_matched, R


def DLA(N, N_particles, N_steps, R0, diagonals, r_ratio = 1.5):
    """
    Computes diffusion-limited aggregation for a grid of size N
    
    Inputs:
        N_particles: the number of particles in the swarm
        radius: the radius of the circle from the center of the grid
        N2: the index of the center of the square grid
        diagonals: if True, uses 8-way nearest neighbors (otherwise 4)
        r_ratio: a scaling factor for the cluster radius used to determine particle regeneration distance and set the stray particle limit
    
    Returns:
        P: a (2 x N_particles) array of particle locations as grid indices 
        R: the cluster radius at the end of the run
    """
    N2 = N//2
    R = int(R0)
    
    grid = np.zeros( (N,N), dtype=np.bool )
    grid[N2,N2] = 1

    P = initialize_circle(N_particles, R, N2)
    full_idx = np.arange(0,N_particles)
    
    for i in range(N_steps):
        R = min(R, N2 - 10)
        lower = N2 - R * r_ratio
        upper = N2 + R * r_ratio

        if lower < 2 or upper > N-3:
            break
        
        if (i+1) % (N_steps//10) == 0: 
            print(".", end="")
        
        random_step(P, N_particles, N2, full_idx)
        N_out = check_out_of_bounds(P, lower, upper, R*r_ratio, N2)
        N_matched, R = check_neighbors(grid, P, R, N2, diagonals, r_ratio)
    
    print("")
    I, J = np.nonzero(grid)
    X, Y = (I - N2), (J - N2)
    R_matched = np.sqrt( X**2 + Y**2 ) 
    R = np.max(R_matched) 
    
    return grid, R