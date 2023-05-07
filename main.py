import timeit
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(7)
g = 200000
n = 200
x_dim, y_dim = 6, 6
dims = np.array([x_dim, y_dim])
eps = 0.1
pts = 3


def f(x,y):
    """
    Function to be optimized
    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    
    Returns
    -------
    float
        Value of the function at (x,y) 
    """
    x = 0.002*x
    y = 0.002*y
    a = 3*(1-x)**2*np.exp(-x**2-(y+1)**2)
    b = 10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2)
    c = 1/3*np.exp(-(x+1)**2 - y**2)
    d = 4*(x**2 - y**3)*np.exp(-(x-1)**2 - (y+1)**2)
    e = 0.1*(3*x**4 - 0.2*y**3 + 12)*np.exp(-(x-2.3)**2 - (y-1)**2)
    f = 0.089*(3*x**4 - 0.2*y**3 + 11)*np.exp(-(x+2.5)**2 - (y+1.2)**2)
    g = 0.5*(3*x**4 - 0.2*y**3 + 11)*np.exp(-(x+0.1)**2 - (y+5)**2)
    h = 1.3*(3*x**4 - 0.2*y**3 + 11)*np.exp(-(x+0.8)**2 - (y-4)**2)
    i = 0.012*(3*x**4 - 0.2*y**3 + 11)*np.exp(-(x+4.3)**2 - (y+4.2)**2)
    j = 0.008*(3*x**4 - 0.2*y**3 + 11)*np.exp(-(x+5)**2 - (y-4.2)**2)
    k = 0.02*(3*x**4 - 0.2*y**3 + 11)*np.exp(-(x-3.7)**2 - (y-3.7)**2)
    l = 4*np.sin(x*y)+13+5*np.cos(2*x*y+y)
    humps = (np.cos(x)*np.cos(y))**2 + (np.sin(3*x**2)*np.sin(x+y))**2
    hill = 8*np.exp(-(0.01*x)**2-(0.01*y)**2)
    return humps*(a - b - c + d + e + f + g + h + i + j + k + l) * hill


def nCr(n,r):
    """
    Computes the number of combinations of n things taken r at a time
    Parameters
    ----------
    n : int
        Number of things
    r : int
        Number of things taken at a time
    Returns
    -------
    int
        Number of combinations
        """
    return int(np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r)))


def compute_diff_vecs(population):
    """
    Computes the difference vectors between all pairs of particles in the population
    Parameters
    ----------
    population : numpy.ndarray
        Population of particles
    Returns
    -------
    numpy.ndarray   
        Difference vectors between all pairs of particles
    """
    difference_vectors = np.zeros((nCr(n, 2),2))
    itr = 0
    for i in range(n-1):
        for j in range(i+1, n):
            difference_vectors[itr] = population[i,:] - population[j,:]
            itr += 1
    return difference_vectors


def generate_mutant(population):
    """
    Generates a mutant vector
    Parameters
    ----------
    population : numpy.ndarray
        Population of particles
    Returns
    -------
    numpy.ndarray
        Mutant vector
    """
    diff_vecs = compute_diff_vecs(population)
    mutant_vectors = np.zeros_like(population)
    F = 30*np.random.rand()*np.random.choice([-1,1])
    for i, vec in enumerate(population):
        chosen_vec = diff_vecs[np.random.choice(np.arange(0, n))]
        mutant_vectors[i] = vec + F*chosen_vec + eps*np.random.rand(2)
    return mutant_vectors


def crossover(population, mutant_vectors):
    """
    Performs crossover between the population and mutant vectors
    Parameters
    ----------
    population : numpy.ndarray
        Population of particles
    mutant_vectors : numpy.ndarray
        Mutant vectors
    Returns
    -------
    numpy.ndarray
        Trial vectors
    """
    trial_vectors = np.zeros_like(population)
    for i, vec in enumerate(population):
        trial_vectors[i,0] = vec[0] + np.random.rand()*(mutant_vectors[i,0] - vec[0])
        trial_vectors[i,1] = vec[1] + np.random.rand()*(mutant_vectors[i,1] - vec[1])
    return trial_vectors


def select(population, trial_vectors):
    selection_pool = np.concatenate((population, trial_vectors))
    pool_fitness = f(selection_pool[:,0], selection_pool[:,1])
    print(f'    best fitness: {pool_fitness.max()}')
    tournament_size = 35
    selection = np.zeros_like(population)

    # Tournament selection
    # for _ in range(2):
    #     temp = selection_pool
    #     for i in range(int(2*len(population)/tournament_size)):
    #         match = np.random.choice(np.arange(0,int(len(temp))), tournament_size,replace=False)
    #         match_fitness = [pool_fitness[j] for j in match]
    #         winner = match[np.argmax(match_fitness)]
    #         # selection.append(temp[winner])
    #         selection[i+_*int(2*len(population)/tournament_size)] = temp[winner]
    #         temp = np.delete(temp, match, axis=0)
    for i in range (int(len(population))):
        # print(f'shape of selection pool and fitness: {selection_pool.shape}, {pool_fitness.shape}')
        match = np.random.choice(np.arange(0,int(selection_pool.shape[0])), tournament_size, replace=False)
        match_fitness = [pool_fitness[j] for j in match]
        winner = match[np.argmax(match_fitness)]
        selection[i] = selection_pool[winner]
        selection_pool = np.delete(selection_pool, winner,axis=0)
        pool_fitness = np.delete(pool_fitness, winner)
    return selection


particles = dims/3*(np.random.rand(n,dims.shape[0]) - 0.5)-10
particles = np.ones((n,2))*-1000
particles[:,0] = particles[:,0]

# make list to keep track of centroid of particles
centroids = np.zeros((g,2))
area = 1
areas = np.zeros((g,2))


# EWMA stuff
rho = 0.95 # Rho value for smoothing
s_prev = 0 # Initial value ewma value
s_prev1 = 1 # Initial value ewma value

xmax, xmin = particles[:,0].max(), particles[:,0].min()
ymax, ymin = particles[:,1].max(), particles[:,1].min()
x = np.linspace(-5000,5000,350)
y = np.linspace(-5000,5000,350)

X, Y = np.meshgrid(x,y)
Z = f(X,Y)

tic = timeit.default_timer()
for i in range(g):
    toc = timeit.default_timer()
    if i !=0:
        hull = ConvexHull(particles)
        # use ewma to store convex hull area
        area = hull.volume
    else:
        area = 0
    # Variables to store smoothed area data point
    s_cur = 0
    s_cur_bc = 0

    s_cur = rho* s_prev1 + (1-rho)*area
    s_cur_bc = s_cur/(1-(rho**(i+1)))
    # Append new smoothed value to array
    s_prev1 = s_cur
    areas[i] = np.array([area, s_cur_bc])


    print(f'\n\n{round(toc-tic,3)}s: Generation {i}, number of particles: {particles.shape[0]} (max std: {np.std(particles, axis=0).max():.{pts}f})')
    print(f'      average, max, and min x coordinate: {particles[:,0].mean():.{pts}f}, {particles[:,0].max():.{pts}f}, {particles[:,0].min():.{pts}f}')
    print(f'      average, max, and min y coordinate: {particles[:,1].mean():.{pts}f}, {particles[:,1].max():.{pts}f}, {particles[:,1].min():.{pts}f}')
    print(f'      Convex hull area: {np.array([area])}')

    diffs = compute_diff_vecs(particles)
    mutants = generate_mutant(particles)
    trials = crossover(particles, mutants)
    particles = select(particles, trials)

    # use ewma to compute update to centroid
    
    centroid = particles.mean(axis=0)
    # Variables to store smoothed centroid point
    s_cur = np.zeros((2))
    s_cur_bc = np.zeros((2))

    s_cur = rho* s_prev + (1-rho)*centroid
    s_cur_bc = s_cur/(1-(rho**(i+1)))
    # Append new smoothed value to array
    s_prev = s_cur
    centroids[i,:] = s_cur_bc.reshape(1,2)
    # visualize the location of particeles on a 2d scatter plot, draw contours of the function f(x,y) over the same scatter plot
    plt.figure(1)
    
    plt.scatter(particles[:,0], particles[:,1], c='r',s=2,zorder=2)
    plt.scatter(centroids[i][0], centroids[i][1], c='b',s=3,zorder=3)
    plt.contour(X,Y,Z, 20,zorder=1)
    plt.xlim(-5000,5000)
    plt.ylim(-5000,5000)
    
    plt.pause(0.01)
    plt.clf()

    if i > 10 and np.std(particles, axis=0).max() < 0.5:
        break
toc = timeit.default_timer()
print(f'\n\n\n************************************************************\n')
print(f'Finished optimizing\n\n\nTotal time: {round(toc-tic,3)}s')

mins, maxs = 1500, 2222
x = np.linspace(mins,maxs,2000)
y = np.linspace(mins,maxs,2000)


X, Y = np.meshgrid(x,y)
Z = f(X,Y)


cg_x, cg_y, cg_z = centroids[::4,0], centroids[::4,1], f(centroids[::4,0], centroids[::4,1])
cg_colors=np.arange(cg_y.size)

print(cg_colors)
plt.figure()
plt.scatter(particles[:,0], particles[:,1], c='r',s=2,zorder=2, label='particles')
plt.scatter(cg_x, cg_y, c='b',s=3,zorder=3, cmap='Blues',label='centroid')
plt.xlim(mins,maxs)
plt.ylim(mins,maxs)
plt.legend()
plt.contour(X,Y,Z, 14,zorder=1)

# plot the final contour and particles in a 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5, zorder=1)
ax.scatter(particles[:,0], particles[:,1], f(particles[:,0], particles[:,1]), c='r',s=2,zorder=2)
ax.scatter(cg_x, cg_y, cg_z, c='b',s=3,zorder=3, cmap='Blues')
ax.set_xlim(mins,maxs)
ax.set_ylim(mins,maxs)


plt.figure()
plt.plot(np.arange(0,i), areas[:i,0], label='Convex hull area')
plt.plot(np.arange(0,i), areas[:i,1], label='Smoothed convex hull area')
plt.grid()
plt
plt.xlabel('Generation')
plt.ylabel('Area')
# plt.ylim(0areas[:,1].max())
plt.yscale('log')
plt.show()

# n = 100
# dn = 0.2
# cg_colors      = np.ones((n,3))
# cg_colors[:,0] = (np.arange(n*dn,int(n+n*dn)))
# cg_colors[:,1] = (np.arange(n*dn,int(n+n*dn)))
# cg_colors[:,2] = (np.arange(n*dn,int(n+n*dn)))
# cg_colors = cg_colors/cg_colors.max()
# cg_colors      = cg_colors*np.array([0.43,0.43,1])


# x = np.arange(0,n)
# y = np.arange(0,n)

# plt.scatter(x,y,c=np.arange(0,x.size), cmap='Blues')
# plt.show()
