#!/usr/bin/env python
# coding: utf-8

# ## Impact Visualizer
# The following code cells operate to generate plots in both 2D and 3D of the impact simulation. 
# In order to operate this file, it must be located in the location of the impact files, for me that is miluphcuda/examples/impact/initials

# In[9]:


# Imports
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import h5py
import plotly.express as px
import os
import sys
import h5py
import scipy.spatial


# In[5]:


## Test reading file 
h5f = h5py.File('impact.0095.h5', 'r')
print(h5f)

# Reads the variables from the file
variables = list(h5f.keys())
print(variables)


# In[6]:


class sph_particles:
    """
    Class to read SPH particle distributions from h5 files
    """

    def __init__(self, filename="impact.0095.h5", path=os.getcwd()):
        self.path = path
        self.h5fn = filename
        print("Reading from directory: {0}".format(self.path))

    """
        returns an array of strings of available data in the h5 file
    """
    def available_data(self, verbose=True):
        try:
            h5f = h5py.File(self.h5fn, 'r')
        except:
            print("Error: Cannot open {0} in {1}".format(self.h5fn, self.path))
            return

        variables = list(h5f.keys())
        h5f.close()
        if verbose:
            print("Found the following data in the h5 file    ")
            for var in variables:
                print("{0} ".format(var), end="")
            print("")
        return variables

    """
        returns dict containing the data from the file
    """
    def read_data(self, keys=('x',)):
        variables = self.available_data(verbose=False)
        d = {}
        try:
            h5f = h5py.File(self.h5fn, 'r')
        except:
            print("Error: Cannot open {0} in {1}".format(self.h5fn, self.path))
            return
        for k in keys:
            if k not in variables:
                print(color.BOLD + "Warning: " + color.END + "Cannot find data {1} in file {0}".format(self.h5fn,k))
            else:
                print("Reading {0}".format(k))
                d[k] = h5f[k][...]
        h5f.close()
        return d


    """
        returns information about the particles data
    """
    def print_info(self, data):
        print("Geometry of the particle distribution:")
        x = data['x']
        xmin = np.min(x[:,0])
        xmax = np.max(x[:,0])
        print("x: {0} -> {1}".format(xmin,xmax))
        try:
            xmin = np.min(x[:,1])
            xmax = np.max(x[:,1])
            print("y: {0} -> {1}".format(xmin,xmax))
        except:
            print("Only 1D data")
        try:
            xmin = np.min(x[:,2])
            xmax = np.max(x[:,2])
            print("z: {0} -> {1}".format(xmin,xmax))
        except:
            print("Only 2D data")

        print("Properties of the loaded data, minima and maxima, for vectors and arrays for all entries:")
        for k in data.keys():
                d = data[k]
                lenbla = len(d.shape)
                if len(d.shape) == 1:
                    mymin = np.min(d)
                    mymax = np.max(d)
                if len(d.shape) == 2:
                    entries = d.shape[1]
                    mymin = 1e300
                    mymax = -1e300
                    for i in range(entries):
                        myd = d[:,i]
                        mydmin = np.min(myd)
                        mydmax = np.max(myd)
                        if mydmin < mymin:
                            mymin = mydmin
                        if mydmax > mymax:
                            mymax = mydmax

                print("{0}: {1} -> {2}".format(k,mymin,mymax))



    """
        change the working directory
    """
    def set_directory(self, path):
        self.path = path
        print("Now reading from directory: {0}".format(self.path))

    """
        set the filename of the hdf5 file
    """
    def set_filename(self, filename):
        self.h5fn = filename

    """
        return a tree structure of the particle coordinates
        uses scipy.spatial module
    """
    def particle_tree(self, x):
        tree = scipy.spatial.cKDTree(x)
        return tree


    """
        kernel function
        the standard b-spline kernel from Monaghan 1984,5
    """
    def bspline_kernel(self, x0, x1, sml, dim):
        r = np.linalg.norm(x0-x1)
        rr = r*r
        if dim == 1:
            f1 = 4./(3.*sml)
        elif dim == 2:
            f1 = 40./(7*np.pi*sml**2)
        elif dim == 3:
            f1 = 8/np.pi * 1./sml**3

        if r > sml:
            w = 0
        else:
            roh = r/sml
            if r < sml/2.:
                w = f1 * ( 6*roh*roh*roh - 6*roh*roh + 1)
            else:
                w = f1*2*(1-roh)*(1-roh)*(1-roh)
        return w

    """
        create grid
    """
    def create_grid(self, data, dx, dims = (-1, 1, -1, 1)):
        dim = int(len(dims)/2)
        print("Creating {0}-dimensional grid".format(dim))
        #sml = np.min(data['sml'])
        sml = dx
        minx = np.min(dims)
        maxx = np.max(dims)
        print(minx, maxx)
        print(dx)
        N = int(np.ceil((maxx - minx) / sml))
        x = np.linspace(minx, maxx, N, endpoint=True)
        return dim, x



    """
        map the values to a grid using the smoothing length of the particles
        input is: tree of the mesh grid, the grid (simple 1D array), data dict containing the
        sph particles data
    """
    def map_particles_to_grid_variable_sml_tree(self, tree, mesh, dim, data, values=('rho',)):
        mapped_data = {}
        # loop over all particles
        N = len(mesh)

        for value in values:
            print("Allocating space for {0} {1} values.".format(N, value))
            mapped_data[value] = np.zeros(N)

        if dim == 1:
            print("code me")
            sys.exit(1)

        elif dim == 2:
            for p, mass in enumerate(data['m']):
                #print(p/len(data['m']), end='\r')
                radius = data['sml'][p]
                interaction_indices = tree.query_ball_point(data['x'][p], radius)
                if len(interaction_indices) > 1:
                    for interaction in interaction_indices:
                        w = self.bspline_kernel(mesh[interaction], data['x'][p], radius, dim)
                        for value in values:
                            mapped_data[value][interaction] += data['m'][p]/data['rho'][p] * data[value][p] * w
                #else:
                    #print("No interaction for particle ", data['x'][p])

        elif dim == 3:
            print("code me")
            sys.exit(1)


        return mapped_data


    """
    map the values to a grid using a variable smoothing length. loop over all particles  to find the sml and with the cell size you +
    determine the area of interaction of each particle. then loop over all cells located in this interacting region and then add the particle contribution

    """
    def map_particles_like_boss(self, data, grid, dim, dx, xmin, xmax, ymin, ymax, zmin, zmax, values=('rho',)):
    #dim==2 as a first shot
    # create the grid data points
        mapped_data = {}
        #gridlistx = {}
        #gridlisty = {}
        N = len(grid)
        size = N ** dim

    # allocate the mapped_data arrays
        for value in values:
            print("Allocating space for {0} {1} values.".format(size, value))
            mapped_data[value] = np.zeros(size)


        #cell_size = dx #defining the grid cell size
        x = data['x'][...]
        sml = data['sml'][...]
        for i, pos in enumerate(x):
            if any([ x < xmin for x in pos]) or any([x > xmax for x in pos]):
                continue
            print(pos[1])
            #print(x[i][1])
            a = int(np.floor((pos[0]-xmin) / dx))

            b = int(np.floor((pos[1]-ymin) / dx))

            kx = int(np.ceil(sml[i] / dx) + 1)

            print("Cell locations is ", grid[a], grid[b])
            print("Particle location is ", pos)
            print("Particle smoothing length is ", sml[i])
            print("Cell interacting length is ", kx)

            #sys.exit(1)


            #gridlistx = np.arange(grid[m] - kx*dx, grid[m] + kx*dx, dx) #ARANGE ME
            gridlistx = np.arange(a-kx,a+kx)
            #gridlisty = np.arange(grid[n] - kx*dx, grid[n] + kx*dx, dx)
            gridlisty = np.arange(b-kx,b+kx)
            print("The numbers of cells feeling the particles from m and n is ", gridlistx)

            print("Cells that feels the particles: ", gridlistx)
            print("Cells that feels the particles: ", gridlisty)
            for j in gridlistx:
                if j < 0 or j >= N:
                    continue
                for k in gridlisty:
                    if k < 0 or k >= N:
                        continue
                    #gridpoint = np.zeros(max(len(gridlistx), len(gridlisty)))
                    print("The integers of the grid cells are ", j, "and ",k)
                    print("And the grid length N is ", N)
                    gridpoint = np.array([grid[j], grid[k]])
                    print("The grid point coordinates are: ", gridpoint)
                    #real_distance = np.sqrt((grid[j]-x[i][0])**2 + (grid[k]-x[i][1])**2)
                    #print("The real distance between the particle and this gridpoint is ", real_distance)
                    w = self.bspline_kernel(gridpoint, x[i], sml[i], dim)
                    #print("Kernel value w= ", w)
                    for value in values:
                        mapped_data[value][j*N+k] += data['m'][i] / data['rho'][i] * data[value][i] * w
                    #print("Density mapped value = ", mapped_data[value])

        return mapped_data



    """
        map the values to a grid using the helping tree
        input is: tree containing sph particles, the grid (simple 1D array), data dict containing the
        sph particles data
    """
    def map_particles_to_grid(self, tree, grid, dim, data, dx, values=('rho',)):
        mapped_data = {}
        # loop over all gridpoints
        radius = dx
        N = len(grid)

        if dim == 1:
            size = N
        elif dim == 2:
            size = N**2
        elif dim == 3:
            size = N**3

        # allocate the mapped_data arrays
        for value in values:
            print("Allocating space for {0} {1} values.".format(size, value))
            mapped_data[value] = np.zeros(size)

        n = 0
        if dim == 3:
            for k in grid:
                for j in grid:
                    for i in grid:
                        gridpoint = np.array([i,j,k])
                        # determine interacting sph particles for this grid point
                        interaction_indices = tree.query_ball_point(gridpoint, radius)
                        if len(interaction_indices) < 1:
                            # grid point has no interaction
                            for value in values:
                                mapped_data[value][n] = 0
                        #
                        else:
                            for p in interaction_indices:
                                w = self.bspline_kernel(gridpoint, data['x'][p], radius, dim)
                                for value in values:
                                    mapped_data[value][n] += data['m'][p]/data['rho'][p] * data[value][p] * w
                        n += 1
                        if n%10000 == 0:
                            print("Processing {0}".format(n), end="\r")
                            sys.stdout.flush()
        elif dim == 2:
            for j in grid:
                for i in grid:
                    gridpoint = np.array([i,j])
                    interaction_indices = tree.query_ball_point(gridpoint, radius)
                    if len(interaction_indices) < 1:
                    # grid point has no interaction
                        for value in values:
                            mapped_data[value][n] = 0
                    #
                    else:
                       # calculate the sph sums for the values
                        for p in interaction_indices:
                            w = self.bspline_kernel(gridpoint, data['x'][p], radius, dim)
                            for value in values:
                                mapped_data[value][n] += data['m'][p]/data['rho'][p] * data[value][p] * w

                    n += 1
        elif dim == 1:
            for i in grid:
                gridpoint = np.array([i])
                interaction_indices = tree.query_ball_point(gridpoint, radius)
                if len(interaction_indices) < 1:
                # grid point has no interaction
                    for value in values:
                        mapped_data[value][n] = 0
                #
                else:
                   # calculate the sph sums for the values
                    for p in interaction_indices:
                        w = self.bspline_kernel(gridpoint, data['x'][p], radius, dim)
                        for value in values:
                            mapped_data[value][n] += data['m'][p]/data['rho'][p] * data[value][p] * w
                n += 1

        return mapped_data


# In[7]:


## Generate a 3d plot using scatter
sph = sph_particles(filename="impact.0005.h5")
xyz135 = sph.read_data()
print(sph.read_data(keys = ['x', 'rho']))

x135 = xyz135["x"].T[0]
y135 = xyz135["x"].T[1]
z135 = xyz135["x"].T[2]

# plt.show()
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
 
my_cmap = plt.get_cmap('coolwarm')
 
# Creating plot
sctt = ax.scatter3D(x135, y135, z135,
                    alpha = 0.8,
                    c = z135,
                    cmap = my_cmap, 
                    marker ='o')
 
plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold')
ax.set_ylabel('Y-axis', fontweight ='bold')
ax.set_zlabel('Z-axis', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
plt.show()

## Generate a 3d plot using Plotly
sph = sph_particles(filename="impact.0005.h5") #call file
xyz135 = sph.read_data(keys = ['x', 'rho']) #read keys
rho135 = xyz135["rho"]
print(sph.read_data(keys = ['x', 'rho']))


df = pd.DataFrame(xyz135["x"], columns=['x', 'y', 'z'])
df['rho'] = xyz135["rho"]
fig = px.scatter_3d(df, x='x', y='y', z ='z', color = 'rho')
fig.update_traces(marker=dict(size=2,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()


# In[11]:


# Set up graphing of multiple timesteps
import pandas as pd
# Below code can be used to suplement different file paths
## path="impact_30deg/"
## filename=path+"filename" #try to loop this for various degrees

# Impact_0005
sph005 = sph_particles(filename= "impact.0005.h5")
xyz005 = sph005.read_data(keys = ['x', 'rho'])
rho005 = xyz005["rho"]
df005 = pd.DataFrame(xyz005["x"], columns=['x', 'y', 'z'])
df005['rho'] = xyz005["rho"]
dfnew005 = df005[(df005['y']>0) & (df005['y']<1)]

# Impact_0015
sph15 = sph_particles(filename="impact.0015.h5")
xyz15 = sph15.read_data(keys = ['x', 'rho'])
rho15 = xyz15["rho"]
df15 = pd.DataFrame(xyz15["x"], columns=['x', 'y', 'z'])
df15['rho'] = xyz15["rho"]
dfnew15 = df15[(df15['y']>0) & (df15['y']<1)]

# Impact_0050
sph50 = sph_particles(filename="impact.0050.h5")
xyz50 = sph50.read_data(keys = ['x', 'rho'])
rho50 = xyz50["rho"]
df50 = pd.DataFrame(xyz50["x"], columns=['x', 'y', 'z'])
df50['rho'] = xyz50["rho"]
dfnew50 = df50[(df50['y']>0) & (df50['y']<1)]

# Impact_0100 
sph100 = sph_particles(filename="impact.0100.h5")
xyz100 = sph100.read_data(keys = ['x', 'rho'])
rho100 = xyz100["rho"]
#print(sph.read_data(keys = ['x', 'rho']))
df100 = pd.DataFrame(xyz100["x"], columns=['x', 'y', 'z'])
df100['rho'] = xyz100["rho"]
dfnew100 = df100[(df100['y']>0) & (df100['y']<1)]

# Impact_0150
sph150 = sph_particles(filename="impact.0150.h5")
xyz150 = sph150.read_data(keys = ['x', 'rho'])
rho150 = xyz150["rho"]
#print(sph.read_data(keys = ['x', 'rho']))
df150 = pd.DataFrame(xyz150["x"], columns=['x', 'y', 'z'])
df150['rho'] = xyz150["rho"]
dfnew150 = df150[(df150['y']>0) & (df150['y']<1)]

# Impact_0200
sph200 = sph_particles(filename="impact.0200.h5")
xyz200 = sph200.read_data(keys = ['x', 'rho'])
rho200 = xyz200["rho"]
#print(sph.read_data(keys = ['x', 'rho']))
df200 = pd.DataFrame(xyz200["x"], columns=['x', 'y', 'z'])
df200['rho'] = xyz200["rho"]
dfnew200 = df200[(df200['y']>0) & (df200['y']<1)]


fig = plt.figure(figsize=(8, 6), tight_layout=True)
ax1 = fig.add_subplot(321)
ax1.scatter(dfnew005["x"], dfnew005["z"], c = dfnew005['rho']/dfnew005['rho'].max())
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.set_xlim(-7,7)
ax1.set_ylim(-2,5)
ax1.set_title('Frame 0005, x-z view')

ax2 = fig.add_subplot(322)
ax2.scatter(dfnew15["x"], dfnew15["z"], c = dfnew15['rho']/dfnew15['rho'].max())
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_xlim(-7,7)
ax2.set_ylim(-2,5)
ax2.set_title('Frame 0015, x-z view')

ax3 = fig.add_subplot(323)
ax3.scatter(dfnew50["x"], dfnew50["z"], c = dfnew50['rho']/dfnew50['rho'].max())
ax3.set_xlabel('x')
ax3.set_ylabel('z')
ax3.set_xlim(-7,7)
ax3.set_ylim(-2,5)
ax3.set_title('Frame 0050, x-z view')

ax4 = fig.add_subplot(324)
ax4.scatter(dfnew100["x"], dfnew100["z"], c = dfnew100['rho']/dfnew100['rho'].max())
ax4.set_xlabel('x')
ax4.set_ylabel('z')
ax4.set_xlim(-7,7)
ax4.set_ylim(-2,5)
ax4.set_title('Frame 0100, x-z view')

ax5 = fig.add_subplot(325)
ax5.scatter(dfnew150["x"], dfnew150["z"], c = dfnew150['rho']/dfnew150['rho'].max())
ax5.set_xlabel('x')
ax5.set_ylabel('z')
ax5.set_xlim(-7,7)
ax5.set_ylim(-2,5)
ax5.set_title('Frame 0150, x-z view')

ax6 = fig.add_subplot(326)
im = ax6.scatter(dfnew200["x"], dfnew200["z"], c = dfnew200['rho']/dfnew200['rho'].max())
ax6.set_xlabel('x')
ax6.set_ylabel('z')
ax6.set_xlim(-7,7)
ax6.set_ylim(-2,5)
ax6.set_title('Frame 0200, x-z view')
fig.colorbar(im, ax=ax6)


# In[ ]:




