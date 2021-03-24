import os
import shutil
import math
import numpy as np
import re

# This script file will create a new simulation folder, copy the necessary files into the folder,
# compile the specified cuda file, and run the simulation. The simulation and all its files will be contained
# within the simulation folder.

# ===== Start: input ======
cudaFile = 'bacteria_surface_geometry_UUU_semicircle.cu'

ell = 3.5                   # half-length of bacteria, used to non-dimensionalize [micro m]

sim_num = 1                 # simulation number
case_num = 1                # case number

dt = 0.0001                 # non-dimensional time step
time_save = 0.6             # non-dimensional time at which to output
start_time = 0		        # non-dimensional start time of simulation
final_time = 2000.4		    # non-dimensional final time of simulation

rho = 0.1                   # non-dimensional bacteria density

l = (3.5 / ell)             # non-dimensional half-length of bacteria
d = (0.5 / ell)             # non-dimensional half-diameter of bacteria

C = (1000 / ell)            # non-dimensional wall surface displacement from origin
L = 6                       # multiple of lambda or 2*R used to calculate non-dimensional wall length

epsilon_r = 300.0           # non-dimensional strengh of bacteria-wall interaction
sigma_bdy = d			    # non-dimensional range parameter for bacteria-wall steric repulsion

inverse_Pe_T = 0.0014       # inverse translational Peclet number
inverse_Pe_parallel = 0.0   # inverse Peclet parallel number
inverse_Pe_perp = 0.0       # inverse Peclet perpendicular number
inverse_Pe_R = 0.01         # inverse rotational Peclet number

delta_run = 5.71429			# non-dimensional run time
delta_tumble = 0.0          # non-dimensional tumble time
kappa = 10                  # concentration parameter for von Mises distribution
vMF_n = 10000               # total example numbers drawn from von Mises distribution

# flat surface parameters:
W_hat_x = 0                 # wall normal x-coordinate for bottom surface
W_hat_y = 1                 # wall normal y-coordinate for bottom surface
W_hat_z = 0                 # wall normal z-coordinate for bottom surface

# sine surface parameters:
A = 1.0 * (7.0 / ell)
lambda_ = 2.9485714 * (7.0 / ell)

# semicircle surface parameters:
R = 12 * (1.0 / ell)

# ===== End: input ======

# file paths: ======================================
mainfolder = os.path.dirname(os.path.abspath(__file__))

cudaFilePath = os.path.join(mainfolder, cudaFile)

kappaFile = 'vonMisesFisher2D_kappa{}_n{}.txt'.format(kappa, vMF_n)
kappaFilePath = os.path.join(mainfolder, kappaFile)

# find surface and run and tumble flag from cudaFile name ======================================
simulationType = re.search('bacteria_surface_geometry_(.*).cu', cudaFile).group(1)

if simulationType.find('run_and_tumble') == -1: # 'run_and_tumble' not found
    RTFlag = False
    surfaceType = simulationType
    simType = 'BD'
else:
    RTFlag = True
    surfaceType = re.search('(.*)_run_and_tumble', simulationType).group(1)
    simType = 'RT'

# make directory ======================================
sim_folder = '%s_%s_sim_%04d_case_%04d' % (surfaceType, simType, sim_num, case_num)
print(sim_folder)
simFolderPath = os.path.join(mainfolder, sim_folder)
if not os.path.isdir(simFolderPath):
    os.makedirs(simFolderPath)

# copy files into new folder ======================================
shutil.copy2(cudaFilePath, simFolderPath)
shutil.copy2(kappaFilePath, simFolderPath)

# change working directory to new folder ======================================
os.chdir(simFolderPath)

# compile cuda file ======================================
cuda_exec_name = 'bacteria_surface'
cuda_global_exec = os.path.join(simFolderPath, cuda_exec_name)

if not os.path.exists(cuda_global_exec):
    cuda_global_script = os.path.join(simFolderPath, cudaFile)
    os.system('nvcc -std=c++11 --output-file {} -lcurand {}'.format(
        cuda_global_exec, cuda_global_script))
    os.chmod(cuda_global_exec, 0o744) #change file permissions

# calculate N ======================================
if surfaceType == 'flat':
    area = 2 * C * L
elif surfaceType == 'sine':
    area = 2 * C * L * lambda_
elif surfaceType == 'UUU_semicircle':
    area = (2 * C) * (L * 2.0 * R) + L * math.pi * R * R

N = np.ceil(rho * area)

# create input file ======================================
inputs_begin = [sim_num,
    case_num,
    dt,
    time_save,
    start_time,
    final_time,
    N,
    l,
    d]

if surfaceType == 'flat':
    L = 50 # non-dimensional wall length
    inputs_surf = [W_hat_x,
        W_hat_y,
        W_hat_z,
        C,
        L]
elif surfaceType == 'sine':
    inputs_surf = [C,
        L,
        A,
        lambda_]
elif surfaceType == 'UUU_semicircle':
    inputs_surf = [C,
        L,
        R]

if RTFlag: # run and tumble simulations
    inputs_end = [epsilon_r,
        sigma_bdy,
        inverse_Pe_T,
        inverse_Pe_parallel,
        inverse_Pe_perp,
        inverse_Pe_R,
        delta_run,
        delta_tumble,
        kappa,
        vMF_n]
else: # Brownian motion simulations
    delta_run = final_time + 100.0
    avg_n_tumble = 	0	    # average tumbling angle in degrees (NOT used if delta_run > final_time)
    std_n_tumble = 0		# std tumbling angle in degrees (NOT used if delta_run > final_time)

    inputs_end = [epsilon_r,
        sigma_bdy,
        inverse_Pe_T,
        inverse_Pe_parallel,
        inverse_Pe_perp,
        inverse_Pe_R,
        delta_run,
        delta_tumble,
        avg_n_tumble,
        std_n_tumble]

inputs = inputs_begin + inputs_surf + inputs_end
input_str = '\n'.join(str(e) for e in inputs)
with open('bacteria_surface_input.txt', 'w') as f:
    f.write(input_str)

# run simulation ======================================
os.system('./' + cuda_exec_name)
