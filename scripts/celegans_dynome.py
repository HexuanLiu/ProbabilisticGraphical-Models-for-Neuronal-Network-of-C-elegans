# coding: utf-8

import time
import os
from functools import reduce

import numpy as np
import scipy.io as sio
from scipy import integrate, signal, sparse, linalg, interpolate

import sys_paths as path
import celegans_utilities as c_util

##################################
### PARAMETERS / CONFIGURATION ###
##################################

""" Number of Neurons """
N = 279

""" Cell membrane conductance (pS) """
Gc = 0.1

""" Cell Membrane Capacitance"""
C = 0.015

""" Gap Junctions (Electrical, 279*279) """
os.chdir(path.connectome_data_dir)

ggap = 1.0
Gg_Static = np.load('Gg.npy')
#Gg_Static = np.load('Gg.npy')

""" Synaptic connections (Chemical, 279*279) """
gsyn = 1.0
Gs_Static = np.load('Gs.npy')
#Gs_Static = np.load('Gs.npy')

""" Leakage potential (mV) """
Ec = -35.0

""" Directionality (279*1) """
E = np.load('emask.npy')
E = -48.0 * E
EMat = np.tile(np.reshape(E, N), (N, 1))

os.chdir(path.default_dir)

""" Synaptic Activity Parameters """
ar = 1.0/1.5 # Synaptic activity's rise time
ad = 5.0/1.5 # Synaptic activity's decay time
B = 0.125 # Width of the sigmoid (mv^-1)

rate = 0.025
offset = 0.12

Gg_Dynamic = Gg_Static.copy()
Gs_Dynamic = Gs_Static.copy()

mask_Healthy = np.ones(N, dtype = 'bool')

neurons = c_util.load_Json('neurons.json')
neurons_list = neurons['neurons']

neuron_names = []

for neuron in neurons_list:

    neuron_names.append(neuron['name'])

""" motor neurons """

VB_ind = np.asarray([150, 138, 170, 179, 186, 193, 202, 212, 217, 229, 234])
DB_ind = np.asarray([164, 152, 172, 188, 203, 218, 235])
VA_ind = np.asarray([160, 169, 177, 185, 191, 201, 211, 216, 228, 233, 239, 244])
DA_ind = np.asarray([167, 173, 184, 194, 207, 224, 237, 247, 248])
VD_ind = np.asarray([166, 168, 174, 183, 190, 200, 205, 215, 221, 232, 238, 241, 250])
DD_ind = np.asarray([163, 181, 195, 214, 231, 245])
AVB_ind = np.asarray([96, 105])
PVC_ind = np.asarray([261, 267])
AVA_ind = np.asarray([47, 55])
AVD_ind = np.asarray([116, 118])
AVE_ind = np.asarray([58, 66])

B_grp = np.union1d(DB_ind, VB_ind)
D_grp = np.union1d(VD_ind, DD_ind)
A_grp = np.union1d(VA_ind, DA_ind)
AVB_B_grp = np.union1d(B_grp, AVB_ind)
AVB_PVC_B_grp = np.union1d(AVB_B_grp, PVC_ind)
FWD_group = np.union1d(AVB_PVC_B_grp, D_grp)

AVA_AVD_grp = np.union1d(AVA_ind, AVD_ind)
AVA_AVD_AVE_grp = np.union1d(AVA_AVD_grp, AVE_ind)
AD_grp = np.union1d(A_grp, D_grp)
BWD_group = np.union1d(AVA_AVD_AVE_grp, AD_grp)

BD_grp = np.union1d(B_grp, D_grp)
BDA_grp = np.union1d(BD_grp, A_grp)

""" Boolean masks for motor neurons """

B_grp_bool = np.zeros(N, dtype = 'bool')
B_grp_bool[B_grp] = True

AVB_B_grp_bool = np.zeros(N, dtype = 'bool')
AVB_B_grp_bool[AVB_B_grp] = True

BD_grp_bool = np.zeros(N, dtype = 'bool')
BD_grp_bool[BD_grp] = True

BDA_grp_bool = np.zeros(N, dtype = 'bool')
BDA_grp_bool[BDA_grp] = True

sensory_list = []
inter_list = []
motor_list = []

for neuron in neurons_list:
    
    if neuron['group'] == 'sensory':
        
        sensory_list.append(neuron['index'])

    if neuron['group'] == 'inter':
        
        inter_list.append(neuron['index'])

    if neuron['group'] == 'motor':
        
        motor_list.append(neuron['index'])

sensory_group = np.asarray(sensory_list)
inter_group = np.asarray(inter_list)        
motor_group = np.asarray(motor_list)

motor_grp_bool = np.zeros(N, dtype = 'bool')
motor_grp_bool[motor_group] = True

##################################
### FUNCTIONS ####################
##################################

def update_Mask(old, new, t, tSwitch):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) + np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))

def EffVth(Gg, Gs):

    Gcmat = np.multiply(Gc, np.eye(N))
    EcVec = np.multiply(Ec, np.ones((N, 1)))

    M1 = -Gcmat
    b1 = np.multiply(Gc, EcVec)

    Ggap = np.multiply(ggap, Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, N, N).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(gsyn, Gs)
    s_eq = round((ar/(ar + 2 * ad)), 4)
    sjmat = np.multiply(s_eq, np.ones((N, N)))
    S_eq = np.multiply(s_eq, np.ones((N, 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, N, N).toarray()

    b3 = np.dot(Gs_ij, np.multiply(s_eq, E))

    M = M1 + M2 + M3

    global LL, UU, bb

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, N)

def EffVth_rhs(iext, inmask):

    InputMask = np.multiply(iext, inmask)
    b = np.subtract(bb, InputMask)

    vth = linalg.solve_triangular(UU, linalg.solve_triangular(LL, b, lower = True, check_finite=False), check_finite=False)

    return vth

def modify_Connectome(ablation_Mask):

    global Gg_Dynamic, Gs_Dynamic

    apply_Col = np.tile(ablation_Mask, (N, 1))
    apply_Row = np.transpose(apply_Col)

    apply_Mat = np.multiply(apply_Col, apply_Row)

    Gg_Dynamic = np.multiply(Gg_Static, apply_Mat)
    Gs_Dynamic = np.multiply(Gs_Static, apply_Mat)

    EffVth(Gg_Dynamic, Gs_Dynamic)

def modify_edges(neurons_from, neurons_to, conn_type):

    global Gg_Dynamic, Gs_Dynamic

    apply_Mat = np.ones((N,N), dtype = 'bool')
    apply_Mat_Identity = np.ones((N,N), dtype = 'bool')

    for k in xrange(len(neurons_from)):

        neuron_from_ind = []
        neurons_target_inds = []

        neuron_from = neurons_from[k]
        neurons_target = neurons_to[k]

        neuron_from_ind.append(neuron_names.index(neuron_from))

        for neuron_target in neurons_target:

            neurons_target_inds.append(neuron_names.index(neuron_target))

        if conn_type == 'syn':

            apply_Mat[neurons_target_inds, neuron_from_ind] = 0

        elif conn_type == 'gap':

            apply_Mat[neurons_target_inds, neuron_from_ind] = 0
            apply_Mat[neuron_from_ind, neurons_target_inds] = 0

    if conn_type == 'syn':

        Gg_Dynamic = np.multiply(Gg_Static, apply_Mat_Identity)
        Gs_Dynamic = np.multiply(Gs_Static, apply_Mat)

    elif conn_type == 'gap':

        Gg_Dynamic = np.multiply(Gg_Static, apply_Mat)
        Gs_Dynamic = np.multiply(Gs_Static, apply_Mat_Identity)

    EffVth(Gg_Dynamic, Gs_Dynamic)

def jimin_rhs_constinput(t, y):

    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(Gc, (Vvec - Ec))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (N, 1))
    GapCon = np.multiply(Gg_Dynamic, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), EMat)
    SynapCon = np.multiply(np.multiply(Gs_Dynamic, np.tile(SVec, (N, 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(ar, (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(ad, SVec)

    """ Input Mask """
    Input = np.multiply(iext, inmask)

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/C
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def run_network(t_start, t_final, t_delta, input_mask, initcond = False, ablation_mask = mask_Healthy):

    t0 = t_start
    tf = t_final
    dt = t_delta

    global nsteps, inmask

    nsteps = int(np.floor((tf - t0)/dt) + 1)
    inmask = input_mask

    """ define the connectivity """

    modify_Connectome(ablation_mask)

    """ Input signal magnitude """
    global iext

    iext = 100000.

    """ Calculate V_threshold """
    global vth

    vth = EffVth_rhs(iext, inmask)

    if type(initcond) != 'numpy.ndarray':

        initcond = 10**(-4)*np.random.normal(0, 0.94, 2*N)

    else:

        initcond = initcond

    """ Configuring the ODE Solver """
    r = integrate.ode(jimin_rhs_constinput).set_integrator('vode', atol = 1e-3, min_step = dt*1e-6, method = 'bdf', with_jacobian = True)
    r.set_initial_value(initcond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, N))

    t[0] = t0
    traj[0, :] = initcond[:N]
    vthmat = np.tile(vth, (nsteps, 1))

    """ Integrate the ODE(s) across each delta_t timestep """
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:N]

        k += 1

    result_dict = {
            "t": t,
            "steps": nsteps,
            "trajectory_mat": traj,
            "v_threshold": vthmat,
            "v_solution" : np.subtract(traj, vthmat)
            }

    return result_dict

