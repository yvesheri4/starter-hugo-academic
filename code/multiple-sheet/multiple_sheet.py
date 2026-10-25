"""Archived drift-space Multiple-sheet numerical core by Yves Heri. See README."""
import numpy as np
from scipy import constants as cst

import numpy as np
from scipy import constants as cst

# Define the particle class
class Particle:
    def __init__(self, mass,charge, index):
        self.mass = mass
        self.i = index
        self.charge = charge
        self.r = np.zeros((1), dtype='float64')
        self.v = np.zeros((1), dtype='float64')

    def initPos(self, x):
        self.r = x

    def initSpeed(self, vx):
        self.v = vx

    def Heri_push(self, dt, E):
        temp = self.v
        self.v += 0.75*E*dt
        self.r += 1.5*(self.v+temp)*dt/2

def Create_Particle_list(mass,charge):
    List = []
    for i in range(len(charge)):
        List.append(Particle(mass[i],charge[i],i+1))
    return List

def Init_Particle_list(List,InitPos,InitSpeed):
    for i in range(len(List)):
        List[i].initPos(InitPos[i])
        List[i].initSpeed(InitSpeed[i])
    return List

def Electric_Field(List,V0,d,t,Dt):
    N_inj = 1+int(t/Dt)
    if N_inj > len(List):
        N_inj = len(List)
    E = np.zeros(len(List))
    sum1 = 0
    List2 = []
    for j in range(N_inj):
        if List[j].r < d and List[j].r >=0:
            sum1 += (List[j].charge*List[j].r)
            List2.append(List[j])
    for j in range(len(List2)):
        sum2 = 0
        for i in range(j):
            sum2 += List2[i].charge
        E[List2[j].i-1] = 4*(sum1 - sum2 - 0.5*List2[j].charge)
    return E

def Current_density(Vel,rho):

    len_t = Vel.shape[0]
    N_sheet = Vel.shape[1]
    J = np.zeros(len_t)

    for i in range(len_t):
        sum1 = 0
        for j in range(N_sheet):
            sum1 += 1.5*Vel[i][j]*rho[j]
        J[i] = sum1

    return J

def Simulation(t_sim,Sys_Params,InitCond,k=100):

    time = np.linspace(0,t_sim,k)
    dt = time[1]-time[0]

    InitPos,InitSpeed = InitCond
    Dt,mass,charge,V0,d = Sys_Params

    List_part = Create_Particle_list(mass,charge)

    List_part = Init_Particle_list(List_part,InitPos,InitSpeed)
#     print(List_part[0].r)

    position = np.zeros([len(time),len(List_part)])
    velocity = np.zeros([len(time),len(List_part)])
    E_field = np.zeros([len(time),len(List_part)])

    count = 0

    for t in time:
        E = Electric_Field(List_part,V0,d,t,Dt)
        E_field[count] = E

        N_inj = 1+int(t/Dt)
        if N_inj > len(List_part):
            N_inj = len(List_part)

        # N_inj = len(List_part)

        for j in range(N_inj):
            position[count][j] = List_part[j].r
            velocity[count][j] = List_part[j].v

            if List_part[j].r < d and List_part[j].r >= 0:
                List_part[j].Heri_push(dt,E[j])
            else:
                List_part[j].v = 0
        count += 1
    return position,velocity,E_field

def distorsion(position,time,kDt,M,q):
    t_init = kDt
    i = 0
    while  position[i,0] < q:
        i +=1
    j = 0
    while  position[j,-1] < q:
        j +=1
    t_final = time[j] - time[i]
    Distorsion = t_final / t_init
    return Distorsion

def Norm_J_crit_s(X):
    t1 = (2/X**3)
    t2 = (3/4)*X**2
    return t1*(1-np.sqrt(1-t2))

def Norm_J_crit_e(X):
    t1 = (16/(27*X**3))
    t2 = (27/16)*X**2
    return t1*(1-np.sqrt(1-t2))

def limit_charge(T0,Params_sys,ini_conds,iterations):
    tol = 1e-3
    Indicator = False
    position,velocity,E_field = Simulation(T0,Params_sys,ini_conds,iterations)
    if (velocity[:,:] < 0).any():
        Indicator = True
    return Indicator

# def sheets_spacing(mass,) = lambda e,E,m,dt: 0.5*(e*E/m)*(dt**2)

Time_transit = lambda Vg,d : np.sqrt(2*d/(cst.e*Vg/(d*cst.m_e)))

Sigma_1 = lambda Vg,d : -cst.epsilon_0*Vg/d

def gaussian_profile(tp,M):
  sig2 = tp **2/np.log(256)
  sig = np.sqrt(sig2)
  t = np.linspace(-2.5*sig, 2.5*sig, M)
  prof_ = np.exp(-0.5*((t)**2) /sig2 )
  prof_ /= np.sum(prof_)
  return prof_

def trapezoid_profile(tp,M,a=.15):
  start_time = 0
  rise_time = a*tp
  fall_time = a*tp
  tpl = (1-a)*tp

  # End time of the trapezoidal pulse
  end_time = start_time + rise_time + tpl + fall_time

  t = np.linspace(0, end_time, M)

  # Initialize the pulse array
  pulse = np.zeros_like(t)

  # Calculate indices for rise, top, and fall segments
  rise_end = start_time + rise_time
  top_end = rise_end + tpl
  fall_start = top_end
  fall_end = fall_start + fall_time

  # Rise segment
  pulse[(t >= start_time) & (t < rise_end)] = (
       (t[(t >= start_time) & (t < rise_end)] - start_time) / rise_time
  )
  # Top segment (FWHM width)
  pulse[(t >= rise_end) & (t < top_end)] = 1

  # Fall segment
  pulse[(t >= fall_start) & (t < fall_end)] = (
       (1 - ((t[(t >= fall_start) & (t < fall_end)] - fall_start) / fall_time))
  )
  return pulse / np.sum(pulse)

Time_DCL = lambda Vg,d : 3*d/np.sqrt(8*cst.e*Vg/cst.m_e)

def pulse_profile(tp,M,a=0.15,Type = 'square'):
    Type = Type.lower()
    if Type == 'square':
        return np.ones(M)/M
    if Type == 'gauss':
        return gaussian_profile(tp,M)
    if Type == 'trapezoidal':
        return trapezoid_profile(tp,M,a)
    else:
        return print('Profile not recognized')
