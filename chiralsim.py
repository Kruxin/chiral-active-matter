import itertools
import math
import gsd.hoomd
import hoomd
import numpy as np
import sys

# get the parameters from the command line arguments given
if len(sys.argv) != 8:
    sys.exit()

z_torque, F_act, density, timestep_size, total_timesteps, equilibrate_steps, steps_per_save = map(float, sys.argv[1:])

# set up particle parameters for simulation
N_particles = 250000
N = N_particles

# these parameters come from the Ni & Ma paper
epsi = F_act/24
if F_act < 24:
    epsi = 1
D_r = 0.1
sig = 1

# simulation details 
#density = 0.4
box_L = np.sqrt(N_particles*np.pi*sig**2/(density*4))
#timestep_size = 6e-5
#total_timesteps = 2.5e6
#equilibrate_steps = 2.5e6
#steps_per_save = 2000


# set up box dimensions and particle positions/spacing for initialisation
K = math.ceil(N_particles**(1/2))
L = box_L 
x = np.linspace(-L/2, L/2, K, endpoint=False)
position = [(a,b,0) for a in x for b in x]

# create a snapshot and create particles
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0]*N_particles
snapshot.configuration.box = [L, L, 0, 0, 0, 0]
snapshot.particles.types = ['A']

# set up simulation environment
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=2)
sim.create_state_from_snapshot(snapshot)

# set up the integrator (rotational degrees of freedom on, else torques aren't integrated)
# activate shifted LJ potential and add to the integrator
integrator = hoomd.md.Integrator(dt=timestep_size, integrate_rotational_dof=True)
n_list = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=n_list, mode='shift')
lj.params[('A', 'A')] = dict(epsilon=epsi, sigma=sig)
lj.r_cut[('A', 'A')] = 2**(1/6)
integrator.forces.append(lj)

# set up an active force for both directed movement and chiral motion and add to integrator
active = hoomd.md.force.Active(filter=hoomd.filter.All())
active.active_force['A'] = (F_act, 0, 0)
active.active_torque['A'] = (0,0,z_torque)

rotational_diffusion_updater = active.create_diffusion_updater(trigger=1, rotational_diffusion=D_r)

# sim.operations += rotational_diffusion_updater_eq
sim.operations += rotational_diffusion_updater
integrator.forces.append(active)

# set up brownian dynamics and add to integrator (kT set to 0 at the moment following a paper)
brownian = hoomd.md.methods.Brownian(kT=0, filter=hoomd.filter.All())
brownian.gamma.default = 1
integrator.methods.append(brownian)
sim.operations.integrator = integrator
snapshot = sim.state.get_snapshot()

# can log thermodynamic quantities if wanted below
# thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
#     filter=hoomd.filter.All())
# sim.operations.computes.append(thermodynamic_properties)

# do some timesteps to equilibrate
sim.run(equilibrate_steps)

# run and write to GSD file
gsd_writer = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(steps_per_save)), dynamic=['property', 'momentum'], \
    filename=f'eps{epsi}_T0{str(int(z_torque*100)).zfill(2)}_F{int(F_act)}_phi0{int(density*10)}_N250k.gsd', mode='wb')
sim.operations.writers.append(gsd_writer)
sim.run(total_timesteps)
