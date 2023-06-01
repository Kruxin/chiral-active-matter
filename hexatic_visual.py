import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
import freud
import sys

def hexatic_ord(filename, snapshot):
    traj = gsd.hoomd.open(f'{filename}', 'rb')
    timesteps = len(traj)
    curr_snapshot = int(timesteps*snapshot)
    box = freud.box.Box.from_box(traj[curr_snapshot].configuration.box)

    # set up the system for freud analysis
    system = freud.locality.NeighborQuery.from_system((box,traj[curr_snapshot].particles.position))

    # compute the g(r)
    hex_order = freud.order.Hexatic()
    hex_order.compute(system)
    # g_r = freud.density.RDF(bins, r_cut)
    # g_r.compute(system)
    # print(np.absolute(hex_order.particle_order))
    per_particle_hexord = np.absolute(hex_order.particle_order)

    # get the angles of the q6 vectors from [0, 2] pi rad
    hex_angles = np.mod(np.angle(hex_order.particle_order)/np.pi, 2)

    # get global average of q6 - in terms of complex numbers
    # glob_avg = np.average(hex_order.particle_order)
    # print(glob_avg)

    # # project all q6i's onto this global average
    # hex_ord_projected = []
    # glob_avg_vec = np.array([glob_avg.real, glob_avg.imag])/np.sqrt(glob_avg.real**2 + glob_avg.imag**2)
    # for i in range(len(per_particle_hexord)):
    #     a = np.array([hex_order.particle_order[i].real, hex_order.particle_order[i].imag])
    #     projection = np.dot(a, glob_avg_vec)
    #     # print(projection)
    #     hex_ord_projected.append(projection)

    # # get the projected angles
    # hex_angles = hex_ord_projected

    particle_positions = traj[curr_snapshot].particles.position
    # plot g(r)
    # plt.plot(cutoff_radii, g_r.rdf)
    global_hex_order = np.average(np.absolute(hex_order.particle_order))
    torque = float(f'{filename[-22]}.{filename[-21]}')
    # plt.plot(torque, np.average(np.absolute(hex_order.particle_order)), 'o')
    
    
    markersize = 0.05
    # plt.subplot(8,4,subplot_position)  
    x_list, y_list = particle_positions[:,:1], particle_positions[:,1:2]
    # plt.rcParams["figure.figsize"] = (10,10)

     # for visual, get only the particles that have < 0.95 hexatic order
    x_nohex = [0]
    y_nohex = [0]
    nohex = [0]

    for i in range(len(per_particle_hexord)):
        if per_particle_hexord[i] > 0.95:
            per_particle_hexord[i] = 1
        else:
            per_particle_hexord[i] = 0
            x_nohex.append(x_list[i])
            y_nohex.append(y_list[i])
            nohex.append(0)

    # plot particles in scatter plot with color being hex angles - shifted the angles around slightly for different color bordering
    plt.scatter(x_list, y_list, s=markersize, c=(hex_angles+0.07)%2, cmap='Dark2')
    # plt.scatter(x_list, y_list, s=markersize, c=(hex_angles+0.07)%2, cmap='hsv')
    # plt.scatter(x_list, y_list, s=markersize, c=hex_angles, cmap='afmhot')
    # plt.scatter(x_list, y_list, s=markersize, c=per_particle_hexord, cmap='Greys')
    plt.colorbar(label=r'$\theta/\pi$' ,ticks=[0.0001, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.9999]).ax.set_yticklabels(['0', 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, '2'])
    # plt.colorbar()
    # for plotting only defection lines on top of hex mosaic
    plt.scatter(x_nohex, y_nohex, s=markersize*0.1, c=nohex, cmap='Greys')
    plt.axis('off')
    plt.gca().set_aspect(aspect='equal')
    plt.tight_layout()
    
    plt.show()
    
    # return torque, global_hex_order


if __name__ == "__main__":

    # hexatic_ord('/Users/seb/Desktop/CLS Uni/Master Thesis/sim_examples.tmp/T005_F36_phi04_N40000k.gsd', 0.9)
    gamma = '96 '
    peclet = '24'
    hexatic_ord(f'/Volumes/T7 Touch/high_density/T0{gamma}_F{peclet}_phi09_N250k.gsd', 0.9)
