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

    # set max r distance and amount of bins to use for calculation
    # r_cut = 10
    # bins = 200
    # cutoff_radii = np.linspace(0, r_cut, bins)

    # compute the g(r)
    hex_order = freud.order.Hexatic()
    hex_order.compute(system)
    # g_r = freud.density.RDF(bins, r_cut)
    # g_r.compute(system)
    print(np.absolute(hex_order.particle_order))
    
    # plot g(r)
    # plt.plot(cutoff_radii, g_r.rdf)
    global_hex_order = np.average(np.absolute(hex_order.particle_order))
    std = np.std(np.absolute(hex_order.particle_order))
    torque = float(f'{filename[-22]}.{filename[-21]}')
    # plt.plot(torque, np.average(np.absolute(hex_order.particle_order)), 'o')
    
    
    return torque, global_hex_order, std


if __name__ == "__main__":

    # params = {'axes.labelsize': 18,'axes.titlesize':14, 'legend.fontsize': 15, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
    params = {'axes.labelsize': 14}
    plt.rcParams.update(params)

    # get the pair correlation plots and show them together
    # hexatic_ord('/Users/seb/Desktop/CLS Uni/Master Thesis/sim_examples.tmp/T005_F36_phi04_N40000k.gsd', 0.9)
    for gamma in ['00', '12', '24', '36', '96']:
        torque_list = []
        hex_ord_list = []
        std_list = []
        for peclet in ['2', '6', '12','24','36', '48', '60']:
            try:
                torque, hex_ord, std = hexatic_ord(f'/Volumes/T7 Touch/Data/T0{gamma}_F{peclet}_phi04_N250k.gsd', 0.9)
                torque_list.append(torque)
                hex_ord_list.append(hex_ord)
                std_list.append(std)
            except:
                pass
        
        plt.plot([20, 60, 120, 240, 360, 480, 600], hex_ord_list, 'o--', label=r'$\omega_0^*$ = '+f'{gamma[0]}.{gamma[1]}')
        plt.fill_between([20, 60, 120, 240, 360, 480, 600], np.array(hex_ord_list)+0.5*np.array(std_list), np.array(hex_ord_list)-0.5*np.array(std_list), alpha=0.4)
    # hexatic_ord('/Volumes/T7 Touch/Data/T00_F12_phi04_N250k.gsd', 0.9)
    # hexatic_ord('/Users/seb/Desktop/CLS Uni/Master Thesis/sim_examples.tmp/T005_F36_phi08_N40000k.gsd', 0.9)
    # pair_corr('/Volumes/T7 Touch/T005_F36_phi04_N250k.gsd', 0.9)
    # pair_corr('/Volumes/T7 Touch/T072_F36_phi04_N250k.gsd', 0.9)
    plt.title(r'Global hexatic order for $\Phi$ = 0.4')
    plt.xticks((20, 60, 120, 240, 360, 480, 600), [20, 60, 120, 240, 360, 480, 600])
    # plt.yticks((0.7, 0.8, 0.9, 1.0), [0.7, 0.8, 0.9, 1.0])
    plt.yticks((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel(r'$Pe$')
    plt.ylabel(r'$q_{6}$')
    # plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()