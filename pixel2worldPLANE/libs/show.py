import matplotlib.pyplot as plt
import numpy as np

def make_figure(x_lim, y_lim, z_lim):
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    plt.gca().set_aspect('equal', adjustable='box')
    return ax


def plot_coord_sys(M_world_sys, scale, sys_name, ax, alpha):
    # Define axes in sys-coord_system
    xyz_s = np.array([[0, 0, 0],[scale, 0, 0], [0, 0, 0],[0, scale, 0], [0, 0, 0],[0, 0, scale]]).T

    # prepare to plot: homogeneous coords:
    xyz1_s = np.row_stack((xyz_s,np.ones((1,xyz_s.shape[1]))))

    # convert to world coords:
    xyz1_w = M_world_sys.dot(xyz1_s)

    # plot axes: x->red, y->green, z->blue.
    ax.plot(xyz1_w[0,:2], xyz1_w[1,:2], xyz1_w[2,:2], color='red', alpha=alpha)
    ax.plot(xyz1_w[0,2:4], xyz1_w[1,2:4], xyz1_w[2,2:4], color='green', alpha=alpha)
    ax.plot(xyz1_w[0,4:], xyz1_w[1,4:], xyz1_w[2,4:], color='blue', alpha=alpha)
    ax.text(xyz1_w[0,0], xyz1_w[1,0], xyz1_w[2,0], sys_name, 'x', color='black', alpha=alpha)
    