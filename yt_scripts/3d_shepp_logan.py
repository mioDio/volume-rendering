from matplotlib import pyplot as plt
import yt
import numpy as np
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import Scene, VolumeSource, OpaqueSource, PointSource
import phantominator
from phantominator import shepp_logan
import os

def linramp(vals, minval, maxval):
    return (vals - vals.min())/(vals.max() - vals.min())

if __name__ == '__main__':
    p, T1, T2 = shepp_logan((256, 256, 256), MR=True, zlims=(-1, 1))
    print(p.shape)
    print(p.max())
    
    # plt.imshow(p[:, :, 10], origin='lower')
    # plt.histogram(p)
    # plt.show()
    vals = np.unique(p)
    print(vals)
    # 0.12  0.617 0.745 0.8   0.822 0.852 0.93  0.95  0.98  0.98  1.185
    p*=1e4
    p+=.1
    alpha = 1.0
    data = dict(density = (p, "g/cm**3"))
    bbox = np.array([[0, p.shape[0]],
                     [0, p.shape[1]],
                     [0, p.shape[2]]])
    ds = yt.load_uniform_grid(data, p.shape,
                              length_unit = "mm",
                              bbox=bbox,
                              nprocs=4)
    ds.use_ghost_zones = False
    sc = yt.create_scene(ds, field=('density'))

    #### Transfer function  ####
    #######################################
    source = sc[0]
    bounds = (0.9 * vals[1]*1e4, 1.1*vals[-1]*1e4)
    print(1.1*vals[-1]*1e4)
    tf = yt.ColorTransferFunction(np.log10(bounds))
    tf.add_gaussian(np.log10(0.12e4), width=0.0001, height=[1., 1., 1., alpha*0.5])
    tf.add_gaussian(np.log10(0.617e4), width=0.0001, height=[0., 0., 1., alpha*0.5])
    tf.add_gaussian(np.log10(0.745e4), width=0.00005, height=[1., 1., 1., alpha*0.2])
    tf.add_gaussian(np.log10(0.8e4), width=0.00005, height=[1., 0., 0., alpha])
    tf.add_gaussian(np.log10(0.98e4), width=0.0001, height=[0., 0.75, 1., alpha])
    tf.add_gaussian(np.log10(0.78e4), width=0.0001, height=[1., 1., 0., 2.*alpha])
    
    # tf.add_gaussian(np.log10(0.95e4), width=0.0001, height=[0., 0.75, 1., alpha])

    # tf.add_layers(11 ,colormap='arbre')
    # tf.map_to_colormap(0.1, p.max()*2., colormap='RdBu_r',)
    #                    # scale_func=linramp)
    source.tfh.tf = tf
    source.tfh.bounds = bounds
    source.tfh.grey_opacity = False
    source.tfh.plot('transfer_function.png', profile_field='density')

    ### Set up camera ###
    #####################
    cam = sc.add_camera()
    cam.set_resolution((512, 512))
    cam.focus = ds.domain_center
    cam.north_vector = np.array([0., 0., 1.0])
    cam.position = ds.arr([0., 1., 0.5], 'unitary')
    # cam.rotate(np.pi/3.,
    #            rot_vector=np.array([0., 0., 1.]),
    #            rot_center=wat_ds.domain_center)


    ### Renering ###
    ################
    sc.render()
    sc.save('modified_shepp_logan.png')#, sigma_clip=2.0)
    # if anim == True:
    frame = 0
    sc.save('gif_creation/phantom_%04i.png' % frame)#, sigma_clip=2.0)
    for _ in cam.iter_rotate(2. * np.pi, 180):
        frame += 1
        fname = 'gif_creation/phantom_%04i.png' % frame
        if not os.path.isfile(fname):
            sc.render()
            sc.save(fname)#, sigma_clip=2.0)

    
