import yt
import numpy as np
import nibabel as nib
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import Scene, VolumeSource, OpaqueSource, PointSource
import time
import os

res_x = 2.
res_y = 2.
res_z = 5.

def yt_render(f_path, anim=False):
    wat_img = nib.load(f_path)
    wat_arr = wat_img.get_fdata() + 0.1
    print(wat_arr.min())
    print(wat_arr.max())
    alpha_wat = 1.0
    wat_data = dict(density = (wat_arr, "g/cm**3"))
    wat_bbox = np.array([[0, wat_arr.shape[0] * res_x],
                         [0, wat_arr.shape[1] * res_y],
                         [0, wat_arr.shape[2] * res_z]])
    wat_ds = yt.load_uniform_grid(wat_data, wat_arr.shape,
                                  length_unit = "mm",
                                  bbox=wat_bbox,
                                  nprocs=4)
    wat_ds.use_ghost_zones =True
    sc = yt.create_scene(wat_ds, field=('density'))

    #### Transfer function Water image ####
    #######################################
    wat_source = sc[0]
    wat_bounds = (0.1, 3e3)
    wat_tf = yt.ColorTransferFunction(np.log10(wat_bounds))
    # wat_tf.add_gaussian(np.log10(0.5e2), width=0.001, height=[0., 0.75, 1., alpha_wat])
    wat_tf.add_gaussian(np.log10(2e2), width=0.001, height=[0., 0.75, 1., alpha_wat - 0.5])
    wat_tf.add_gaussian(np.log10(5e2), width=0.001, height=[0., 0.75, 1., alpha_wat])

    # wat_tf.add_gaussian(np.log10(1.4e3), width=0.001, height=[1., 0., 0., alpha_wat])
    # wat_tf.add_gaussian(np.log10(1.8e3), width=0.001, height=[0., 1., 0., alpha_wat])
    
    #wat_tf.add_layers(8)
    wat_source.tfh.tf = wat_tf
    wat_source.tfh.bounds = wat_bounds
    wat_source.tfh.grey_opacity = False
    wat_source.tfh.plot('../render/wat_transfer_function.png')

    ### Set up camera ###
    #####################
    cam = sc.add_camera()
    cam.set_resolution((512, 512))
    cam.focus = wat_ds.domain_center
    cam.north_vector = np.array([0., 0., 1.0])
    cam.position = wat_ds.arr([0., 1., 0.5], 'unitary')
    cam.rotate(np.pi/2.,
               rot_vector=np.array([0., 0., 1.]),
               rot_center=wat_ds.domain_center)


    ### Renering ###
    ################
    sc.render()
    sc.save('../render/mri_head.png', sigma_clip=2.0)
    if anim == True:
        frame = 0
        sc.save('../gif_creation/wat_head_%04i.png' % frame, sigma_clip=2.0)
        for _ in cam.iter_rotate(2. * np.pi, 180):
            frame += 1
            fname = '../gif_creation/wat_mri_head_%04i.png' % frame
            if not os.path.isfile(fname):
                sc.render()
                sc.save(fname, sigma_clip=2.0)

    
if __name__ == "__main__":
    f_path = "../data/2_water_01_headneck.nii.gz"
    start_time = time.time()
    yt_render(f_path, anim=False)
    print("--- %s seconds ---" % (time.time() - start_time))


    
