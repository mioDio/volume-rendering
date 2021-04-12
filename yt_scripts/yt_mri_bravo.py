import yt
import numpy as np
import nibabel as nib
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import Scene, VolumeSource, OpaqueSource, PointSource
import time
import os

res_x = 1.2
res_y = 1.
res_z = 1.

def yt_render(f_path, brain_path, anim=False):
    wat_img = nib.load(f_path)
    wat_arr = wat_img.get_fdata() + 0.1
    # wat_arr =  np.rot90(wat_arr,
    #                       k=3,
    #                       axes=(1, 2))
    print(wat_img.header)
    print(f"Water min: {wat_arr.min()}")
    print(f"Water max: {wat_arr.max()}")
    print(f"Water shape: {wat_arr.shape}")
    
    alpha_wat = 3.
    alpha_brain = 1.0
    
    wat_data = dict(density = (wat_arr, "g/cm**3"))
    ext_x = wat_arr.shape[0] * res_x
    ext_y = wat_arr.shape[1] * res_y
    ext_z = wat_arr.shape[2] * res_z
    wat_bbox = np.array([[-ext_x/2.,ext_x/2.],
                         [-ext_y/2.,ext_y/2.],
                         [-ext_z/2.,ext_z/2.]])
    wat_ds = yt.load_uniform_grid(wat_data, wat_arr.shape,
                                  length_unit = "mm",
                                  bbox=wat_bbox,
                                  nprocs=4)
    wat_ds.use_ghost_zones =True
    sc = yt.create_scene(wat_ds, field=('density'))

    #### Brain mask #######################
    #######################################
    brain_img = nib.load(brain_path)
    brain_arr = brain_img.get_fdata() + 0.1
    brain_arr =  np.rot90(brain_arr,
                          k=3,
                          axes=(1, 2))
    print(brain_img.header)
    print(f"Brain min: {brain_arr.min()}")
    print(f"Brain max: {brain_arr.max()}")
    print(f"Brain shape: {brain_arr.shape}")
    brain_data = dict(density = (brain_arr, "g/cm**3"))
    print(res_x)
    print(res_y)
    scale = 1.2
    brain_bbox = np.array([[-scale *ext_x/2.,scale*ext_x/2.],
                           [-ext_y/2.,ext_y/2.],
                           [-ext_z/2.,ext_z/2.]])
    brain_ds = yt.load_uniform_grid(brain_data, brain_arr.shape,
                                    length_unit = "mm",
                                    bbox=brain_bbox,
                                    nprocs=4)
    brain_source = VolumeSource(brain_ds, field='density')
    sc.add_source(brain_source)

    #### Transfer function Brain image ####
    #######################################
    source_brain = sc[1]
    brain_bounds = (0.1, 3e2)
    brain_tf = yt.ColorTransferFunction(np.log10(brain_bounds))
    brain_tf.add_gaussian(np.log10(.75e2), width=0.005, height=[0., 0., 1., alpha_brain])
    brain_tf.add_gaussian(np.log10(0.5e2), width=0.005, height=[1., 0., 0., alpha_brain])

    source_brain.tfh.tf = brain_tf
    source_brain.tfh.bounds = brain_bounds
    source_brain.tfh.grey_opacity = False
    source_brain.tfh.plot('../render/brain_transfer_function.png', profile_field='density')
    
    #### Transfer function Water image ####
    #######################################
    wat_source = sc[0]
    wat_bounds = (0.1, 5e3)
    wat_tf = yt.ColorTransferFunction(np.log10(wat_bounds))
    # wat_tf.add_gaussian(np.log10(0.5e2), width=0.001, height=[0., 0.75, 1., alpha_wat])
    # wat_tf.add_gaussian(np.log10(1e2), width=0.01, height=[1., 1., 1., alpha_wat])
    wat_tf.add_gaussian(np.log10(2e2), width=0.001, height=[0., .75, 1., alpha_wat])
    # wat_tf.add_gaussian(np.log10(2e3), width=0.001, height=[1., 0., 0., alpha_wat])


    wat_source.tfh.tf = wat_tf
    wat_source.tfh.bounds = wat_bounds
    wat_source.tfh.grey_opacity = False
    wat_source.tfh.plot('../render/wat_transfer_function.png', profile_field='density')

    ### Set up camera ###
    #####################
    cam = sc.add_camera()
    cam.set_resolution((512, 512))
    cam.focus = wat_ds.domain_center
    cam.north_vector = np.array([0., 0., 1.0])
    cam.position = wat_ds.arr([0., 2., 0.], 'unitary')
    cam.rotate(np.pi,
               rot_vector=np.array([0., 0., 1.]),
               rot_center=wat_ds.domain_center)


    ### Renering ###
    ################
    sc.render()
    sc.save('../render/bravo_mri_head.png', sigma_clip=2.0)
    if anim == True:
        frame = 0
        sc.save('../gif_creation/bravo_head_%04i.png' % frame, sigma_clip=2.0)
        for _ in cam.iter_rotate(2. * np.pi, 180):
            frame += 1
            fname = '../gif_creation/bravo_mri_head_%04i.png' % frame
            if not os.path.isfile(fname):
                sc.render()
                sc.save(fname, sigma_clip=2.0)

    
if __name__ == "__main__":
    f_path = "../data/3d_sag_t1_bravo_nq.nii.gz"
    # f_path = "../data/wm.nii.gz"
    brain_path = "../data/brainmask.nii.gz"
    start_time = time.time()
    yt_render(f_path, brain_path, anim=True)
    print("--- %s seconds ---" % (time.time() - start_time))
