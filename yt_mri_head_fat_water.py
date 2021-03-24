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

def yt_render(wat_path, fat_path, anim=False):
    wat_img = nib.load(wat_path)
    wat_arr = wat_img.get_fdata() + 0.1

    fat_img = nib.load(fat_path)
    fat_arr = fat_img.get_fdata() + 0.1

    print(wat_arr.min())
    print(wat_arr.max())
    print(fat_arr.min())
    print(fat_arr.max())

    alpha_wat = 1.0
    alpha_fat = 1.0

    ### Set up water dataset ###
    ############################
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

    ### Set up fat dataset ###
    ##########################
    fat_data = dict(fat = (fat_arr, "g/cm**3"))
    fat_bbox = np.array([[0, fat_arr.shape[0] * res_x],
                         [0, fat_arr.shape[1] * res_y],
                         [0, fat_arr.shape[2] * res_z]])
    fat_ds = yt.load_uniform_grid(fat_data, fat_arr.shape,
                                  length_unit = "mm",
                                  bbox=fat_bbox,
                                  nprocs=4)
    fat_source = VolumeSource(fat_ds, field='fat')
    fat_source.use_ghost_zones =True
    sc.add_source(fat_source)
    
    #### Transfer function Fat image ####
    #######################################
    source_fat = sc[1]
    fat_bounds = (0.1, 3e3)
    fat_tf = yt.ColorTransferFunction(np.log10(fat_bounds))
    # fat_tf.add_gaussian(np.log10(0.5e2), width=0.001, height=[0., 0.75, 1., alpha_fat])
    # fat_tf.add_gaussian(np.log10(2e2), width=0.001, height=[1., 0., 0., alpha_fat - 0.5])
    fat_tf.add_gaussian(np.log10(6e2), width=0.005, height=[1., 0., 0., alpha_fat])

    source_fat.tfh.tf = fat_tf
    source_fat.tfh.bounds = fat_bounds
    source_fat.tfh.grey_opacity = False
    source_fat.tfh.plot('render/fat_transfer_function.png')

    #### Transfer function Water image ####
    #######################################
    wat_source = sc[0]
    wat_bounds = (0.1, 3e3)
    wat_tf = yt.ColorTransferFunction(np.log10(wat_bounds))
    wat_tf.add_gaussian(np.log10(1.2e3), width=0.001, height=[1., 0., 0., alpha_wat])
    wat_tf.add_gaussian(np.log10(2e2), width=0.001, height=[0., 0.75, 1., alpha_wat - 0.5])
    wat_tf.add_gaussian(np.log10(5e2), width=0.001, height=[0., 0.75, 1., alpha_wat])

    # wat_tf.add_gaussian(np.log10(1.4e3), width=0.001, height=[1., 0., 0., alpha_wat])
    # wat_tf.add_gaussian(np.log10(1.8e3), width=0.001, height=[0., 1., 0., alpha_wat])
    
    #wat_tf.add_layers(8)
    wat_source.tfh.tf = wat_tf
    wat_source.tfh.bounds = wat_bounds
    wat_source.tfh.grey_opacity = False
    wat_source.tfh.plot('render/wat_transfer_function.png')

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
    sc.save('render/wat_fat_mri_head.png', sigma_clip=2.0)
    if anim == True:
        frame = 0
        sc.save('gif_creation/wat_fat_mri_head_%04i.png' % frame, sigma_clip=2.0)
        for _ in cam.iter_rotate(2. * np.pi, 180):
            frame += 1
            fname = 'gif_creation/wat_fat_mri_head_%04i.png' % frame
            if not os.path.isfile(fname):
                sc.render()
                sc.save(fname, sigma_clip=2.0)

    
if __name__ == "__main__":
    wat_path = "data/2_water_01_headneck.nii.gz"
    fat_path = "data/200_fat_01_headneck.nii.gz"
    start_time = time.time()
    yt_render(wat_path, fat_path, anim=False)
    print("--- %s seconds ---" % (time.time() - start_time))
