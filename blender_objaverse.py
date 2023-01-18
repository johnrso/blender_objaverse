import bpy, bpycv
from bpycv import pose_utils
import objaverse
import math, mathutils
from mathutils import Vector, Matrix
import numpy as np
import cv2
from PIL import Image
import os
import mediapy as mp

def get_objaverse_objects(tag_list=["faucet"]):
    """
    Get a list of objects from the objaverse with the given tag list

    :param tag_list: the list of tags to search for
    :type tag_list: list

    :return: a dictionary of objects with the uid as the key, and the glb filepath as the value
    """
    def find_tag(anno, tag_list=["faucet"]):
        for tag in anno['tags']:
            if tag['name'] in tag_list:
                return True

        return False

    annotations = objaverse.load_annotations()
    uids = [uid for uid, annotation in annotations.items() if find_tag(annotation, tag_list=tag_list)]
    obs = objaverse.load_objects(uids[:50])

    return obs

def save_rendered_image(camera, path, file_name):
    """
    Save the rendered image from the given camera to the given path

    :param camera: the camera to render from
    :type camera: bpy.types.Object
    :param path: the path to save the image to
    :type path: str
    :param file_name: the name of the file to save
    :type file_name: str
    """
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = f"{path}/{file_name}"
    bpy.ops.render.render(write_still=True)
    res = bpycv.render_data()

    # ipdb checkpoint
    rgb_img = Image.fromarray(res['image'])
    rgb_img.save(f"{path}/{file_name}_rgb.png")

    mask = (res["inst"] / 1001 * 255)
    mask = np.stack((mask, mask, mask), axis=2)
    mask_img = Image.fromarray(np.uint8(mask))
    mask_img.save(f"{path}/{file_name}_mask.png")

    depth_img = Image.fromarray(np.uint16(res['depth'] * 1000))
    depth_img.save(f"{path}/{file_name}_depth_vis.png")

    # change depth shape from (640, 640) to (640, 640, 3)
    depth = (res["depth"] / 1000) # default blender units is mm, switch to meters
    depth = np.stack((depth, depth, depth), axis=2)
    np.save(f"{path}/{file_name}_depth.npy", depth)


def rotate_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), location=mathutils.Vector((0.0, 0.0, 0.0)), distance=50.0):
    """
    Moves the camera, then focuses the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera to move
    :type camera: bpy.types.Object
    :param focus_point: the point to focus on
    :type focus_point: mathutils.Vector
    :param location: the location of the camera
    :type location: mathutils.Vector
    :param distance: the distance from the camera to the focus point
    :type distance: float
    """
    camera.location = location

    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))

def render_object(object):
    """
    Render the given object. Clears the scene before rendering and adds a camera and light.
    The object is placed at the origin.

    :param object: the glb file of the object to render
    :type object: str

    """
    print(object)
    # clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # import the object
    bpy.ops.import_scene.gltf(filepath=object)

    # merge the mesh into one object. this is so that the object's dimensions can be calculated
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            # bpy.ops.object.mode_set(mode='EDIT')
            # bpy.ops.mesh.split_normals()

    bpy.ops.object.mode_set(mode='OBJECT')
    # get the dimensions of all selected objects
    # print([x.dimensions for x in bpy.context.selected_objects])
    bpy.ops.object.join()
    bpy.ops.object.select_all(action='DESELECT')

    # select the mesh object
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            this_obj.select_set(True)

    # get the dimensions of the object
    obj = bpy.context.selected_objects[0]
    context.view_layer.objects.active = obj
    scale = max(obj.dimensions)

    # clear the scene again and import the object again
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=object)

    # merge the mesh into one object. this is so that the object's dimensions can be calculated
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            this_obj['inst_id'] = 1001
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.split_normals()

    bpy.ops.transform.resize(value=(500/scale, 500/scale, 500/scale))
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    # import ipdb; ipdb.set_trace()

    # add a camera
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 640
    bpy.context.object.data.clip_end = 10000

    # add an area light that points to the origin and is high up
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 2000), rotation=(0, 0, 0))
    bpy.context.object.data.energy = 2.0

    bpy.ops.object.light_add(type='AREA')
    light2 = bpy.data.lights['Area']
    light2.energy = 30000
    bpy.data.objects['Area'].scale[0] = 1000
    bpy.data.objects['Area'].scale[1] = 1000
    bpy.data.objects['Area'].scale[2] = 1000

def dump_object(save_dir, i):
    """
    Dump the captured render, the camera matrix, and the depth map to the given directory

    :param save_dir: the directory to save the data to
    :type: save_dir: str
    :param i: the index of the object
    :type: int
    """

    # set the index as a string padded to 6 digits
    i_str = str(i).zfill(6)

    # save the rendered image
    save_rendered_image(bpy.context.scene.camera, save_dir, f"{i_str}")

    # save the camera matrix
    # camera_matrix = bpy.context.scene.camera.matrix_world

    camera_matrix = pose_utils.get_4x4_world_to_cam_from_blender(bpy.context.scene.camera)

    # camera_matrix is currently uvz (RDF); convert to FLU
    # note that this is the same as in get_uvz_to_sapien
    rot = np.array([[0, -1,  0,  0],
                    [0,  0, -1,  0],
                    [1,  0,  0,  0],
                    [0,  0,  0,  1]])

    camera_matrix = rot @ camera_matrix
    np.save(f"{save_dir}/{i_str}_cam_pose.npy", camera_matrix)

def collect_one_object(root_save_dir, uid, glb, num_samples=100, distance=1500.0, phi=2*np.pi/3):
    """
    Collect data for one object

    :param root_save_dir: the root directory to save the data to
    :type root_save_dir: str
    :param uid: the uid of the object
    :type uid: str
    :param glb: the glb file of the object
    :type glb: str
    :param num_samples: the number of samples to take
    :type num_samples: int
    :param distance: the distance from the object to the camera
    :type distance: float
    :param phi: the angle to rotate the camera on the vertical axis (0 is straight down, pi/2 is straight out)
    :type phi: float
    """

    save_dir = f"{root_save_dir}/{uid}"
    render_object(glb)

    for i in range(num_samples):
        # rotate about a circle around the object. i controls the angle theta. phi controls the angle phi
        theta = 2 * math.pi * i / num_samples
        x = math.cos(theta) * math.cos(phi)
        y = math.sin(theta) * math.cos(phi)
        z = math.sin(phi)

        rotate_camera(bpy.context.scene.camera, location=mathutils.Vector((x, y, z)), distance=distance)
        dump_object(save_dir, i)

    # create a video of all of the images that end in rgb.png
    def load_rgb_images(folder):
        rgb_images = []
        for file in os.listdir(folder):
            if file.endswith("rgb.png"):
                rgb_images.append(Image.open((os.path.join(folder, file))))
        # sort the images by the index
        rgb_images.sort(key=lambda x: int(x.filename.split('/')[-1].split('_')[0]))
        rgb_images = [np.array(img) for img in rgb_images]
        return rgb_images

    rgb_images = load_rgb_images(save_dir)
    mp.write_video(f"{save_dir}/_pan.mp4", rgb_images, fps=len(rgb_images))

# use argv to get the filename from the command line and the run in a main wrapper
if __name__ == '__main__':
    import argparse
    import sys, os

    parser = argparse.ArgumentParser()
    # add an argument for the save directory
    parser.add_argument('--save_dir', type=str, default='./data', help='the directory to save the data to')
    parser.add_argument('--debug', action='store_true', help='if true, delete the save directory if it exists')
    parser.add_argument('--num_samples', type=int, default=100, help='the number of samples to take')
    parser.add_argument('--distance', type=float, default=1500.0, help='the distance from the object to the camera')
    parser.add_argument('--phi', type=float, default=2*np.pi/3, help='the angle to rotate the camera on the vertical axis (0 is straight down, pi/2 is straight out)')
    parser.add_argument('--cat', type=str, default='faucet', help='the category to collect data for (e.g. faucet, chair, etc.)')
    parser.add_argument('--N', type=int, default=10, help='the number of objects to collect data for')
    parser.add_argument('--clear', action='store_true', help='if true, clear the cache before collecting data')
    parser.add_argument('--gpu', action='store_true', help='if true, use the GPU')
    parser.add_argument('--tag', type=str, default='', help='a tag to add to the save directory')
    args = parser.parse_args()

    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = "CYCLES"
    render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
    render.image_settings.file_format = 'PNG'
    render.resolution_x = 640
    render.resolution_y = 640
    render.resolution_percentage = 100
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.render.film_transparent = True

    # bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.use_denoising = True

    def enable_cuda_devices():
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        cprefs.get_devices()

        # Attempt to set GPU device types if available
        for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
            try:
                cprefs.compute_device_type = compute_device_type
                print("Compute device selected: {0}".format(compute_device_type))
                break
            except TypeError:
                pass

        # Any CUDA/OPENCL devices?
        acceleratedTypes = ['CUDA', 'OPENCL']
        accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
        print('Accelerated render = {0}'.format(accelerated))

        # If we have CUDA/OPENCL devices, enable only them, otherwise enable
        # all devices (assumed to be CPU)
        print(cprefs.devices)
        for device in cprefs.devices:
            device.use = not accelerated or device.type in acceleratedTypes
            print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

        return accelerated

    enable_cuda_devices()

    save_dir = f"{args.save_dir}/{args.cat}{'_' + args.tag if args.tag else ''}"

    if args.debug or args.clear:
        # delete the save directory if it exists
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)

    # create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # in save_dir, dump the arguments into a json file
    import json
    with open(f"{save_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    # get the objects
    obs = get_objaverse_objects(tag_list=[args.cat])

    # render the objects
    i = 0
    from tqdm import tqdm
    num_obj = min(args.N, len(obs))
    obs = list(obs.items())[:min(5, num_obj)] if args.debug else list(obs.items())[:num_obj]
    for uid, glb in tqdm(obs, total=len(obs), desc='rendering objects'):
        collect_one_object(save_dir,
                           f"{i}_{uid}",
                           glb,
                           num_samples=10 if args.debug else args.num_samples,
                           distance=args.distance,
                           phi=2*np.pi/3)
        i += 1
