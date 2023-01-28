import bpy, bpycv
import tqdm
from bpycv import pose_utils
import objaverse
import math, mathutils
from mathutils import Vector, Matrix
import numpy as np
import cv2
from PIL import Image
import os
import mediapy as mp
import matplotlib.pyplot as plt
import random

def get_objaverse_objects(tag_list=["faucet"], lvis=True):
    """
    Get a list of objects from the objaverse with the given tag list

    :param tag_list: the list of tags to search for
    :type tag_list: list

    :return: a dictionary of objects with the uid as the key, and the glb filepath as the value
    """

    lvis_annotations = objaverse.load_lvis_annotations()
    if tag_list[0] in lvis_annotations and lvis:
        print("tag found in lvis annotations")
        uids = lvis_annotations[tag_list[0]]

    else:
        def find_tag(anno, tag_list=["faucet"]):
            for tag in anno['tags']:
                if tag['name'] in tag_list:
                    return True

            return False

        annotations = objaverse.load_annotations()
        uids = [uid for uid, annotation in annotations.items() if find_tag(annotation, tag_list=tag_list)]
    obs = objaverse.load_objects(uids)

    return obs

class BlenderObjaverseRenderer:
    def __init__(self, args):

        self.context = bpy.context
        self.scene = self.context.scene
        self.render = self.scene.render

        self.render.engine = "CYCLES"
        self.render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
        self.render.image_settings.file_format = 'PNG'
        self.render.resolution_x = 640
        self.render.resolution_y = 640
        self.render.resolution_percentage = 100
        bpy.context.scene.cycles.filter_width = 0.01
        bpy.context.scene.render.film_transparent = True

        if args.gpu:
            bpy.context.scene.cycles.device = 'GPU'
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

        self.save_dir = f"{args.save_dir}/{args.cat}{'_lvis' if args.lvis else ''}_samp{args.num_samples}_num{num_obj}{'_' + args.tag if args.tag else ''}{f'_debug' if args.debug else ''}"

        self.cam = self.scene.objects["Camera"]
        self.cam.location = (0, 1.2, 0)
        self.cam.data.lens = 35
        self.cam.data.sensor_width = 32

        # get self.cam's intrinsic matrix
        self.cam_constraint = self.cam.constraints.new(type="TRACK_TO")
        self.cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        self.cam_constraint.up_axis = "UP_Y"

        # setup lighting
        bpy.ops.object.light_add(type="AREA")
        self.light2 = bpy.data.lights["Area"]
        self.light2.energy = 30000
        bpy.data.objects["Area"].location[2] = 0.5
        bpy.data.objects["Area"].scale[0] = 100
        bpy.data.objects["Area"].scale[1] = 100
        bpy.data.objects["Area"].scale[2] = 100

        self.mesh = None
        self.num_samples = args.num_samples
        self.distance_range = args.distance_range
        self.phi_range = args.phi_range
        self.debug = args.debug

    def randomize_lighting(self):
        self.light2.energy = random.uniform(25000, 50000)
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 0
        bpy.data.objects["Area"].location[2] = random.uniform(1, 2)


    def reset_lighting(self):
        self.light2.energy = 30_000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 0
        bpy.data.objects["Area"].location[2] = 0.5


    def join_meshes(self) -> bpy.types.Object:
        """Joins all the meshes in the scene into one mesh."""
        # get all the meshes in the scene
        meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
        bpy.ops.object.select_all(action="DESELECT")
        for mesh in meshes:
            for uvmap in mesh.data.uv_layers:
                uvmap.name = 'UVMap'
            mesh.select_set(True)
            bpy.context.view_layer.objects.active = mesh
        # join the meshes
        bpy.ops.object.join()

        meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
        assert len(meshes) == 1
        mesh = meshes[0]
        self.mesh = mesh
        self.mesh['inst_id'] = 1001


    def center_mesh(self):
        """Centers the mesh at the origin."""
        # select the mesh
        bpy.ops.object.select_all(action="DESELECT")
        self.mesh.select_set(True)
        # clear and keep the transformation of the parent
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        # set the mesh position to the origin, use the bounding box center
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        bpy.context.object.location = (0, 0, 0)
        # 0 out the transform
        bpy.ops.object.transforms_to_deltas(mode="ALL")


    def resize_object(self, max_side_length_meters) -> None:
        """Resizes the object to have a max side length of max_side_length_meters meters."""
        # select the mesh
        bpy.ops.object.select_all(action="DESELECT")
        self.mesh.select_set(True)
        # get the bounding box
        x_size, y_size, z_size = self.mesh.dimensions
        # get the max side length
        curr_max_side_length = max([x_size, y_size, z_size])
        # get the scale factor
        scale_factor = max_side_length_meters / curr_max_side_length
        # scale the object
        bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
        # 0 out the transform
        bpy.ops.object.transforms_to_deltas(mode="ALL")


    def reset_scene(self):
        """Resets the scene to a clean state."""
        # delete everything that isn't part of a camera or a light
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        # delete all the materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        # delete all the textures
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        # delete all the images
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    # load the glb model
    def load_object(self, object_path: str) -> None:
        """Loads a glb model into the scene."""
        assert object_path.endswith(".glb")
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)

    def render_object(self, glb):
        self.reset_scene()
        self.load_object(glb)
        self.join_meshes()
        self.center_mesh()
        self.resize_object(0.7)
        # bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

        # render the object
        self.randomize_lighting()


    def save_rendered_image(self, path, file_name):
        """
        Save the rendered image from the given camera to the given path

        :param camera: the camera to render from
        :type camera: bpy.types.Object
        :param path: the path to save the image to
        :type path: str
        :param file_name: the name of the file to save
        :type file_name: str
        """
        bpy.context.scene.camera = self.cam
        bpy.ops.render.render(write_still=True)
        res = bpycv.render_data()

        rgb_img = Image.fromarray(res['image'])
        rgb_img.save(f"{path}/{file_name}_rgb.png")

        mask = (res["inst"] / 1001 * 255)
        mask = np.stack((mask, mask, mask), axis=2)
        mask_img = Image.fromarray(np.uint8(mask))
        mask_img.save(f"{path}/{file_name}_mask.png")

        # change depth shape from (640, 640) to (640, 640, 3)
        depth = (res["depth"]) # default blender units is mm, switch to meters
        depth = np.stack((depth, depth, depth), axis=2)
        np.save(f"{path}/{file_name}_depth.npy", depth)
        plt.imsave(f"{path}/{file_name}_depth_vis.png", depth[..., 0])

    def sapien_camera_matrix_from_lookat(self, eye, at, up):
        # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # The principle axis of the camera is the x-axis
        forward = (at - eye) / np.linalg.norm(at - eye)
        left = np.cross(up, forward) / np.linalg.norm(np.cross(up, forward))
        upward = np.cross(forward, left)
        R = np.stack([forward, left, upward], axis=1)
        mat44 = np.eye(4)
        mat44[:3, :3] = R
        mat44[:3, 3] = eye
        return mat44

    def dump_object(self, save_dir, i):
        """
        Dump the captured render, the camera matrix, and the depth map to the given directory

        :param save_dir: the directory to save the data to
        :type: save_dir: str
        :param i: the index of the object
        :type: int
        """

        # set the index as a string padded to 6 digits
        i_str = str(i).zfill(6)
        self.randomize_lighting()
        # save the rendered image
        self.save_rendered_image(save_dir, f"{i_str}")
        self.reset_lighting()
        # save the camera matrix
        # camera_matrix = bpy.context.scene.camera.matrix_world
        info = pose_utils.get_K_world_to_cam(bpy.context.scene.camera)
        K = info["intrinsic_matrix"]

        eye = np.array(self.cam.location)
        at_axis = np.array(self.cam.matrix_world.to_quaternion() @ Vector((0, 0, -1)))
        up_axis = np.array(self.cam.matrix_world.to_quaternion() @ Vector((0, 1, 0)))
        camera_matrix = self.sapien_camera_matrix_from_lookat(eye, at_axis, up_axis)

        K_fn = f"{self.save_dir}/_K.npy"
        if not os.path.exists(K_fn):
            np.save(K_fn, K)

        rot = np.array([[0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [1,  0,  0,  0],
                        [0,  0,  0,  1]]).T # this matches the one in the other dataset

        # camera_matrix = camera_matrix
        camera_matrix = camera_matrix @ rot

        np.save(f"{save_dir}/{i_str}_cam_pose.npy", camera_matrix)

    def collect_one_object(self, uid, glb):
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


        save_dir = f"{self.save_dir}/{uid}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.render_object(glb)

        num_samples = 24 if self.debug else self.num_samples
        for i in range(num_samples):
            if self.debug:
                theta = i * 2 * np.pi / num_samples
                phi = np.pi / 4
                distance = 1.5
            else:
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(self.phi_range[0], self.phi_range[1])
                distance = np.random.uniform(self.distance_range[0], self.distance_range[1])
            x = math.cos(theta) * math.cos(phi) * distance
            y = math.sin(theta) * math.cos(phi) * distance
            z = math.sin(phi) * distance

            print(x,y,z)


            self.cam_constraint.target = self.mesh
            self.cam.location = (x, y, z)
            self.dump_object(save_dir, i)

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
    # add an argument for a range of distances from the camera to the object
    parser.add_argument('--distance_range', type=float, nargs=2, default=[.7, 1.5], help='the range of distances from the object to the camera')
    # add an argument for the range of angles to rotate the camera on the vertical axis (0 is straight down, pi/2 is straight out)
    parser.add_argument('--phi_range', type=float, nargs=2, default=[np.pi / 6, np.pi / 2], help='the range of angles to rotate the camera on the vertical axis (0 is straight down, pi/2 is straight out)')
    parser.add_argument('--cat', type=str, default='faucet', help='the category to collect data for (e.g. faucet, chair, etc.)')
    parser.add_argument('--N', type=int, default=1000, help='the number of objects to collect data for')
    parser.add_argument('--clear', action='store_true', help='if true, clear the cache before collecting data')
    parser.add_argument('--gpu', action='store_true', help='if true, use the GPU')
    parser.add_argument('--tag', type=str, default='', help='a tag to add to the save directory')
    parser.add_argument('--lvis', action='store_true', help='if true, use the lvis dataset')
    args = parser.parse_args()

    obs = get_objaverse_objects(tag_list=[args.cat], lvis=args.lvis)
    num_obj = min(args.N, len(obs))
    obs = list(obs.items())[:min(5, num_obj)] if args.debug else list(obs.items())[:num_obj]

    save_dir = f"{args.save_dir}/{args.cat}{'_lvis' if args.lvis else ''}_samp{args.num_samples}_num{num_obj}{'_' + args.tag if args.tag else ''}{f'_debug' if args.debug else ''}"
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

    i = 0
    obj_rend = BlenderObjaverseRenderer(args)
    for uid, glb in tqdm.tqdm(obs, total=len(obs), desc='rendering objects'):
        obj_rend.collect_one_object(f"{i}_{uid}", glb)
        i += 1