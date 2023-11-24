# imports
import os
import numpy as np
import moviepy.editor as mpy
from mayavi import mlab

# folder paths
input_path = "./exports/tensors"
output_path = "./exports/images"

# render tensor with mayavi
def render(tensor, show=True, save=False, name="image.png"):
    """
    Render a tensor using Mayavi for visualization.

    Parameters:
    - tensor: The tensor object.
    - show: Whether to display the visualization.
    - save: Whether to save the visualization.
    - name: Name of the saved image.
    """
    # figure
    fig = mlab.figure(size=(1024, 1024))
    # Define colors based on your values
    render_list = {
        1: (1, 1, 1),    # Soil
        #-1: (1, 0, 0),    # Objects
        #-2:(0, 0, 1),    # Agents
        2:(1, 0.75, 0) # Built structure
        }

    # plotting voxels
    for value, color in render_list.items():
        x_vals, y_vals, z_vals = np.where(tensor == value)
        mlab.points3d(x_vals, y_vals, z_vals,
                        mode="cube",
                        color=color,
                        scale_factor=1)

    # save image
    if save:
        mlab.savefig(name)
        mlab.close()
    # show image
    if show:
        mlab.show()

# render all files in a folder and export to other
def render_all(folder_in, folder_out):
    """
    Creates images from stored data and stores in a folder.

    Parameters:
    - folder_in (str): The path to the folder containing data.
    - folder_out (str): The output path

    Returns:
    - None

    Note:
    - Tensors need to be named as something_time.npy
    """
    # set offscreen option
    mlab.options.offscreen = True
    # File list
    file_list = os.listdir(folder_in)
    print(file_list)
    
    # start time
    start_time = time.time()
    # Iterate through files and render images
    for file_name in file_list:
        # Construct the full file path
        file_path = os.path.join(folder_in, file_name)
        # Load tensor
        tensor = np.load(file_path)
        # Extract name and time from the file_name (assuming the format is something_time.npy)
        separated = file_name.split('_')
        name, time = separated[-2], separated[-1]
        time = time.split('.')[0]  # Remove the '.npy' extension
        # Construct the output file path
        output_path = os.path.join(folder_out, f"{name}_{time}.png")
        # Render the image (Assuming the render function is defined elsewhere)
        render(tensor, show=False, save=True, name=output_path)
    
    # end time
    end_time = time.time()
    print("total time taken for this loop: ", end_time - start_time)

# make an mp4 from folder of images
def make_mp4(folder="./exports/images", name="generic_name"):
    """
    Creates an MP4 video from a folder containing images.

    Parameters:
    - folder (str): The path to the folder containing images. Default is 'animation'.
    - name (str): The name of the output MP4 file. Default is 'NAME'.

    Returns:
    - None

    Note:
    - The images in the folder should be named with a common prefix and a numerical index
      (e.g., 'frame_1.png', 'frame_2.png', ...).
    - The resulting MP4 video will be saved in the 'video_exports' directory with the specified name.
    """
    # file list
    file_list = (os.listdir(folder))
    print (file_list)
    # remove clutter
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    # sort
    list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0]))
    file_list = [folder+'/'+f for f in file_list]
    # Here set the seconds per frame 0.1 fast, 0.3 average, 0.5 is slow
    clips = [mpy.ImageClip(m).set_duration(0.3) for m in file_list]
    concat_clip = mpy.concatenate_videoclips(clips, method="compose")
    # choose NAME here
    concat_clip.write_videofile("./exports/videos/{}.mp4".format(name), fps=24)