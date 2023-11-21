# imports
import os
import moviepy.editor as mpy

# make an mp4 from folder of images
def make_mp4(folder='animation', name='NAME'):
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
    concat_clip.write_videofile("video_exports/{}.mp4".format(name), fps=24)

# create and export an animation from images in animation folder
make_mp4(folder='animation', name='NAME')