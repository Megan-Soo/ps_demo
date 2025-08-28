import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import time

# Start polyscope
ps.init()

# Example geometry (sphere of points)
points = np.random.randn(100, 3) * 0.1
ps.register_point_cloud("points", points)

# Animation parameters
n_frames = 180
radius = 2.0
elevation = 0.3  # radians
target = (0, 0, 0)
up_dir = (0, 0, 1)   # <- important: prevents flipping halfway during turntable animation.
# Flipping happens bc Polyscope keeps trying to choose the “up” direction automatically when you only give camera_location and target to ps.look_at(cam_coord, target_coord).
# When the camera goes all the way around, the computed up vector flips, which makes the scene look like it’s turning upside down halfway.
# To avoid flipping, you want to lock the camera’s up direction. Use ps.look_at_dir(cam_coord, target_coord, up_dir)

curr_frame = 0 # parameter to manipulate thru UI
auto_playing = False

def callback():
    global curr_frame, auto_playing

    update_frame_data = False
    _, auto_playing = psim.Checkbox("Autoplay", auto_playing)

    # Advance the frame
    if auto_playing:
        update_frame_data = True
        curr_frame = (curr_frame + 1) % n_frames
        time.sleep(0.05) # add latency to slow down animation (optional)

    # Slider to manually scrub through frames  
    slider_updated, curr_frame = psim.SliderInt("Curr Frame", curr_frame, 0, n_frames-1)
    update_frame_data = update_frame_data or slider_updated

    # Update the scene content if-needed
    if update_frame_data:
        angle = 2 * np.pi * curr_frame / n_frames  # full revolution
        
        # spherical to cartesian camera position

        # # for all around revolution, don't fix up_dir (use ps.look_at instead of ps.look_at_dir)
        # x = radius * np.cos(angle) * np.cos(elevation)
        # y = radius * np.sin(angle) * np.cos(elevation)
        # z = radius * np.sin(elevation)
        # camera_pos = (x,y,z)
        # ps.look_at(camera_pos,target,fly_to=False)
        
        # for turntable revolution
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = elevation # set camera's z to a fixed height

        camera_pos = (x, y, z)
    
        # set camera to look at the object
        ps.look_at_dir(camera_pos, target, up_dir, fly_to=False)

ps.set_user_callback(callback)

ps.set_background_color([0,0,0])
ps.set_ground_plane_mode('none')
ps.show()