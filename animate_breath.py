import polyscope as ps
import numpy as np
import polyscope.imgui as psim
import os,re,time,math

def read_terminaldvdt_bin(binpath):
    """
    Binary file stores the volume of each terminal node at each timestep (assuming that nodes have the same no. of timesteps)
    Returns:
    - sorted terminal node numbers
    - terminal volumes sorted by node numbers and stored in timesteps.
    """
    nodes = []
    all_values = []
    with open(binpath, "rb") as f:
        while True:
            raw_node = f.read(4)
            if not raw_node:
                break
            node = np.frombuffer(raw_node, dtype=np.int32)[0]
            nodes.append(node)

            count = np.frombuffer(f.read(4), dtype=np.int32)[0]
            vals = np.frombuffer(f.read(count * 8), dtype=np.float64)
            all_values.append(vals)

    nodes = np.array(nodes, dtype=np.int32)
    all_values = np.array(all_values)  # shape: (num_nodes, num_timesteps)

    # sort by ascending node number
    sort_idx = np.argsort(nodes)
    nodes_sorted = nodes[sort_idx]
    all_values_sorted = all_values[sort_idx, :]  # reorder rows

    # transpose to (num_timesteps, num_nodes)
    values_by_timestep = all_values_sorted.T
    return nodes_sorted, values_by_timestep

def read_exnodedata(file_path,extn='.exnode'):
    """
    Extract data from .exnode
    Return data in dictionary = {(field name, num Components):[array of field values]}
    """
    try:
        file_path, ext = os.path.splitext(file_path)
        if bool(ext) is False:
            ext = extn

        with open((file_path+ext), 'r') as file:
            lines = file.readlines()

        results = {}  # Dictionary to store the results
        for line in lines:
            if ')' in line:
                # Find the closing parenthesis
                close_paren_index = line.find(')')
                # Find the next comma after the closing parenthesis
                next_comma_index = line.find(',', close_paren_index)
                if next_comma_index != -1:
                    # Extract the substring between ')' and ','
                    words_between = line[close_paren_index + 1:next_comma_index].strip()
                else:
                    words_between = line[close_paren_index + 1:].strip()  # If no comma, get till the end

                # Look for "Components=" and check the subsequent character
                components_index = line.find("Components=")
                if components_index != -1:
                    start_index = components_index + len("Components=")
                    if start_index < len(line):  # Ensure there's a character after "Components="
                        subsequent_char = line[start_index]
                        if subsequent_char.isdigit():
                            # Create the key as a tuple (word, integer)
                            key = (words_between, int(subsequent_char))
                            # Add the tuple as a key to the dictionary with a placeholder value
                            results[key] = None  # Placeholder value; replace with desired value

        ## Collect and store index of each node
        indices = [i for i, s in enumerate(lines) if 'Node' in s]
        float_pattern = r"-?\d+\.\d+" # Regex pattern to match float numbers
        int_pattern = r"\b\d+\b" # Regex pattern to match integer numbers
        float_int_pattern = r'[+-]?\d+(?:\.\d+)?' # Regex pattern to match float or integer numbers
        
        list_node_num = []
        for idx in range(len(indices)):
            line = lines[indices[idx]]
            node_number = int(re.findall(int_pattern, line)[-1])
            list_node_num.append(node_number)

        prev_num_components = 0
        iterator = iter(results.items())
        for field in range(len(results)):
            entry = next(iterator)
            key = entry[0]
            components = entry[0][1]
            # print("Key (field name, components):", key) # field name, components
            # print("Components:", components)

            my_list = []
            for idx in range(len(indices)):
                for i in range(components):
                    line = lines[indices[idx]+1+prev_num_components+i]
                    value = float(re.findall(float_int_pattern, line)[0])
                    my_list.append(value)

            if components>1:
                # Reshape into a 2D list
                my_list = [my_list[i:i+components] for i in range(0, len(my_list), components)]

            results[key] = my_list

            prev_num_components = prev_num_components + components

        results[("Node number",None)] = list_node_num
        return results

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

def extract_coordinates(file_path):
    """
    Read coordinates from .ipnode into numpy array.
    """
    coordinates = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Node number"):
            i += 1
            x = float(lines[i].split(":")[-1].strip())
            i += 1
            y = float(lines[i].split(":")[-1].strip())
            i += 1
            z = float(lines[i].split(":")[-1].strip())
            coordinates.append([x, y, z])
        i += 1
    
    return np.array(coordinates)

def extract_global_numbers(file_path):
    """
    Reads tree.ipelem connections into numpy array.
    """
    global_numbers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Element number"):
            while not lines[i].strip().startswith("Enter the 2 global numbers"):
                i += 1
            numbers = list(map(lambda x: int(x), lines[i].split(":")[-1].strip().split()))
            global_numbers.append(numbers)
        i += 1
    
    return np.array(global_numbers)

# === READ DATA ===

# Read terminal unit coordinates
units_dict = read_exnodedata('ps_demo_data/terminal.exnode')

# Read terminaldvdt.bin. A human-readable txt file is fine too, we use a binary file here as it's 2x smaller than txt.
node_numbers, vals_by_frame = read_terminaldvdt_bin('ps_demo_data/terminaldvdt.bin')

# Sort coordinates by Node numbers
combined = list(zip(units_dict[('Node number', None)], units_dict[('coordinates', 3)])) # Zip node numbers with coordinates
combined_sorted = sorted(combined, key=lambda x: x[0]) # Sort both by node number
sorted_nodes, sorted_coords = zip(*combined_sorted) # Unzip the sorted pairs back into separate lists

# Read tree (ipnode or exelem, as long the connections are read in correctly)
nodes = extract_coordinates('grown.ipnode')
edges = extract_global_numbers('grown.ipelem')
edges = edges - 1 # IMPORTANT: python is zero-indexing. Shift corresponding Node numbers accordingly.

# === POLYSCOPING BEGINS ===

ps.init() # initialise polyscope

# == Register point cloud ==
pts = ps.register_point_cloud("units", np.array(sorted_coords),radius=0.007)

# Add properties to point cloud

# But first, to visibly see unit expansion in animation, scale up values
vols_all_frames = [arr**2+500 for arr in vals_by_frame] # arbitrary scaling

# Get min, max values to clamp colormap range. If no clamp, colour no change in animation.
vmin = math.floor(np.min(vols_all_frames))
vmax = math.ceil(np.max(vols_all_frames))
pts.add_scalar_quantity("volume", vols_all_frames[0],vminmax=(vmin,vmax),enabled=True) # varies node colours. Initialise w/ first timestep values.
pts.set_point_radius_quantity("volume") # varies node size

# == Register curve network
tree = ps.register_curve_network("airway",nodes,edges,radius=0.0009,color=[0.9,1.0,0.1])

# == CALLBACK FUNCTION FOR ANIMATION ==

n_frames = len(vols_all_frames) # get number of frames
curr_frame = 0 # parameter to manipulate thru UI
auto_playing = False # Play/Pause button

def callback():
    global curr_frame, auto_playing

    update_frame_data = False
    _, auto_playing = psim.Checkbox("Autoplay", auto_playing)

    # Advance the frame
    if auto_playing:
        update_frame_data = True
        curr_frame = (curr_frame + 1) % n_frames
        time.sleep(0.1) # add latency to slow down animation (optional)

    # Slider to manually scrub through frames  
    slider_updated, curr_frame = psim.SliderInt("Curr Frame", curr_frame, 0, n_frames-1)
    update_frame_data = update_frame_data or slider_updated

    # Update the scene content if-needed
    if update_frame_data:
        pts.add_scalar_quantity("volume", vols_all_frames[curr_frame],vminmax=(vmin,vmax),enabled=True)
        pts.set_point_radius_quantity("volume")

ps.set_user_callback(callback)

cor_plane_pos = ps.add_scene_slice_plane()
cor_plane_pos.set_pose([0,0,0],[0,1,0])
cor_plane_pos.set_active(False)
cor_plane_neg = ps.add_scene_slice_plane()
cor_plane_neg.set_pose([0,0,0],[0,-1,0])
cor_plane_neg.set_active(False)

ax_plane_pos = ps.add_scene_slice_plane()
ax_plane_pos.set_pose([0,0,0],[0,0,1])
ax_plane_pos.set_active(False)
ax_plane_neg = ps.add_scene_slice_plane()
ax_plane_neg.set_pose([0,0,0],[0,0,-1])
ax_plane_neg.set_active(False)

ps.set_ground_plane_mode("none")
ps.set_navigation_style("free")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0])
ps.show() # give control to the main loop, blocks until window is exited
