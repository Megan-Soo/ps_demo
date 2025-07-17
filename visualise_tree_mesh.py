import numpy as np
import polyscope as ps
import re
from collections import defaultdict

def extract_coordinates(file_path):
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

def extract_radius(file_path):
    radius_values = []
    pattern = re.compile(r'The field variable value is \[ .*?\]: ([\d\.D\+\-]+)')
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                value = match.group(1).replace('D', 'E')  # Convert Fortran-style exponent
                radius_values.append(float(value))
    
    return np.array(radius_values)

def compute_joint_radii(nodes, edges, edge_mid_radii):
    """
    Given a radius for each edge (at midpoint), compute per-node radius
    as the average of connected edge radii. If a node has only one incident
    edge, use that edge's radius directly.
    
    Parameters
    ----------
    nodes : (N,3)
    edges : (E,2)
    edge_mid_radii : (E,)
    
    Returns
    -------
    joint_radii : (N,) per-node radii
    edge_radii : (E,2) per-edge start/end radii
    """
    N = nodes.shape[0]
    E = edges.shape[0]

    # collect radii per node
    incident = defaultdict(list)
    for e, (i0, i1) in enumerate(edges):
        incident[i0].append(edge_mid_radii[e])
        incident[i1].append(edge_mid_radii[e])

    joint_radii = np.zeros(N, dtype=float)
    for i in range(N):
        if len(incident[i]) == 0:
            joint_radii[i] = 0.0
        elif len(incident[i]) == 1:
            # leaf: just use that edge's radius
            joint_radii[i] = incident[i][0]
        else:
            # average of all connected edge radii
            joint_radii[i] = np.mean(incident[i])

    # now expand to per-edge start/end
    edge_radii = np.zeros((E, 2), dtype=float)
    for e, (i0, i1) in enumerate(edges):
        edge_radii[e, 0] = joint_radii[i0]
        edge_radii[e, 1] = joint_radii[i1]

    return joint_radii, edge_radii

def generate_variable_radius_cylinder_mesh(p0, p1, r0, r1, segments=16):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-8:
        raise ValueError("p0 and p1 are too close together")

    # Normalize axis
    axis = axis / length

    # Find a vector not parallel to axis to build an orthonormal frame
    if abs(axis[0]) < 0.9:
        tmp = np.array([1, 0, 0], dtype=float)
    else:
        tmp = np.array([0, 1, 0], dtype=float)
    v = np.cross(axis, tmp)
    v /= np.linalg.norm(v)
    u = np.cross(axis, v)

    # Generate angular positions
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)

    vertices = []
    normals = []

    # bottom ring
    for t in theta:
        dir_vec = np.cos(t)*u + np.sin(t)*v
        vertices.append(p0 + r0 * dir_vec)
        normals.append(dir_vec)

    # top ring
    for t in theta:
        dir_vec = np.cos(t)*u + np.sin(t)*v
        vertices.append(p1 + r1 * dir_vec)
        normals.append(dir_vec)

    vertices = np.array(vertices)
    normals = np.array(normals)

    # Build faces (triangles)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        # bottom i -> bottom j -> top j
        faces.append([i, j, segments + j])
        # bottom i -> top j -> top i
        faces.append([i, segments + j, segments + i])

    faces = np.array(faces, dtype=int)

    return vertices, normals, faces

def generate_network_tube_mesh(nodes, edges, radii, segments=16):
    """
    Build a triangle mesh by sweeping a cylinder along each edge.
    
    Parameters
    ----------
    nodes : (N,3) float array
        3D coordinates of nodes
    edges : (E,2) int array
        indices into `nodes` for each edge
    radii : (E,2) float array
        radius at start and end of each edge
    segments : int
        number of subdivisions around the circle
    
    Returns
    -------
    vertices : (M,3) float array
    faces : (K,3) int array
    """
    all_vertices = []
    all_faces = []
    vert_offset = 0

    for e, (i0, i1) in enumerate(edges):
        p0 = nodes[i0]
        p1 = nodes[i1]
        r0, r1 = radii[e]

        # --- build tapered cylinder for this edge ---
        axis = p1 - p0
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis = axis / length

        # build frame
        if abs(axis[0]) < 0.9:
            tmp = np.array([1, 0, 0], dtype=float)
        else:
            tmp = np.array([0, 1, 0], dtype=float)
        v = np.cross(axis, tmp)
        v /= np.linalg.norm(v)
        u = np.cross(axis, v)

        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)

        verts = []
        # bottom ring
        for t in theta:
            dir_vec = np.cos(t)*u + np.sin(t)*v
            verts.append(p0 + r0 * dir_vec)
        # top ring
        for t in theta:
            dir_vec = np.cos(t)*u + np.sin(t)*v
            verts.append(p1 + r1 * dir_vec)

        verts = np.array(verts)
        faces = []
        for k in range(segments):
            j = (k+1) % segments
            faces.append([k, j, segments+j])
            faces.append([k, segments+j, segments+k])
        faces = np.array(faces, dtype=int)

        # offset indices
        faces = faces + vert_offset
        vert_offset += verts.shape[0]

        all_vertices.append(verts)
        all_faces.append(faces)

    if len(all_vertices) == 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=int)

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    return vertices, faces

def export_mesh_to_ply(vertices, faces, filepath, ascii=True):
    """
    Export a triangle mesh to a PLY file.

    Parameters
    ----------
    vertices : (N,3) float array
    faces    : (M,3) int array (0-based indices)
    filepath : str
    ascii    : bool, if True write ASCII, else binary little-endian
    """
    N = vertices.shape[0]
    M = faces.shape[0]

    mode = 'ascii' if ascii else 'binary_little_endian'

    with open(filepath, 'wb' if not ascii else 'w') as f:
        # ---- PLY header ----
        header = []
        header.append('ply')
        header.append(f'format {mode} 1.0')
        header.append(f'element vertex {N}')
        header.append('property float x')
        header.append('property float y')
        header.append('property float z')
        header.append(f'element face {M}')
        header.append('property list uchar int vertex_indices')
        header.append('end_header\n')
        header_str = "\n".join(header)
        if ascii:
            f.write(header_str + "\n")
        else:
            f.write(header_str.encode('utf-8'))

        # ---- vertices ----
        if ascii:
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            # ---- faces ----
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        else:
            import struct
            for v in vertices:
                f.write(struct.pack('<fff', *v))
            for face in faces:
                # write uchar count then int32 indices
                f.write(struct.pack('<Biii', 3, face[0], face[1], face[2]))

# Read grown tree data
nodes = extract_coordinates('ps_demo_data/grown.ipnode')
edges = extract_global_numbers('ps_demo_data/grown.ipelem')
edges = edges - 1  # Adjust for zero-indexing
radius = extract_radius('ps_demo_data/grown_radius.ipfiel')
radii = np.column_stack([radius, radius]) # since we only have one radius per edge, we duplicate it for both ends

# Generate mesh
verts, faces = generate_network_tube_mesh(nodes, edges, radii, segments=24)

# For smooth tree mesh: Compute joint radii & per-edge radii
joint_radii, edge_radii = compute_joint_radii(nodes, edges, radius)
# Generate mesh with joint radii
verts_joint, faces_joint = generate_network_tube_mesh(nodes, edges, edge_radii, segments=24)

# Gives us an option to read in mesh later so we don't have to generate mesh from scratch every time
export_mesh_to_ply(verts, faces, "grown.ply", ascii=False)

# Initialize polyscope
ps.init()

# Register your surface mesh
ps.register_surface_mesh("CMGUI-style", verts, faces, smooth_shade=True,enabled=False)
ps.register_surface_mesh("Smooth", verts_joint, faces_joint, smooth_shade=True,enabled=True)

# Misc settings
ps.set_ground_plane_mode("none")
ps.set_navigation_style("free")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0])
ps.show()
