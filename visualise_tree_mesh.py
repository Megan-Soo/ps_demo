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

def generate_network_tube_mesh(nodes, edges, radii, data=None, segments=16):
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
    data : (E,) float array or None
        optional per-edge data to be propagated to mesh vertices
    segments : int
        number of subdivisions around the circle

    Returns
    -------
    vertices : (M,3) float array
    faces : (K,3) int array
    data_per_vertex : (M,) float array or None
        repeated data values for each vertex (or None if no data provided)
    """
    all_vertices = []
    all_faces = []
    all_data = []
    vert_offset = 0

    # flag to know if we assign data
    assign_data = data is not None and len(data) > 0

    for e, (i0, i1) in enumerate(edges):
        p0 = nodes[i0]
        p1 = nodes[i1]
        r0, r1 = radii[e]

        axis = p1 - p0
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis = axis / length

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

        if assign_data:
            all_data.append(np.full(verts.shape[0], data[e]))

    if len(all_vertices) == 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=int), None

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    data_per_vertex = np.concatenate(all_data) if assign_data else None

    return vertices, faces, data_per_vertex

def export_mesh_to_ply(vertices, faces, filepath, ascii=True,
                       n_segments=None, n_edges=None,
                       vertex_data=None):
    """
    Export a triangle mesh with optional per-vertex data to a PLY file.

    Parameters
    ----------
    vertices : (N,3) float array
    faces    : (M,3) int array (0-based indices)
    filepath : str
    ascii    : bool, write ASCII if True, else binary little-endian
    n_segments : int or None, optional for comment
    n_edges : int or None, optional for comment
    vertex_data : (N,) float array or None, extra per-vertex data to store as 'value'
    """
    N = vertices.shape[0]
    M = faces.shape[0]
    if vertex_data is not None and len(vertex_data) != N:
        raise ValueError("vertex_data must have same length as vertices")

    mode = 'ascii' if ascii else 'binary_little_endian'

    with open(filepath, 'wb' if not ascii else 'w') as f:
        # ---- Header ----
        header = []
        header.append('ply')
        header.append(f'format {mode} 1.0')
        if n_segments is not None:
            header.append(f'comment n segments per edge {n_segments}')
        if n_edges is not None:
            header.append(f'comment n edges in tree {n_edges}')
        header.append(f'element vertex {N}')
        header.append('property float x')
        header.append('property float y')
        header.append('property float z')
        if vertex_data is not None:
            header.append('property float value')  # custom property
        header.append(f'element face {M}')
        header.append('property list uchar int vertex_indices')
        header.append('end_header')
        header_str = "\n".join(header) + "\n"
        if ascii:
            f.write(header_str)
        else:
            f.write(header_str.encode('utf-8'))

        # ---- Vertex data ----
        if ascii:
            for i, v in enumerate(vertices):
                if vertex_data is not None:
                    f.write(f"{v[0]} {v[1]} {v[2]} {vertex_data[i]}\n")
                else:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
            # ---- Face data ----
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        else:
            import struct
            for i, v in enumerate(vertices):
                if vertex_data is not None:
                    f.write(struct.pack('<fff', *v))
                    f.write(struct.pack('<f', float(vertex_data[i])))
                else:
                    f.write(struct.pack('<fff', *v))
            for face in faces:
                f.write(struct.pack('<Biii', 3, face[0], face[1], face[2]))

def read_ply_header_comments(filepath):
    """
    Reads the header of a PLY file (ASCII or binary) and extracts
    'n segments per edge' and 'n edges in tree' values from comments.

    Parameters
    ----------
    filepath : str
        Path to the PLY file.

    Returns
    -------
    dict with keys 'n_segments' and 'n_edges', values are int or None if not found.
    """
    n_segments = None
    n_edges = None

    with open(filepath, 'rb') as f:
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                # EOF before end_header
                break
            line = line_bytes.decode('utf-8').strip()
            if line.startswith('comment'):
                comment_text = line[len('comment '):]
                # parse known comments
                if comment_text.startswith('n segments per edge'):
                    # extract integer after the phrase
                    try:
                        n_segments = int(comment_text.split()[-1])
                    except Exception:
                        pass
                elif comment_text.startswith('n edges in tree'):
                    try:
                        n_edges = int(comment_text.split()[-1])
                    except Exception:
                        pass
            elif line == 'end_header':
                break

    return {'n_segments': n_segments, 'n_edges': n_edges}

"""
This script generates a mesh from a 1D tree structure defined by nodes and edges,
and assigns synthetic flow values to the mesh vertices.
A smoothed version of the mesh is also created by computing joint radii at nodes.
The mesh is exported to a PLY file, which includes metadata about the number of edges and segments.
The mesh is visualized using Polyscope.
"""

# Read grown tree data
nodes = extract_coordinates('ps_demo_data/grown.ipnode')
edges = extract_global_numbers('ps_demo_data/grown.ipelem')
edges = edges - 1  # Adjust for zero-indexing
radius = extract_radius('ps_demo_data/grown_radius.ipfiel')
radii = np.column_stack([radius, radius]) # since we only have one radius per edge, we duplicate it for both ends

# Generate synthetic flow values per edge to map to mesh vertices
flow = np.random.rand(len(edges)) * 5  # Random flow values between 0 and 5 mm3/s

# Generate mesh
verts, faces, data = generate_network_tube_mesh(nodes, edges, radii, data=flow, segments=24) # if no data input, data returned is None

# For smooth tree mesh: Compute joint radii & per-edge radii
joint_radii, edge_radii = compute_joint_radii(nodes, edges, radius)
# Generate mesh with joint radii
n_segments = 10
verts_joint, faces_joint, data_joint = generate_network_tube_mesh(nodes, edges, edge_radii, data=flow, segments=n_segments)

# Gives us an option to read in mesh later so we don't have to generate mesh from scratch every time
export_mesh_to_ply(verts_joint, faces_joint, "ps_demo_data/grown.ply", ascii=True, n_edges=len(edges), n_segments=n_segments,vertex_data=data_joint)
meta = read_ply_header_comments("ps_demo_data/grown.ply")
print(meta) # check that n edges and n segments are read correctly

# Initialize polyscope
ps.init()

# Register your surface mesh
normal = ps.register_surface_mesh("CMGUI-style", verts, faces, smooth_shade=True,enabled=True)
normal.add_scalar_quantity("synthetic flow", data, defined_on='vertices', enabled=True)
smooth = ps.register_surface_mesh("Smooth", verts_joint, faces_joint, smooth_shade=True,enabled=True)
smooth.add_scalar_quantity("synthetic flow", data_joint, defined_on='vertices', enabled=True)
smooth.translate((200,0,0))
# NB: also tested data defined on 'faces', looks the same as if defined on 'vertices' in this case.

# Misc settings
ps.set_ground_plane_mode("none")
ps.set_navigation_style("free")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0])
ps.set_view_projection_mode('orthographic')
ps.show()
