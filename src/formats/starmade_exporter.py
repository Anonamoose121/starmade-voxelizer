import struct
from src.voxel_model import VoxelModel

_SHAPE_CUBE = 0
_SHAPE_SLAB_BOTTOM = 1
_SHAPE_SLAB_TOP = 2
_SHAPE_WEDGE_PX = 3
_SHAPE_WEDGE_NX = 4
_SHAPE_WEDGE_PY = 5
_SHAPE_WEDGE_NY = 6
_SHAPE_WEDGE_PZ = 7
_SHAPE_WEDGE_NZ = 8
_SHAPE_CORNER_PXPYPZ = 9
_SHAPE_CORNER_NXPYPZ = 10
_SHAPE_CORNER_PXNYPZ = 11
_SHAPE_CORNER_NXNYPZ = 12
_SHAPE_CORNER_PXPYNZ = 13
_SHAPE_CORNER_NXPYNZ = 14
_SHAPE_CORNER_PXNYNZ = 15
_SHAPE_CORNER_NXNYNZ = 16


def _is_int_axis_normal(nx, ny, nz):
    return abs(nx) + abs(ny) + abs(nz) == 1


def export_starmade_obj(model, output_filename):
    xmin, xmax, ymin, ymax, zmin, zmax = model.get_bounds()

    width = xmax - xmin + 1
    height = ymax - ymin + 1
    depth = zmax - zmin + 1

    core_x = model.core_x - xmin
    core_y = model.core_y - ymin
    core_z = model.core_z - zmin

    vertices = []
    normals = []
    faces = []

    vertex_map = {}
    vertex_counter = 0

    def get_or_create_vertex(v):
        nonlocal vertex_counter
        if v not in vertex_map:
            vertex_map[v] = vertex_counter
            vertices.append(v)
            vertex_counter += 1
        return vertex_map[v]

    for (x, y, z), info in model.voxels.items():
        vci = info.get('color', 1)
        shape = info.get('shape', 0)

        base_x = x - xmin
        base_y = y - ymin
        base_z = z - zmin

        from src.block_shapes import get_block_faces, SHAPE_CUBE, SHAPE_SLAB_BOTTOM
        if shape == _SHAPE_CUBE:
            directions = [
                (1, 0, 0, [(base_x+1, base_y, base_z), (base_x+1, base_y+1, base_z), (base_x+1, base_y+1, base_z+1), (base_x+1, base_y, base_z+1)]),
                (-1, 0, 0, [(base_x, base_y+1, base_z), (base_x, base_y, base_z), (base_x, base_y, base_z+1), (base_x, base_y+1, base_z+1)]),
                (0, 1, 0, [(base_x, base_y+1, base_z), (base_x+1, base_y+1, base_z), (base_x+1, base_y+1, base_z+1), (base_x, base_y+1, base_z+1)]),
                (0, -1, 0, [(base_x, base_y, base_z+1), (base_x+1, base_y, base_z+1), (base_x+1, base_y, base_z), (base_x, base_y, base_z)]),
                (0, 0, 1, [(base_x, base_y, base_z+1), (base_x, base_y+1, base_z+1), (base_x+1, base_y+1, base_z+1), (base_x+1, base_y, base_z+1)]),
                (0, 0, -1, [(base_x+1, base_y, base_z), (base_x+1, base_y+1, base_z), (base_x, base_y+1, base_z), (base_x, base_y, base_z)]),
            ]
            for (dx, dy, dz, quad_verts) in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                neighbor = model.get_voxel(nx, ny, nz)
                if neighbor is not None:
                    continue
                nx_val, ny_val, nz_val = float(dx), float(dy), float(dz)
                indices = [get_or_create_vertex(v) + 1 for v in quad_verts]
                faces.append({
                    'indices': indices,
                    'normal': (nx_val, ny_val, nz_val),
                    'color': vci,
                    'plane': (dx, dy, dz),
                })
            continue

        verts, face_data = get_block_faces(shape, base_x, base_y, base_z)
        all_verts = list(verts)
        for face_indices, normal in face_data:
            dx, dy, dz = normal
            if not _is_int_axis_normal(dx, dy, dz):
                indices = [get_or_create_vertex((round(v[0], 6), round(v[1], 6), round(v[2], 6))) + 1 for i, v in enumerate([all_verts[idx] for idx in face_indices])]
                faces.append({
                    'indices': indices,
                    'normal': (round(dx, 6), round(dy, 6), round(dz, 6)),
                    'color': vci,
                    'plane': (0, 0, 0),
                })
                continue
            nx, ny, nz = x + int(round(dx)), y + int(round(dy)), z + int(round(dz))
            if model.get_voxel(nx, ny, nz) is not None:
                continue
            indices = [get_or_create_vertex((round(v[0], 6), round(v[1], 6), round(v[2], 6))) + 1 for i, v in enumerate([all_verts[idx] for idx in face_indices])]
            faces.append({
                'indices': indices,
                'normal': (float(dx), float(dy), float(dz)),
                'color': vci,
                'plane': (int(round(dx)), int(round(dy)), int(round(dz))),
            })

    if not faces:
        with open(output_filename, 'w') as f:
            f.write("# StarMade OBJ - Empty\n")
            f.write(f"o Core\n")
            f.write(f"v {core_x} {core_y} {core_z}\n")
        return

    with open(output_filename, 'w') as f:
        f.write("# StarMade OBJ Export\n")
        f.write(f"o VoxelShip\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        normal_map = {}
        normal_counter = 1
        for face in faces:
            n = face['normal']
            n_key = (round(n[0], 6), round(n[1], 6), round(n[2], 6))
            if n_key not in normal_map:
                normal_map[n_key] = normal_counter
                normal_counter += 1

        for n_key, idx in normal_map.items():
            f.write(f"vn {n_key[0]} {n_key[1]} {n_key[2]}\n")

        for i, face in enumerate(faces):
            n_key = (round(face['normal'][0], 6), round(face['normal'][1], 6), round(face['normal'][2], 6))
            n_idx = normal_map[n_key]
            f.write(f"usemtl material_{face['color']}\n")
            f.write(f"f {face['indices'][0]}//{n_idx} {face['indices'][1]}//{n_idx} {face['indices'][2]}//{n_idx} {face['indices'][3]}//{n_idx}\n")

        f.write(f"\n# Core position: {core_x} {core_y} {core_z}\n")
