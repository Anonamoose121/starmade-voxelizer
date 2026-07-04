import numpy as np
import struct
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage

from src.voxel_model import VoxelModel


def load_obj(filename):
    vertices = []
    faces = []
    vertex_index_map = {}

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in vertex_index_map:
                    vertex_index_map[key] = len(vertices)
                    vertices.append([x, y, z])
            elif parts[0] == 'f':
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face)

    return vertices, faces


def load_ply(filename):
    vertices = []
    faces = []
    vertex_index_map = {}

    with open(filename, 'r') as f:
        header = []
        for line in f:
            header.append(line.strip())
            if line.strip() == 'end_header':
                break

        vertex_count = 0
        face_count = 0
        is_binary = False

        for line in header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
            elif line.startswith('format'):
                if 'binary' in line:
                    is_binary = True

        if is_binary:
            data = f.buffer.read()
            offset = 0
            for i in range(vertex_count):
                x, y, z = struct.unpack_from('<fff', data, offset)
                offset += 12
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in vertex_index_map:
                    vertex_index_map[key] = len(vertices)
                    vertices.append([x, y, z])

            for i in range(face_count):
                count = struct.unpack_from('<B', data, offset)[0]
                offset += 1
                face = list(struct.unpack_from(f'<{count}I', data, offset))
                offset += 4 * count
                faces.append(face)
        else:
            for i in range(vertex_count):
                parts = f.readline().strip().split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in vertex_index_map:
                    vertex_index_map[key] = len(vertices)
                    vertices.append([x, y, z])

            for i in range(face_count):
                parts = f.readline().strip().split()
                face = [int(p) for p in parts[1:]]
                faces.append(face)

    return vertices, faces


def load_stl(filename):
    vertices = []
    faces = []
    vertex_index_map = {}

    with open(filename, 'rb') as f:
        header = f.read(80)
        is_ascii = header[:5].lower() == b'solid'

    if is_ascii:
        return _load_stl_ascii(filename)
    else:
        return _load_stl_binary(filename)


def _load_stl_ascii(filename):
    vertices = []
    faces = []
    vertex_index_map = {}

    with open(filename, 'r') as f:
        current_face = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'vertex':
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in vertex_index_map:
                    vertex_index_map[key] = len(vertices)
                    vertices.append([x, y, z])
                current_face.append(vertex_index_map[key])
            elif parts[0] == 'endsolid':
                if len(current_face) >= 3:
                    faces.append(current_face)
                current_face = []
            elif parts[0] == 'endfacet':
                if len(current_face) >= 3:
                    faces.append(current_face)
                current_face = []

    return vertices, faces


def _load_stl_binary(filename):
    vertices = []
    faces = []
    vertex_index_map = {}

    with open(filename, 'rb') as f:
        header = f.read(80)
        triangle_count = struct.unpack('<I', f.read(4))[0]

        for i in range(triangle_count):
            normal = struct.unpack('<fff', f.read(12))
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            v3 = struct.unpack('<fff', f.read(12))
            f.read(2)

            tri_indices = []
            for v in (v1, v2, v3):
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key not in vertex_index_map:
                    vertex_index_map[key] = len(vertices)
                    vertices.append([v[0], v[1], v[2]])
                tri_indices.append(vertex_index_map[key])
            faces.append(tri_indices)

    return vertices, faces


def load_vox_from_mesh(filename=None, vertices=None, faces=None, width=None, height=None, depth=None, resolution=32,
                        auto_center=True, rotation_x=0, rotation_y=0, rotation_z=0,
                        hollow=True, thickness=1):
    if vertices is None or faces is None:
        ext = filename.lower().split('.')[-1]
        if ext == 'obj':
            vertices, faces = load_obj(filename)
        elif ext == 'ply':
            vertices, faces = load_ply(filename)
        elif ext == 'stl':
            vertices, faces = load_stl(filename)
        else:
            raise ValueError(f"Unsupported mesh format: {ext}")

    if not vertices or not faces:
        raise ValueError("Mesh has no geometry")

    vertices = np.array(vertices, dtype=float)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    if bbox_size.max() == 0:
        raise ValueError("Mesh has zero size")

    if auto_center:
        center = (bbox_min + bbox_max) / 2.0
        vertices = vertices - center

    for _ in range(rotation_z // 90):
        vertices[:, [0, 1]] = np.column_stack((-vertices[:, 1], vertices[:, 0]))
    for _ in range(rotation_y // 90):
        vertices[:, [0, 2]] = np.column_stack((vertices[:, 2], -vertices[:, 0]))
    for _ in range(rotation_x // 90):
        vertices[:, [1, 2]] = np.column_stack((-vertices[:, 2], vertices[:, 1]))

    new_min = vertices.min(axis=0)
    new_max = vertices.max(axis=0)
    bbox_size = new_max - new_min

    if width is None or height is None or depth is None:
        max_dim = bbox_size.max()
        scale = (resolution - 1) / max_dim
        vertices = vertices * scale
        width = int(np.ceil(vertices[:, 0].max())) + 1
        height = int(np.ceil(vertices[:, 1].max())) + 1
        depth = int(np.ceil(vertices[:, 2].max())) + 1
    else:
        width = max(1, int(width))
        height = max(1, int(height))
        depth = max(1, int(depth))
        target_size = np.array([width - 1, height - 1, depth - 1], dtype=float)
        scale = np.divide(target_size, bbox_size, out=np.ones_like(target_size), where=bbox_size > 0)
        scale = scale.min()
        vertices = vertices * scale

    vertices = vertices - vertices.min(axis=0)
    width = max(1, int(np.ceil(vertices[:, 0].max())) + 1)
    height = max(1, int(np.ceil(vertices[:, 1].max())) + 1)
    depth = max(1, int(np.ceil(vertices[:, 2].max())) + 1)

    if width > 2048 or height > 2048 or depth > 2048 or width * height * depth > 16777216:
        raise ValueError(f"Voxel grid too large: {width}x{height}x{depth}")

    valid_faces = [face[:3] for face in faces if len(face) >= 3]
    if not valid_faces:
        return VoxelModel()

    workers = min(8, max(1, len(valid_faces) // 16))
    voxels = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for face in valid_faces:
            tri = vertices[face[:3]]
            futures.append(executor.submit(_voxelize_triangle_surface, tri, width, height, depth))
        for f in futures:
            voxels.update(f.result())

    model = VoxelModel()
    for (x, y, z) in voxels:
        if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
            model.add_voxel(x, y, z)

    if not hollow and model.voxels:
        model = _fill_interior(model, width, height, depth)

    from src.block_shapes import optimize_blocks
    model = optimize_blocks(model)

    return model


def _clip_triangle_against_z_plane(a, b, c, z_val, keep_below):
    verts = np.array([a, b, c])
    zs = verts[:, 2]
    if keep_below:
        inside = zs <= z_val + 1e-10
    else:
        inside = zs >= z_val - 1e-10
    if inside.all():
        return verts
    if not inside.any():
        return None
    n = 3
    output = []
    for i in range(n):
        j = (i + 1) % n
        pi, pj = verts[i], verts[j]
        zi, zj = pi[2], pj[2]
        inside_i = inside[i]
        inside_j = inside[j]
        if inside_i and inside_j:
            output.append(pj)
        elif inside_i and not inside_j:
            if abs(zj - zi) > 1e-10:
                t = (z_val - zi) / (zj - zi)
                t = max(0.0, min(1.0, t))
                output.append(pi + t * (pj - pi))
        elif not inside_i and inside_j:
            if abs(zj - zi) > 1e-10:
                t = (z_val - zi) / (zj - zi)
                t = max(0.0, min(1.0, t))
                output.append(pi + t * (pj - pi))
            output.append(pj)
    if len(output) < 3:
        return None
    return np.array(output)


def _clip_poly_against_plane(polygon, val, axis, keep_below):
    if polygon is None or len(polygon) < 3:
        return None
    n = len(polygon)
    output = []
    for i in range(n):
        j = (i + 1) % n
        pi, pj = polygon[i], polygon[j]
        vi, vj = pi[axis], pj[axis]
        if keep_below:
            inside_i = vi <= val + 1e-10
            inside_j = vj <= val + 1e-10
        else:
            inside_i = vi >= val - 1e-10
            inside_j = vj >= val - 1e-10
        if inside_i and inside_j:
            output.append(pj)
        elif inside_i and not inside_j:
            if abs(vj - vi) > 1e-10:
                t = (val - vi) / (vj - vi)
                t = max(0.0, min(1.0, t))
                output.append(pi + t * (pj - pi))
        elif not inside_i and inside_j:
            if abs(vj - vi) > 1e-10:
                t = (val - vi) / (vj - vi)
                t = max(0.0, min(1.0, t))
                output.append(pi + t * (pj - pi))
            output.append(pj)
    if len(output) < 3:
        return None
    return np.array(output)


def _clip_tri_to_slab(a, b, c, val0, val1, axis):
    above = _clip_poly_against_plane(np.array([a, b, c]), val0, axis, keep_below=False)
    if above is None or len(above) < 3:
        return None
    clipped = _clip_poly_against_plane(above, val1, axis, keep_below=True)
    if clipped is None or len(clipped) < 3:
        return None
    return clipped


def _polygon_scanline_xs(polygon, y):
    intersections = []
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        yi, yj = polygon[i][1], polygon[j][1]
        xi, xj = polygon[i][0], polygon[j][0]
        if abs(yi - yj) < 1e-10:
            if abs(yi - y) < 1e-10:
                intersections.append(xi)
                intersections.append(xj)
            continue
        if (yi > y) != (yj > y):
            t = (y - yi) / (yj - yi)
            x = xi + t * (xj - xi)
            intersections.append(x)
    return sorted(intersections)


def _voxelize_triangle_surface(tri, width, height, depth):
    voxels = set()
    a, b, c = tri
    x_min, y_min, z_min = tri.min(axis=0)
    x_max, y_max, z_max = tri.max(axis=0)
    ix0 = max(0, int(np.floor(x_min)))
    ix1 = min(width - 1, int(np.floor(x_max)))
    iy0 = max(0, int(np.floor(y_min)))
    iy1 = min(height - 1, int(np.floor(y_max)))
    iz0 = max(0, int(np.floor(z_min)))
    iz1 = min(depth - 1, int(np.floor(z_max)))
    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return voxels

    def _add_vertex_voxels(polygon, fixed_val, fixed_axis):
        for p in polygon:
            if fixed_axis == 0:
                voxels.add((fixed_val, int(np.floor(p[1])), int(np.floor(p[2]))))
            elif fixed_axis == 1:
                voxels.add((int(np.floor(p[0])), fixed_val, int(np.floor(p[2]))))
            else:
                voxels.add((int(np.floor(p[0])), int(np.floor(p[1])), fixed_val))

    # Pass 1: XY projection, scan z-slabs
    for iz in range(iz0, iz1 + 1):
        polygon = _clip_tri_to_slab(a, b, c, float(iz), float(iz + 1), axis=2)
        if polygon is None or len(polygon) < 3:
            continue
        _add_vertex_voxels(polygon, iz, 2)
        poly2d = polygon[:, :2]
        px_min, py_min = poly2d.min(axis=0)
        px_max, py_max = poly2d.max(axis=0)
        jx0 = max(ix0, int(np.floor(px_min)))
        jx1 = min(ix1, int(np.ceil(px_max)) - 1)
        jy0 = max(iy0, int(np.floor(py_min)))
        jy1 = min(iy1, int(np.ceil(py_max)) - 1)
        if jx0 > jx1 or jy0 > jy1:
            continue
        for iy in range(jy0, jy1 + 1):
            xs = _polygon_scanline_xs(poly2d, iy + 0.5)
            if len(xs) < 2:
                continue
            x_start = int(np.floor(xs[0]))
            x_end = int(np.ceil(xs[-1])) - 1
            for jx in range(max(jx0, x_start), min(jx1, x_end) + 1):
                voxels.add((jx, iy, iz))

    # Pass 2: XZ projection, scan y-slabs
    for iy in range(iy0, iy1 + 1):
        polygon = _clip_tri_to_slab(a, b, c, float(iy), float(iy + 1), axis=1)
        if polygon is None or len(polygon) < 3:
            continue
        _add_vertex_voxels(polygon, iy, 1)
        poly2d = polygon[:, [0, 2]]
        px_min, pz_min = poly2d.min(axis=0)
        px_max, pz_max = poly2d.max(axis=0)
        jx0 = max(ix0, int(np.floor(px_min)))
        jx1 = min(ix1, int(np.ceil(px_max)) - 1)
        iz0p = max(iz0, int(np.floor(pz_min)))
        iz1p = min(iz1, int(np.ceil(pz_max)) - 1)
        if jx0 > jx1 or iz0p > iz1p:
            continue
        for iz in range(iz0p, iz1p + 1):
            xs = _polygon_scanline_xs(poly2d, iz + 0.5)
            if len(xs) < 2:
                continue
            x_start = int(np.floor(xs[0]))
            x_end = int(np.ceil(xs[-1])) - 1
            for jx in range(max(jx0, x_start), min(jx1, x_end) + 1):
                voxels.add((jx, iy, iz))

    # Pass 3: YZ projection, scan x-slabs
    for ix in range(ix0, ix1 + 1):
        polygon = _clip_tri_to_slab(a, b, c, float(ix), float(ix + 1), axis=0)
        if polygon is None or len(polygon) < 3:
            continue
        _add_vertex_voxels(polygon, ix, 0)
        poly2d = polygon[:, 1:3]
        py_min, pz_min = poly2d.min(axis=0)
        py_max, pz_max = poly2d.max(axis=0)
        jy0 = max(iy0, int(np.floor(py_min)))
        jy1 = min(iy1, int(np.ceil(py_max)) - 1)
        iz0p = max(iz0, int(np.floor(pz_min)))
        iz1p = min(iz1, int(np.ceil(pz_max)) - 1)
        if jy0 > jy1 or iz0p > iz1p:
            continue
        for iz in range(iz0p, iz1p + 1):
            xs = _polygon_scanline_xs(poly2d, iz + 0.5)
            if len(xs) < 2:
                continue
            x_start = int(np.floor(xs[0]))
            x_end = int(np.ceil(xs[-1])) - 1
            for jy in range(max(jy0, x_start), min(jy1, x_end) + 1):
                voxels.add((ix, jy, iz))

    return voxels


def _flood_fill_exterior(grid, width, height, depth):
    exterior = np.zeros_like(grid)
    from collections import deque
    queue = deque()
    for x in range(width):
        for y in range(height):
            for z in [0, depth - 1]:
                if not grid[x, y, z] and not exterior[x, y, z]:
                    exterior[x, y, z] = True
                    queue.append((x, y, z))
    for x in range(width):
        for z in range(depth):
            for y in [0, height - 1]:
                if not grid[x, y, z] and not exterior[x, y, z]:
                    exterior[x, y, z] = True
                    queue.append((x, y, z))
    for y in range(height):
        for z in range(depth):
            for x in [0, width - 1]:
                if not grid[x, y, z] and not exterior[x, y, z]:
                    exterior[x, y, z] = True
                    queue.append((x, y, z))
    dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in dirs:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                if not grid[nx, ny, nz] and not exterior[nx, ny, nz]:
                    exterior[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
    return exterior


def _fill_interior(model, width, height, depth):
    if not model.voxels:
        return model
    grid = np.zeros((width, height, depth), dtype=bool)
    for (x, y, z) in model.voxels:
        if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
            grid[x, y, z] = True
    exterior = _flood_fill_exterior(grid, width, height, depth)
    interior = ~exterior
    xs, ys, zs = np.where(interior)
    new_model = VoxelModel()
    for i in range(len(xs)):
        new_model.add_voxel(int(xs[i]), int(ys[i]), int(zs[i]))
    return new_model


def _hollow_model(model, thickness=1):
    if not model.voxels:
        return model

    xmin, xmax, ymin, ymax, zmin, zmax = model.get_bounds()
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    depth = zmax - zmin + 1

    grid = np.zeros((width, height, depth), dtype=bool)
    for (x, y, z), info in model.voxels.items():
        gx = x - xmin
        gy = y - ymin
        gz = z - zmin
        if 0 <= gx < width and 0 <= gy < height and 0 <= gz < depth:
            grid[gx, gy, gz] = True

    K = np.ones((thickness + 1, thickness + 1, thickness + 1), dtype=bool)
    interior = ndimage.binary_erosion(grid, structure=K)
    surface = grid & ~interior

    new_model = type(model)()
    new_model.palette = model.palette.copy()
    new_model.core_x = model.core_x
    new_model.core_y = model.core_y
    new_model.core_z = model.core_z
    new_model.next_voxel_id = 1

    xs, ys, zs = surface.nonzero()
    for i in range(len(xs)):
        gx, gy, gz = xs[i], ys[i], zs[i]
        x = gx + xmin
        y = gy + ymin
        z = gz + zmin
        new_model.add_voxel(x, y, z)

    return new_model


def progressive_scan_mesh(vertices, faces, target_resolution=64, width=None, height=None, depth=None,
                           rotation_x=0, rotation_y=0, rotation_z=0, hollow=True, thickness=1):
    vertices = np.array(vertices, dtype=float)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    if bbox_size.max() == 0:
        raise ValueError("Mesh has zero size")

    center = (bbox_min + bbox_max) / 2.0
    vertices = vertices - center

    for _ in range(rotation_z // 90):
        vertices[:, [0, 1]] = np.column_stack((-vertices[:, 1], vertices[:, 0]))
    for _ in range(rotation_y // 90):
        vertices[:, [0, 2]] = np.column_stack((vertices[:, 2], -vertices[:, 0]))
    for _ in range(rotation_x // 90):
        vertices[:, [1, 2]] = np.column_stack((-vertices[:, 2], vertices[:, 1]))

    new_min = vertices.min(axis=0)
    new_max = vertices.max(axis=0)
    bbox_size = new_max - new_min

    if width is None or height is None or depth is None:
        max_dim = bbox_size.max()
        scale = (target_resolution - 1) / max_dim
        vertices = vertices * scale
        width = max(1, int(np.ceil(vertices[:, 0].max())) + 1)
        height = max(1, int(np.ceil(vertices[:, 1].max())) + 1)
        depth = max(1, int(np.ceil(vertices[:, 2].max())) + 1)
    else:
        width = max(1, int(width))
        height = max(1, int(height))
        depth = max(1, int(depth))
        target_size = np.array([width - 1, height - 1, depth - 1], dtype=float)
        scale = np.divide(target_size, bbox_size, out=np.ones_like(target_size), where=bbox_size > 0)
        scale = scale.min()
        vertices = vertices * scale

    vertices = vertices - vertices.min(axis=0)
    width = max(1, int(np.ceil(vertices[:, 0].max())) + 1)
    height = max(1, int(np.ceil(vertices[:, 1].max())) + 1)
    depth = max(1, int(np.ceil(vertices[:, 2].max())) + 1)

    if width > 2048 or height > 2048 or depth > 2048 or width * height * depth > 16777216:
        raise ValueError(f"Voxel grid too large: {width}x{height}x{depth}")

    passes = []
    if target_resolution <= 8:
        passes = [target_resolution]
    elif target_resolution <= 16:
        passes = [8, target_resolution]
    elif target_resolution <= 32:
        passes = [8, 16, target_resolution]
    elif target_resolution <= 64:
        passes = [8, 16, 32, target_resolution]
    else:
        passes = [8, 16, 32, 64, target_resolution] if target_resolution <= 128 else [8, 16, 32, 64, 128, target_resolution]

    base_scale = (target_resolution - 1) / max(width - 1, height - 1, depth - 1) if max(width - 1, height - 1, depth - 1) > 0 else 1.0
    final_vertices = (vertices - vertices.min(axis=0)) * base_scale

    for pass_idx, res in enumerate(passes):
        pass_scale = (res - 1) / max(width - 1, height - 1, depth - 1) if max(width - 1, height - 1, depth - 1) > 0 else 1.0
        pass_vertices = vertices * pass_scale
        pass_width = max(1, int(np.ceil(pass_vertices[:, 0].max())) + 1)
        pass_height = max(1, int(np.ceil(pass_vertices[:, 1].max())) + 1)
        pass_depth = max(1, int(np.ceil(pass_vertices[:, 2].max())) + 1)

        valid_faces = [face[:3] for face in faces if len(face) >= 3]
        workers = min(8, max(1, len(valid_faces) // 16))
        voxels = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for face in valid_faces:
                tri = pass_vertices[face[:3]]
                futures.append(executor.submit(_voxelize_triangle_surface, tri, pass_width, pass_height, pass_depth))
            for f in futures:
                voxels.update(f.result())

        pass_model = VoxelModel()
        for (x, y, z) in voxels:
            if 0 <= x < pass_width and 0 <= y < pass_height and 0 <= z < pass_depth:
                pass_model.add_voxel(x, y, z)

        if not hollow and pass_model.voxels:
            pass_model = _fill_interior(pass_model, pass_width, pass_height, pass_depth)

        from src.block_shapes import optimize_blocks
        pass_model = optimize_blocks(pass_model)

        pass_name = f"Pass {pass_idx + 1}/{len(passes)} ({res} res)"
        yield pass_model, pass_idx, len(passes), pass_name


def scan_mesh(filename, rotation_x=0, rotation_y=0, rotation_z=0):
    ext = filename.lower().split('.')[-1]
    if ext == 'obj':
        vertices, faces = load_obj(filename)
    elif ext == 'ply':
        vertices, faces = load_ply(filename)
    elif ext == 'stl':
        vertices, faces = load_stl(filename)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    if not vertices or not faces:
        raise ValueError("Mesh has no geometry")

    vertices = np.array(vertices, dtype=float)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    if bbox_size.max() == 0:
        raise ValueError("Mesh has zero size")

    center = (bbox_min + bbox_max) / 2.0
    vertices = vertices - center

    for _ in range(rotation_z // 90):
        vertices[:, [0, 1]] = np.column_stack((-vertices[:, 1], vertices[:, 0]))
    for _ in range(rotation_y // 90):
        vertices[:, [0, 2]] = np.column_stack((vertices[:, 2], -vertices[:, 0]))
    for _ in range(rotation_x // 90):
        vertices[:, [1, 2]] = np.column_stack((-vertices[:, 2], vertices[:, 1]))

    new_min = vertices.min(axis=0)
    new_max = vertices.max(axis=0)
    bbox_size = new_max - new_min

    tri_areas = []
    edge_lengths = []
    min_edge = float('inf')
    max_edge = 0.0

    for face in faces:
        if len(face) < 3:
            continue
        tri = vertices[face[:3]]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        tri_areas.append(area)

        for i in range(3):
            a = tri[i]
            b = tri[(i + 1) % 3]
            length = np.linalg.norm(b - a)
            edge_lengths.append(length)
            if length < min_edge:
                min_edge = length
            if length > max_edge:
                max_edge = length

    total_area = sum(tri_areas) if tri_areas else 0.0
    avg_tri_area = total_area / len(tri_areas) if tri_areas else 0.0

    smallest_feature = min_edge if min_edge != float('inf') else 0.0
    largest_feature = max_edge if edge_lengths else 0.0

    if smallest_feature > 0 and bbox_size.max() > 0:
        detail_ratio = smallest_feature / bbox_size.max()
        suggested_resolution = int(np.ceil(1.0 / detail_ratio))
        suggested_resolution = max(4, min(1024, suggested_resolution))
    else:
        suggested_resolution = 32

    return {
        'vertices': len(vertices),
        'faces': len(faces),
        'bbox_min': new_min.tolist(),
        'bbox_max': new_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'total_surface_area': total_area,
        'avg_triangle_area': avg_tri_area,
        'smallest_feature': smallest_feature,
        'largest_feature': largest_feature,
        'suggested_resolution': suggested_resolution,
    }
