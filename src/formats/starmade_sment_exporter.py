import os
import struct
import zipfile
import zlib
from io import BytesIO
from src.voxel_model import VoxelModel

_HULL_IDS = [5, 69, 70, 75, 76, 77, 78, 79, 81]
_WEDGE_IDS = [293, 294, 295, 296, 297, 298, 299, 300, 301]
_CORNER_IDS = [302, 303, 304, 305, 306, 307, 308, 309, 310]

_WEDGE_ORI = {3: 0, 4: 4, 5: 1, 6: 5, 7: 2, 8: 6}
_CORNER_ORI = {9: 1, 10: 2, 11: 5, 12: 6, 13: 0, 14: 3, 15: 4, 16: 7}


def _resolve_block(color, shape):
    c = max(0, color)
    if shape == 0:
        idx = c % len(_HULL_IDS)
        return _HULL_IDS[idx], 0
    if shape in _WEDGE_ORI:
        idx = c % len(_WEDGE_IDS)
        return _WEDGE_IDS[idx], _WEDGE_ORI[shape]
    if shape in _CORNER_ORI:
        idx = c % len(_CORNER_IDS)
        return _CORNER_IDS[idx], _CORNER_ORI[shape]
    return color, 0


def export_starmade_sment(model, output_path, entity_type=0, classification=0, blueprint_name=None):
    if not isinstance(model, VoxelModel) or not model.voxels:
        raise ValueError("Model is empty or invalid")

    if blueprint_name is None:
        blueprint_name = os.path.splitext(os.path.basename(output_path))[0] or "Ship"

    xmin, xmax, ymin, ymax, zmin, zmax = model.get_bounds()
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    depth = zmax - zmin + 1

    core_x = model.core_x - xmin
    core_y = model.core_y - ymin
    core_z = model.core_z - zmin

    element_counts = {}
    color_scope = {}
    for voxel in model.voxels.values():
        color = voxel.get('color', 1)
        shape = voxel.get('shape', 0)
        bid, _ = _resolve_block(color, shape)
        element_counts[bid] = element_counts.get(bid, 0) + 1
        color_scope[(bid, shape)] = color_scope.get((bid, shape), 0) + 1

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        bp_dir = f'{blueprint_name}'
        zf.writestr(f'{bp_dir}/header.smbph', _build_header(xmin, ymin, zmin, xmax, ymax, zmax, element_counts, entity_type, classification))
        zf.writestr(f'{bp_dir}/meta.smbpm', _build_meta())
        zf.writestr(f'{bp_dir}/logic.smbpl', _build_logic(core_x, core_y, core_z))
        
        data_dir = f'{bp_dir}/DATA'
        smd3_data = _build_smd3_data(model, xmin, ymin, zmin, xmax, ymax, zmax)
        zf.writestr(f'{data_dir}/{blueprint_name}.0.0.0.smd3', smd3_data)


def _build_header(xmin, ymin, zmin, xmax, ymax, zmax, element_counts, entity_type, classification):
    buf = BytesIO()
    
    version = 5
    buf.write(struct.pack('>i', version))
    
    version_str = b'0.0.0'
    buf.write(struct.pack('>h', len(version_str)))
    buf.write(version_str)
    buf.write(struct.pack('>i', entity_type))
    buf.write(struct.pack('>i', classification))
    buf.write(struct.pack('>fff', float(xmin), float(ymin), float(zmin)))
    buf.write(struct.pack('>fff', float(xmax), float(ymax), float(zmax)))
    buf.write(struct.pack('>i', len(element_counts)))
    for bid, count in element_counts.items():
        buf.write(struct.pack('>h', bid))
        buf.write(struct.pack('>i', count))
    
    buf.write(struct.pack('>B', 0))
    
    return buf.getvalue()


def _build_meta():
    buf = BytesIO()
    
    version = 0
    buf.write(struct.pack('>I', version))
    
    buf.write(struct.pack('>B', 2))
    
    seg_manager_tag = _build_seg_manager_tag()
    buf.write(seg_manager_tag)
    
    buf.write(struct.pack('>B', 1))
    
    return buf.getvalue()


def _build_seg_manager_tag():
    buf = BytesIO()
    buf.write(struct.pack('>B', 1))
    version = 0
    buf.write(struct.pack('>h', version))
    
    buf.write(struct.pack('>B', 13))
    buf.write(struct.pack('>B', 6))
    buf.write(struct.pack('>B', 10))
    buf.write(struct.pack('>i', 1))
    buf.write(struct.pack('>B', 0))
    buf.write(struct.pack('>i', 1))
    buf.write(struct.pack('>B', 0))
    
    buf.write(struct.pack('>B', 0))
    
    return buf.getvalue()


def _build_logic(core_x, core_y, core_z):
    buf = BytesIO()
    
    version = 0
    buf.write(struct.pack('>i', version))
    
    num_controllers = 1
    buf.write(struct.pack('>i', num_controllers))
    
    buf.write(struct.pack('>h', core_x))
    buf.write(struct.pack('>h', core_y))
    buf.write(struct.pack('>h', core_z))
    
    num_groups = 1
    buf.write(struct.pack('>i', num_groups))
    
    buf.write(struct.pack('>h', 1))  # blockId = 1 (ship core)
    buf.write(struct.pack('>i', 1))  # numBlocks
    buf.write(struct.pack('>h', core_x))
    buf.write(struct.pack('>h', core_y))
    buf.write(struct.pack('>h', core_z))
    
    return buf.getvalue()


def _build_smd3_data(model, xmin, ymin, zmin, xmax, ymax, zmax):
    segments = {}
    
    for (gx, gy, gz), info in model.voxels.items():
        x = gx - xmin
        y = gy - ymin
        z = gz - zmin
        
        sx = x // 32
        sy = y // 32
        sz = z // 32
        
        seg_key = (sx, sy, sz)
        if seg_key not in segments:
            segments[seg_key] = {}
        
        local_x = x % 32
        local_y = y % 32
        local_z = z % 32
        
        linear_index = local_z * 32 * 32 + local_y * 32 + local_x
        color = info.get('color', 1)
        shape = info.get('shape', 0)
        bid, orientation = _resolve_block(color, shape)
        is_active = 0
        hitpoints = 255
        
        segments[seg_key][linear_index] = {
            'blockId': bid,
            'orientation': orientation,
            'isActive': is_active,
            'hitpoints': hitpoints,
        }
    
    min_sx = min(k[0] for k in segments.keys())
    max_sx = max(k[0] for k in segments.keys())
    min_sy = min(k[1] for k in segments.keys())
    max_sy = max(k[1] for k in segments.keys())
    min_sz = min(k[2] for k in segments.keys())
    max_sz = max(k[2] for k in segments.keys())
    
    header = BytesIO()
    header.write(struct.pack('>i', 0))
    
    indices = {key: i for i, key in enumerate(sorted(segments.keys()))}
    
    for sz in range(0, 16):
        for sy in range(0, 16):
            for sx in range(0, 16):
                seg_key = (sx, sy, sz)
                sid = indices.get(seg_key, -1)
                header.write(struct.pack('>h', sid))
                header.write(struct.pack('>h', 0))
    
    segment_data_list = []
    for seg_key in sorted(segments.keys()):
        sx, sy, sz = seg_key
        
        block_array = bytearray(32 * 32 * 32 * 3)
        for local_linear, block_info in segments[seg_key].items():
            offset = local_linear * 3
            block_data = (block_info['orientation'] & 0x7) << 21
            block_data |= (block_info['isActive'] & 0x1) << 20
            block_data |= (block_info['hitpoints'] & 0x1FF) << 11
            block_data |= (block_info['blockId'] & 0x7FF)
            block_array[offset] = (block_data >> 16) & 0xFF
            block_array[offset + 1] = (block_data >> 8) & 0xFF
            block_array[offset + 2] = block_data & 0xFF
        
        compressed = zlib.compress(bytes(block_array), 9)
        
        seg_buf = BytesIO()
        seg_buf.write(struct.pack('>B', 1))
        seg_buf.write(struct.pack('>q', 0))
        seg_buf.write(struct.pack('>iii', sx * 32 + xmin, sy * 32 + ymin, sz * 32 + zmin))
        seg_buf.write(struct.pack('>B', 1))
        seg_buf.write(struct.pack('>i', len(compressed)))
        seg_buf.write(compressed)
        
        seg_data = seg_buf.getvalue()
        padding = 49152 - len(seg_data)
        if padding > 0:
            seg_buf.write(b'\x00' * padding)
        
        segment_data_list.append(seg_buf.getvalue())
    
    return header.getvalue() + b''.join(segment_data_list)
