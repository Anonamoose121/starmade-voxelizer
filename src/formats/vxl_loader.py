import struct
from src.voxel_model import VoxelModel


def load_vxl(filename):
    model = VoxelModel()
    with open(filename, 'rb') as f:
        data = f.read()

    if len(data) < 16:
        raise ValueError("Not a valid .vxl file")

    if data[:4] != b'VOX ':
        raise ValueError("Not a valid .vxl file")

    chunk_id, chunk_size, num_children, content_start, offset = _read_chunk(data, 4)
    
    if chunk_id == 'MAIN':
        if chunk_size >= 12:
            model_size_offset = offset
            sz = struct.unpack_from('<III', data, model_size_offset)
        else:
            raise ValueError("Invalid .vxl format")

    offset = content_start
    for i in range(num_children):
        chunk_id, chunk_size, num_children, content_start, offset = _read_chunk(data, offset)
        if chunk_id == 'PACK':
            count = chunk_size // 4
            for j in range(count):
                x, y, z, ci = struct.unpack_from('<BBBB', data, content_start + j * 4)
                if ci != 0:
                    model.add_voxel(x, y, z, ci)
            offset = content_start + chunk_size
        elif chunk_id == 'RGBA':
            for j in range(256):
                r, g, b, a = struct.unpack_from('<BBBB', data, content_start + j * 4)
                model.palette[j + 1] = (r, g, b, a)
            offset = content_start + chunk_size
        elif chunk_id == 'SIZE':
            sz_x, sz_y, sz_z = struct.unpack_from('<III', data, content_start)
            offset = content_start + chunk_size
        elif chunk_id == 'nTRN':
            x, y, z = struct.unpack_from('<iii', data, content_start + 8)
            model.core_x, model.core_y, model.core_z = x, y, z
            offset = content_start + chunk_size
        else:
            offset = content_start + chunk_size

    if model.core_x == 0 and model.core_y == 0 and model.core_z == 0 and model.voxels:
        xmin, xmax, ymin, ymax, zmin, zmax = model.get_bounds()
        model.core_x = (xmax - xmin) // 2
        model.core_y = (ymax - ymin) // 2
        model.core_z = (zmax - zmin) // 2

    return model


def _read_chunk(data, offset):
    if offset + 8 > len(data):
        raise ValueError("Unexpected end of data")
    chunk_id = data[offset:offset + 4].decode('ascii', errors='replace')
    chunk_size, num_children = struct.unpack_from('<II', data, offset + 4)
    return chunk_id, chunk_size, num_children, offset + 12, offset + 12
