import struct
import zlib
from src.voxel_model import VoxelModel


def _read_chunk(data, offset):
    chunk_id = data[offset:offset + 4].decode('ascii', errors='replace')
    offset += 4
    chunk_size, num_children = struct.unpack_from('<II', data, offset)
    offset += 8
    content_start = offset + chunk_size
    return chunk_id, chunk_size, num_children, content_start, offset


def _read_string(data, offset):
    length = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    string = data[offset:offset + length].decode('utf-8', errors='replace')
    return string, offset + length


def load_vox(filename):
    model = VoxelModel()
    with open(filename, 'rb') as f:
        data = f.read()

    if data[:4] != b'VOX ':
        raise ValueError("Not a valid .vox file")

    offset = 4
    _, chunk_size, num_children, content_start, offset = _read_chunk(data, offset)

    if chunk_size == 0:
        return model

    models = {}
    size_keys = {}

    def parse_chunk(offset, depth=0):
        chunk_id, chunk_size, num_children, content_start, offset = _read_chunk(data, offset)
        content_offset = offset

        if chunk_id == 'SIZE':
            sz_x, sz_y, sz_z = struct.unpack_from('<III', data, content_offset)
            size_id = struct.unpack_from('<I', data, content_offset + 12)[0]
            size_keys[size_id] = (sz_x, sz_y, sz_z)

        elif chunk_id == 'XYLI':
            pass

        elif chunk_id == 'RGBA':
            for i in range(256):
                r, g, b, a = struct.unpack_from('<BBBB', data, content_offset + i * 4)
                model.palette[i + 1] = (r, g, b, a)

        elif chunk_id == 'MATL':
            pass

        elif chunk_id == 'nTRN':
            x, y, z = struct.unpack_from('<iii', data, content_offset + 8)
            model.core_x, model.core_y, model.core_z = x, y, z

        elif chunk_id == 'nGRP':
            pass

        elif chunk_id == 'MODL':
            model_id = struct.unpack_from('<I', data, content_offset)[0]
            x, y, z = struct.unpack_from('<iii', data, content_offset + 8)
            size_id = struct.unpack_from_from('<I', data, content_offset + 20)[0]
            pass

        elif chunk_id == 'rOBJ':
            pass

        elif chunk_id == 'rCAM':
            pass

        elif chunk_id == 'NOTE':
            pass

        elif chunk_id == 'IMAP':
            pass

        elif chunk_id == 'LAYR':
            pass

        elif chunk_id == 'PACK':
            count = chunk_size // 4
            for i in range(count):
                x, y, z, ci = struct.unpack_from('<BBBB', data, content_offset + i * 4)
                if ci != 0:
                    model.add_voxel(x, y, z, ci)

        elif chunk_id == 'nSHP':
            children_offset = offset + chunk_size
            num_children = struct.unpack_from('<I', data, children_offset)[0]

        offset = content_start
        if depth < 10:
            for i in range(num_children):
                try:
                    offset = parse_chunk(offset, depth + 1)
                except Exception:
                    break

    for i in range(num_children):
        try:
            offset = parse_chunk(offset)
        except Exception:
            break

    if model.core_x == 0 and model.core_y == 0 and model.core_z == 0 and model.voxels:
        xmin, xmax, ymin, ymax, zmin, zmax = model.get_bounds()
        model.core_x = (xmax - xmin) // 2
        model.core_y = (ymax - ymin) // 2
        model.core_z = (zmax - zmin) // 2

    return model
