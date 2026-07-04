import numpy as np


class VoxelModel:
    def __init__(self):
        self.voxels = {}
        self.palette = {}
        self.palette[0] = (0, 0, 0, 0)
        self.core_x = 0
        self.core_y = 0
        self.core_z = 0
        self.front_direction = (0, 0, -1)
        self.next_voxel_id = 1

    def add_voxel(self, x, y, z, color_index=1):
        self.voxels[(x, y, z)] = {
            'color': color_index,
            'id': self.next_voxel_id,
        }
        self.next_voxel_id += 1

    def remove_voxel(self, x, y, z):
        self.voxels.pop((x, y, z), None)

    def get_voxel(self, x, y, z):
        return self.voxels.get((x, y, z))

    def clear(self):
        self.voxels.clear()
        self.next_voxel_id = 1

    def get_bounds(self):
        if not self.voxels:
            return 0, 0, 0, 0, 0, 0
        xs, ys, zs = zip(*self.voxels.keys())
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def get_dimensions(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bounds()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        depth = zmax - zmin + 1
        return width, height, depth

    @classmethod
    def from_array(cls, grid, offset_x=0, offset_y=0, offset_z=0):
        model = cls()
        if hasattr(grid, 'nonzero'):
            xs, ys, zs = grid.nonzero()
        else:
            return model
        n = len(xs)
        if n == 0:
            return model
        ids = range(model.next_voxel_id, model.next_voxel_id + n)
        model.next_voxel_id += n
        model.voxels = {
            (xs[i] + offset_x, ys[i] + offset_y, zs[i] + offset_z): {
                'color': 1,
                'id': ids[i],
            }
            for i in range(n)
        }
        return model
