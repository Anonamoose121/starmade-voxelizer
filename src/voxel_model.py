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

    def remove_floating_blocks(self):
        if not self.voxels:
            return
        cx, cy, cz = self.core_x, self.core_y, self.core_z
        if (cx, cy, cz) not in self.voxels:
            return
        visited = set()
        queue = [(cx, cy, cz)]
        visited.add((cx, cy, cz))
        while queue:
            x, y, z = queue.pop(0)
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (nx, ny, nz) in self.voxels and (nx, ny, nz) not in visited:
                    visited.add((nx, ny, nz))
                    queue.append((nx, ny, nz))
        self.voxels = {k: v for k, v in self.voxels.items() if k in visited}

    def fill_holes(self, max_hole_size=4):
        if not self.voxels:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bounds()
        to_fill = []
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                for z in range(zmin, zmax + 1):
                    if (x, y, z) in self.voxels:
                        continue
                    neighbors = sum(1 for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] if (x+dx, y+dy, z+dz) in self.voxels)
                    if neighbors >= 5:
                        to_fill.append((x, y, z))
        for x, y, z in to_fill:
            self.add_voxel(x, y, z, color_index=1)
        if not to_fill:
            return
        xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = self.get_bounds()
        for x in range(xmin2, xmax2 + 1):
            for y in range(ymin2, ymax2 + 1):
                for z in range(zmin2, zmax2 + 1):
                    if (x, y, z) in self.voxels:
                        continue
                    neighbors = sum(1 for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] if (x+dx, y+dy, z+dz) in self.voxels)
                    if neighbors >= 5:
                        self.add_voxel(x, y, z, color_index=1)

    def cleanup(self):
        self.remove_floating_blocks()
        self.fill_holes()

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
