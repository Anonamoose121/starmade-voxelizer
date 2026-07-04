import json
import csv
from src.voxel_model import VoxelModel


def load_csv(filename):
    model = VoxelModel()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                x, y, z = int(row[0]), int(row[1]), int(row[2])
                color_index = int(row[3]) if len(row) > 3 else 1
                model.add_voxel(x, y, z, color_index)
    return model


def load_json(filename):
    model = VoxelModel()
    with open(filename, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        for point in data:
            if isinstance(point, dict):
                x, y, z = int(point.get('x', 0)), int(point.get('y', 0)), int(point.get('z', 0))
                color_index = int(point.get('c', 1))
                model.add_voxel(x, y, z, color_index)
            elif isinstance(point, (list, tuple)) and len(point) >= 3:
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                color_index = int(point[3]) if len(point) > 3 else 1
                model.add_voxel(x, y, z, color_index)
    elif isinstance(data, dict):
        voxels = data.get('voxels', data.get('points', []))
        for point in voxels:
            if isinstance(point, dict):
                x, y, z = int(point.get('x', 0)), int(point.get('y', 0)), int(point.get('z', 0))
                color_index = int(point.get('c', 1))
                model.add_voxel(x, y, z, color_index)
            elif isinstance(point, (list, tuple)) and len(point) >= 3:
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                color_index = int(point[3]) if len(point) > 3 else 1
                model.add_voxel(x, y, z, color_index)

    return model
