import numpy as np
from concurrent.futures import ThreadPoolExecutor


SHAPE_CUBE = 0
SHAPE_SLAB_BOTTOM = 1
SHAPE_SLAB_TOP = 2
SHAPE_WEDGE_PX = 3
SHAPE_WEDGE_NX = 4
SHAPE_WEDGE_PY = 5
SHAPE_WEDGE_NY = 6
SHAPE_WEDGE_PZ = 7
SHAPE_WEDGE_NZ = 8
SHAPE_CORNER_PXPYPZ = 9
SHAPE_CORNER_NXPYPZ = 10
SHAPE_CORNER_PXNYPZ = 11
SHAPE_CORNER_NXNYPZ = 12
SHAPE_CORNER_PXPYNZ = 13
SHAPE_CORNER_NXPYNZ = 14
SHAPE_CORNER_PXNYNZ = 15
SHAPE_CORNER_NXNYNZ = 16
SHAPE_TETRA_PX = 17
SHAPE_TETRA_NX = 18
SHAPE_TETRA_PY = 19
SHAPE_TETRA_NY = 20
SHAPE_TETRA_PZ = 21
SHAPE_TETRA_NZ = 22
SHAPE_COUNT = 23

_FACE_NEIGHBORS = [
    ((1, 0, 0), SHAPE_WEDGE_NX, SHAPE_CORNER_NXPYPZ, SHAPE_CORNER_NXNYPZ, SHAPE_CORNER_NXPYNZ, SHAPE_CORNER_NXNYNZ, SHAPE_WEDGE_PX),
    ((-1, 0, 0), SHAPE_WEDGE_PX, SHAPE_CORNER_PXPYPZ, SHAPE_CORNER_PXNYPZ, SHAPE_CORNER_PXPYNZ, SHAPE_CORNER_PXNYNZ, SHAPE_WEDGE_NX),
    ((0, 1, 0), SHAPE_WEDGE_NY, SHAPE_CORNER_PXPYPZ, SHAPE_CORNER_NXPYPZ, SHAPE_CORNER_PXPYNZ, SHAPE_CORNER_NXPYNZ, SHAPE_WEDGE_PY),
    ((0, -1, 0), SHAPE_WEDGE_PY, SHAPE_CORNER_PXNYPZ, SHAPE_CORNER_NXNYPZ, SHAPE_CORNER_PXNYNZ, SHAPE_CORNER_NXNYNZ, SHAPE_WEDGE_NY),
    ((0, 0, 1), SHAPE_WEDGE_NZ, SHAPE_CORNER_PXPYPZ, SHAPE_CORNER_NXPYPZ, SHAPE_CORNER_PXNYPZ, SHAPE_CORNER_NXNYPZ, SHAPE_WEDGE_PZ),
    ((0, 0, -1), SHAPE_WEDGE_PZ, SHAPE_CORNER_PXPYNZ, SHAPE_CORNER_NXPYNZ, SHAPE_CORNER_PXNYNZ, SHAPE_CORNER_NXNYNZ, SHAPE_WEDGE_NZ),
]

_SLAB_MAP = {
    (True, True, False, False, False, False): SHAPE_SLAB_BOTTOM,
    (False, False, True, True, False, False): SHAPE_SLAB_TOP,
    (True, False, False, False, True, False): SHAPE_SLAB_BOTTOM,
    (False, True, False, False, False, True): SHAPE_SLAB_BOTTOM,
    (False, False, True, False, False, True): SHAPE_SLAB_BOTTOM,
    (False, False, False, True, True, False): SHAPE_SLAB_BOTTOM,
    (True, False, False, False, False, True): SHAPE_SLAB_BOTTOM,
    (False, True, False, False, True, False): SHAPE_SLAB_BOTTOM,
    (False, False, True, True, False, False): SHAPE_SLAB_TOP,
    (False, False, False, True, False, True): SHAPE_SLAB_BOTTOM,
    (True, False, False, True, False, False): SHAPE_SLAB_BOTTOM,
    (False, True, False, False, False, True): SHAPE_SLAB_BOTTOM,
}

_CORNER_MAP = {
    (True, False, True, False, True, False): SHAPE_CORNER_PXPYPZ,
    (True, False, True, False, False, True): SHAPE_CORNER_PXPYNZ,
    (True, False, False, True, True, False): SHAPE_CORNER_PXNYPZ,
    (True, False, False, True, False, True): SHAPE_CORNER_PXNYNZ,
    (False, True, True, False, True, False): SHAPE_CORNER_NXPYPZ,
    (False, True, True, False, False, True): SHAPE_CORNER_NXPYNZ,
    (False, True, False, True, True, False): SHAPE_CORNER_NXNYPZ,
    (False, True, False, True, False, True): SHAPE_CORNER_NXNYNZ,
}

_CORNER_SKIP = {
    SHAPE_CORNER_PXPYPZ:  ('right', 'top', 'front'),
    SHAPE_CORNER_NXPYPZ:  ('left', 'top', 'front'),
    SHAPE_CORNER_PXNYPZ:  ('right', 'bottom', 'front'),
    SHAPE_CORNER_NXNYPZ:  ('left', 'bottom', 'front'),
    SHAPE_CORNER_PXPYNZ:  ('right', 'top', 'back'),
    SHAPE_CORNER_NXPYNZ:  ('left', 'top', 'back'),
    SHAPE_CORNER_PXNYNZ:  ('right', 'bottom', 'back'),
    SHAPE_CORNER_NXNYNZ:  ('left', 'bottom', 'back'),
}

_CORNER_DIAGONAL = {
    SHAPE_CORNER_PXPYPZ:  ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)),
    SHAPE_CORNER_NXPYPZ:  ((0.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)),
    SHAPE_CORNER_PXNYPZ:  ((1.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 1.0)),
    SHAPE_CORNER_NXNYPZ:  ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 1.0)),
    SHAPE_CORNER_PXPYNZ:  ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 0.0)),
    SHAPE_CORNER_NXPYNZ:  ((0.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 0.0)),
    SHAPE_CORNER_PXNYNZ:  ((1.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
    SHAPE_CORNER_NXNYNZ:  ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
}

_CORNER_DIAGONAL_NORMAL = {
    SHAPE_CORNER_PXPYPZ:  (1.0, 1.0, 1.0),
    SHAPE_CORNER_NXPYPZ:  (-1.0, 1.0, 1.0),
    SHAPE_CORNER_PXNYPZ:  (1.0, -1.0, 1.0),
    SHAPE_CORNER_NXNYPZ:  (-1.0, -1.0, 1.0),
    SHAPE_CORNER_PXPYNZ:  (1.0, 1.0, -1.0),
    SHAPE_CORNER_NXPYNZ:  (-1.0, 1.0, -1.0),
    SHAPE_CORNER_PXNYNZ:  (1.0, -1.0, -1.0),
    SHAPE_CORNER_NXNYNZ:  (-1.0, -1.0, -1.0),
}

_WEDGE_DIRS = {
    SHAPE_WEDGE_PX: ('+X', 1, 0, 0),
    SHAPE_WEDGE_NX: ('-X', -1, 0, 0),
    SHAPE_WEDGE_PY: ('+Y', 0, 1, 0),
    SHAPE_WEDGE_NY: ('-Y', 0, -1, 0),
    SHAPE_WEDGE_PZ: ('+Z', 0, 0, 1),
    SHAPE_WEDGE_NZ: ('-Z', 0, 0, -1),
}


def _get_neighbors(voxels, x, y, z):
    nx = (x-1,y,z) in voxels
    px = (x+1,y,z) in voxels
    ny = (x,y-1,z) in voxels
    py = (x,y+1,z) in voxels
    nz = (x,y,z-1) in voxels
    pz = (x,y,z+1) in voxels
    return (nx, px, ny, py, nz, pz)


def _corner_shape_key(neighbors):
    return tuple(neighbors)


def optimize_blocks(model, workers=None):
    if not model.voxels:
        return model
    voxels = set(model.voxels.keys())
    new_voxels = {}
    skip = set()
    for (x, y, z) in voxels:
        if (x, y, z) in skip:
            continue
        neighbors = _get_neighbors(voxels, x, y, z)
        occupied = sum(neighbors)
        shape = None
        if occupied == 6:
            shape = SHAPE_CUBE
        elif occupied == 0:
            shape = SHAPE_CUBE
        elif occupied == 5:
            shape = SHAPE_CUBE
        elif occupied == 4:
            shape = SHAPE_CUBE
        elif occupied == 1:
            shape = SHAPE_CUBE
        elif occupied == 2:
            if neighbors[0] and neighbors[1]:
                shape = SHAPE_CUBE
            elif neighbors[2] and neighbors[3]:
                shape = SHAPE_CUBE
            elif neighbors[4] and neighbors[5]:
                shape = SHAPE_CUBE
            else:
                idx = [i for i, v in enumerate(neighbors) if v]
                if len(idx) == 2:
                    i1, i2 = idx
                    if (i1 in (0, 1) and i2 in (2, 3)) or (i1 in (0, 1) and i2 in (4, 5)) or (i1 in (2, 3) and i2 in (4, 5)):
                        face1 = _FACE_NEIGHBORS[i1]
                        face2 = _FACE_NEIGHBORS[i2]
                        edge1 = face1[6]
                        edge2 = face2[6]
                        if edge1 == edge2:
                            shape = edge1
        elif occupied == 3:
            if sorted([i for i, v in enumerate(neighbors) if v]) in (
                [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
                [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
            ):
                shape = _CORNER_MAP.get(tuple(neighbors), SHAPE_CUBE)
        if shape is None:
            shape = SHAPE_CUBE
        old_info = model.voxels.get((x, y, z), {})
        new_voxels[(x, y, z)] = {
            'color': old_info.get('color', 1),
            'id': old_info.get('id', 1),
            'shape': shape,
            'orientation': 0,
        }
    new_model = type(model)()
    new_model.voxels = new_voxels
    new_model.palette = model.palette.copy()
    new_model.core_x = model.core_x
    new_model.core_y = model.core_y
    new_model.core_z = model.core_z
    new_model.front_direction = getattr(model, 'front_direction', (0, 0, -1))
    new_model.next_voxel_id = model.next_voxel_id
    return new_model


_CUBE_FACES = [
    (0, 1, 2, 3),
    (7, 6, 5, 4),
    (3, 7, 4, 0),
    (6, 2, 1, 5),
    (4, 5, 1, 0),
    (2, 6, 7, 3),
]


def get_block_faces(shape, x, y, z):
    if shape == SHAPE_CUBE:
        verts = [
            (x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
            (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1),
        ]
        face_data = [
            ((0,1,2,3), (0.0,0.0,-1.0)),
            ((7,6,5,4), (0.0,0.0,1.0)),
            ((3,7,4,0), (-1.0,0.0,0.0)),
            ((6,2,1,5), (1.0,0.0,0.0)),
            ((4,5,1,0), (0.0,-1.0,0.0)),
            ((2,6,7,3), (0.0,1.0,0.0)),
        ]
        return verts, face_data
    if shape in (SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP):
        verts = [
            (x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
            (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1),
        ]
        if shape == SHAPE_SLAB_BOTTOM:
            face_data = [
                ((0,1,2,3), (0.0,0.0,-1.0)),
                ((7,6,5,4), (0.0,0.0,1.0)),
                ((3,7,6,2), (-1.0,0.0,0.0)),
                ((0,1,5,4), (1.0,0.0,0.0)),
                ((0,4,7,3), (0.0,-1.0,0.0)),
            ]
            return verts, face_data
        face_data = [
            ((0,1,2,3), (0.0,0.0,-1.0)),
            ((7,6,5,4), (0.0,0.0,1.0)),
            ((1,2,6,5), (-1.0,0.0,0.0)),
            ((0,4,5,1), (1.0,0.0,0.0)),
            ((2,3,7,6), (0.0,1.0,0.0)),
        ]
        return verts, face_data

    if shape in (SHAPE_WEDGE_PX, SHAPE_WEDGE_NX, SHAPE_WEDGE_PY, SHAPE_WEDGE_NY,
                 SHAPE_WEDGE_PZ, SHAPE_WEDGE_NZ):
        return _wedge_faces(shape, x, y, z)

    if shape in range(9, 17):
        return _corner_faces(shape, x, y, z)

    verts = [
        (x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
        (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1),
    ]
    face_data = [
        ((0,1,2,3), (0.0,0.0,-1.0)),
        ((7,6,5,4), (0.0,0.0,1.0)),
        ((3,7,4,0), (-1.0,0.0,0.0)),
        ((6,2,1,5), (1.0,0.0,0.0)),
        ((4,5,1,0), (0.0,-1.0,0.0)),
        ((2,6,7,3), (0.0,1.0,0.0)),
    ]
    return verts, face_data


_CUBE_NORMALS = [
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
]


def _wedge_faces(shape, x, y, z):
    verts = [
        (x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
        (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1),
    ]
    skip_idx = {3: 3, 4: 2, 5: 5, 6: 4, 7: 1, 8: 0}.get(shape, -1)
    face_data = []
    for i in range(len(_CUBE_FACES)):
        if i == skip_idx:
            continue
        face_data.append((_CUBE_FACES[i], _CUBE_NORMALS[i]))
    return verts, face_data


def _corner_faces(shape, x, y, z):
    verts = [
        (x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
        (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1),
    ]
    diag_offsets = _CORNER_DIAGONAL.get(shape, ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)))
    diag_base = 8
    diag_verts = [(x+dx, y+dy, z+dz) for dx, dy, dz in diag_offsets]
    all_verts = verts + diag_verts
    dn = _CORNER_DIAGONAL_NORMAL.get(shape, (0.0, 0.0, 1.0))
    face_data = [
        ((0,1,2,3), (0.0,-1.0,0.0)),
        ((4,5,6,7), (0.0,1.0,0.0)),
        ((1,5,4,0), (0.0,0.0,1.0)),
        ((3,7,6,2), (0.0,0.0,-1.0)),
        ((0,4,7,3), (-1.0,0.0,0.0)),
        ((1,2,6,5), (1.0,0.0,0.0)),
        ((diag_base, diag_base+1, diag_base+2, diag_base), dn),
    ]
    skip = list(_CORNER_SKIP.get(shape, []))
    skip_indices = {
        'bottom': {0},
        'top': {1},
        'front': {2},
        'back': {3},
        'left': {4},
        'right': {5},
        'diagonal': {6},
    }
    remove = set()
    for name in skip:
        remove.update(skip_indices.get(name, set()))
    face_data = [f for i, f in enumerate(face_data) if i not in remove]
    return all_verts, face_data


def _find_block_type(bid):
    try:
        from tools.SMEditClassic.JoFileLibrary.src.jo.sm.data.BlockTypes import BlockTypes
        if BlockTypes.isHull(bid):
            return BlockTypes.getColoredBlock(BlockTypes.HULL_COLOR_GREY_ID, BlockTypes.getColor(bid))
        if BlockTypes.isPowerHull(bid):
            return BlockTypes.POWERHULL_COLOR_GREY
        if BlockTypes.isGlass(bid):
            return BlockTypes.GLASS_WEDGE_ID
    except Exception:
        pass
    return bid
