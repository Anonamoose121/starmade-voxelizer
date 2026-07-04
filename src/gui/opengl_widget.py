from PyQt6.QtWidgets import QFrame
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QCursor, QMouseEvent, QWheelEvent
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import ctypes


class VoxelGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)

        self.model = None

        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.zoom = 1.0
        self.translation_x = 0.0
        self.translation_y = 0.0

        self.last_mouse_pos = None
        self.is_panning = False

        self.show_grid = True
        self.show_axes = True
        self.show_faces = True
        self.show_wireframe = False
        self.show_mesh_wireframe = False

        self.core_x = 0
        self.core_y = 0
        self.core_z = 0
        self.show_core_indicator = True

        self.front_direction = (0, 0, -1)

        self.mesh_vertices = None
        self.mesh_faces = None

        self.voxel_color_cache = {}

    def initializeGL(self):
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        glViewport(0, 0, w, h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / max(h, 1)
        far_plane = 10000.0
        if self.model:
            xmin, xmax, ymin, ymax, zmin, zmax = self.model.get_bounds()
            model_range = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
            far_plane = max(10000.0, model_range * 20.0)
        gluPerspective(45.0, aspect, 0.1, far_plane)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 0.0, 1.0)
        if self.model:
            xmin, xmax, ymin, ymax, zmin, zmax = self.model.get_bounds()
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            cz = (zmin + zmax) / 2.0
            glTranslatef(-cx, -cy, -cz)
        glTranslatef(self.translation_x, self.translation_y, -50.0 / max(self.zoom, 0.01))

        if self.show_grid:
            glDepthMask(GL_FALSE)
            self._draw_grid()
            glDepthMask(GL_TRUE)

        if self.show_axes:
            glDepthMask(GL_FALSE)
            self._draw_axes()
            glDepthMask(GL_TRUE)

        if self.show_core_indicator:
            glDepthMask(GL_FALSE)
            self._draw_core_indicator()
            glDepthMask(GL_TRUE)

        if self.show_mesh_wireframe:
            glDepthMask(GL_FALSE)
            self._draw_mesh_wireframe()
            glDepthMask(GL_TRUE)

        if self.show_faces and self.model is not None and self.voxel_color_cache:
            self._draw_voxels_immediate()

    def _draw_voxels_immediate(self):
        if self.model is None or not self.model.voxels:
            return
        voxel_set = set(self.model.voxels.keys())
        glBegin(GL_TRIANGLES)
        for (x, y, z), info in self.model.voxels.items():
            r, g, b, a = self.voxel_color_cache.get((x, y, z), (128, 128, 128, 255))
            shape = info.get('shape', 0)
            self._draw_shaped_block(x, y, z, shape, voxel_set, r, g, b, a)
        glEnd()

    def _draw_shaped_block(self, x, y, z, shape, voxel_set, r, g, b, a):
        glColor4f(r, g, b, a)

        def tri_quad(v0, v1, v2, v3):
            glVertex3f(*v0); glVertex3f(*v1); glVertex3f(*v2)
            glVertex3f(*v0); glVertex3f(*v2); glVertex3f(*v3)

        def face_ok(nx, ny, nz):
            return (nx, ny, nz) not in voxel_set

        if shape == 0:
            if face_ok(x, y - 1, z):
                glNormal3f(0.0, -1.0, 0.0)
                tri_quad((x,y,z), (x+1,y,z), (x+1,y,z+1), (x,y,z+1))
            if face_ok(x, y + 1, z):
                glNormal3f(0.0, 1.0, 0.0)
                tri_quad((x,y+1,z+1), (x+1,y+1,z+1), (x+1,y+1,z), (x,y+1,z))
            if face_ok(x, y, z - 1):
                glNormal3f(0.0, 0.0, -1.0)
                tri_quad((x+1,y,z), (x+1,y+1,z), (x,y+1,z), (x,y,z))
            if face_ok(x, y, z + 1):
                glNormal3f(0.0, 0.0, 1.0)
                tri_quad((x,y,z+1), (x,y+1,z+1), (x+1,y+1,z+1), (x+1,y,z+1))
            if face_ok(x - 1, y, z):
                glNormal3f(-1.0, 0.0, 0.0)
                tri_quad((x,y+1,z), (x,y+1,z+1), (x,y,z+1), (x,y,z))
            if face_ok(x + 1, y, z):
                glNormal3f(1.0, 0.0, 0.0)
                tri_quad((x+1,y+1,z+1), (x+1,y+1,z), (x+1,y,z), (x+1,y,z+1))
        elif shape == 1:
            if (x, y, z - 1) not in voxel_set:
                glNormal3f(0.0, 0.0, -1.0)
                tri_quad((x, y, z), (x, y + 0.5, z), (x + 1, y + 0.5, z), (x + 1, y, z))
            if (x, y, z + 1) not in voxel_set:
                glNormal3f(0.0, 0.0, 1.0)
                tri_quad((x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 0.5, z + 1), (x, y + 0.5, z + 1))
            if (x - 1, y, z) not in voxel_set:
                glNormal3f(-1.0, 0.0, 0.0)
                tri_quad((x, y, z + 1), (x, y, z), (x, y + 0.5, z), (x, y + 0.5, z + 1))
            if (x + 1, y, z) not in voxel_set:
                glNormal3f(1.0, 0.0, 0.0)
                tri_quad((x + 1, y, z), (x + 1, y, z + 1), (x + 1, y + 0.5, z + 1), (x + 1, y + 0.5, z))
            glNormal3f(0.0, 1.0, 0.0)
            tri_quad((x, y + 0.5, z + 1), (x, y + 0.5, z), (x + 1, y + 0.5, z), (x + 1, y + 0.5, z + 1))
        elif shape == 2:
            if (x, y, z - 1) not in voxel_set:
                glNormal3f(0.0, 0.0, -1.0)
                tri_quad((x, y + 0.5, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y + 0.5, z))
            if (x, y, z + 1) not in voxel_set:
                glNormal3f(0.0, 0.0, 1.0)
                tri_quad((x + 1, y + 0.5, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1), (x, y + 0.5, z + 1))
            if (x - 1, y, z) not in voxel_set:
                glNormal3f(-1.0, 0.0, 0.0)
                tri_quad((x, y + 0.5, z), (x, y + 0.5, z + 1), (x, y + 1, z + 1), (x, y + 1, z))
            if (x + 1, y, z) not in voxel_set:
                glNormal3f(1.0, 0.0, 0.0)
                tri_quad((x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y + 0.5, z + 1), (x + 1, y + 0.5, z))
            glNormal3f(0.0, -1.0, 0.0)
            tri_quad((x, y + 0.5, z), (x + 1, y + 0.5, z), (x + 1, y + 0.5, z + 1), (x, y + 0.5, z + 1))
        elif shape in (3, 4, 5, 6, 7, 8):
            skip_axis, skip_dir = self._wedge_skip(shape)
            if (skip_axis != 0 or skip_dir != -1) and (x - 1, y, z) not in voxel_set:
                glNormal3f(-1.0, 0.0, 0.0)
                tri_quad((x, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1), (x, y, z))
            if (skip_axis != 0 or skip_dir != 1) and (x + 1, y, z) not in voxel_set:
                glNormal3f(1.0, 0.0, 0.0)
                tri_quad((x + 1, y + 1, z + 1), (x + 1, y + 1, z), (x + 1, y, z), (x + 1, y, z + 1))
            if (skip_axis != 1 or skip_dir != -1) and (x, y - 1, z) not in voxel_set:
                glNormal3f(0.0, -1.0, 0.0)
                tri_quad((x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1))
            if (skip_axis != 1 or skip_dir != 1) and (x, y + 1, z) not in voxel_set:
                glNormal3f(0.0, 1.0, 0.0)
                tri_quad((x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z), (x, y + 1, z))
            if (skip_axis != 2 or skip_dir != -1) and (x, y, z - 1) not in voxel_set:
                glNormal3f(0.0, 0.0, -1.0)
                tri_quad((x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z), (x, y, z))
            if (skip_axis != 2 or skip_dir != 1) and (x, y, z + 1) not in voxel_set:
                glNormal3f(0.0, 0.0, 1.0)
                tri_quad((x, y, z + 1), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y, z + 1))
        elif shape in range(9, 23):
            skip_faces = self._corner_skip_faces(shape)
            face_verts = {
                'bottom': [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)],
                'top': [(x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z), (x, y + 1, z)],
                'front': [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)],
                'back': [(x + 1, y, z), (x, y, z), (x, y + 1, z), (x + 1, y + 1, z)],
                'left': [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)],
                'right': [(x + 1, y, z + 1), (x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1)],
            }
            face_normals = {
                'bottom': (0.0, -1.0, 0.0),
                'top': (0.0, 1.0, 0.0),
                'front': (0.0, 0.0, 1.0),
                'back': (0.0, 0.0, -1.0),
                'left': (-1.0, 0.0, 0.0),
                'right': (1.0, 0.0, 0.0),
            }
            for name, verts in face_verts.items():
                if name in skip_faces:
                    continue
                if (name == 'bottom' and (x, y - 1, z) in voxel_set) or \
                   (name == 'top' and (x, y + 1, z) in voxel_set) or \
                   (name == 'front' and (x, y, z + 1) in voxel_set) or \
                   (name == 'back' and (x, y, z - 1) in voxel_set) or \
                   (name == 'left' and (x - 1, y, z) in voxel_set) or \
                   (name == 'right' and (x + 1, y, z) in voxel_set):
                    continue
                glNormal3f(*face_normals[name])
                tri_quad(*verts)
            from src.block_shapes import _CORNER_DIAGONAL, _CORNER_DIAGONAL_NORMAL
            diag_pts = _CORNER_DIAGONAL.get(shape)
            if diag_pts is not None:
                dn = _CORNER_DIAGONAL_NORMAL.get(shape, (0.0, 0.0, 1.0))
                glNormal3f(dn[0], dn[1], dn[2])
                glVertex3f(x + diag_pts[0][0], y + diag_pts[0][1], z + diag_pts[0][2])
                glVertex3f(x + diag_pts[1][0], y + diag_pts[1][1], z + diag_pts[1][2])
                glVertex3f(x + diag_pts[2][0], y + diag_pts[2][1], z + diag_pts[2][2])
        else:
            if (x, y - 1, z) not in voxel_set:
                glNormal3f(0.0, -1.0, 0.0); glVertex3f(x, y, z); glVertex3f(x + 1, y, z); glVertex3f(x + 1, y, z + 1); glVertex3f(x, y, z + 1)
            if (x, y + 1, z) not in voxel_set:
                glNormal3f(0.0, 1.0, 0.0); glVertex3f(x, y + 1, z + 1); glVertex3f(x + 1, y + 1, z + 1); glVertex3f(x + 1, y + 1, z); glVertex3f(x, y + 1, z)
            if (x, y, z - 1) not in voxel_set:
                glNormal3f(0.0, 0.0, -1.0); glVertex3f(x + 1, y, z); glVertex3f(x + 1, y + 1, z); glVertex3f(x, y + 1, z); glVertex3f(x, y, z)
            if (x, y, z + 1) not in voxel_set:
                glNormal3f(0.0, 0.0, 1.0); glVertex3f(x, y, z + 1); glVertex3f(x, y + 1, z + 1); glVertex3f(x + 1, y + 1, z + 1); glVertex3f(x + 1, y, z + 1)
            if (x - 1, y, z) not in voxel_set:
                glNormal3f(-1.0, 0.0, 0.0); glVertex3f(x, y + 1, z); glVertex3f(x, y + 1, z + 1); glVertex3f(x, y, z + 1); glVertex3f(x, y, z)
            if (x + 1, y, z) not in voxel_set:
                glNormal3f(1.0, 0.0, 0.0); glVertex3f(x + 1, y + 1, z + 1); glVertex3f(x + 1, y + 1, z); glVertex3f(x + 1, y, z); glVertex3f(x + 1, y, z + 1)

    def _wedge_skip(self, shape):
        mapping = {3: (0, 1), 4: (0, -1), 5: (1, 1), 6: (1, -1), 7: (2, 1), 8: (2, -1)}
        return mapping.get(shape, (0, 1))

    def _corner_skip_faces(self, shape):
        from src.block_shapes import _CORNER_SKIP
        return list(_CORNER_SKIP.get(shape, ['bottom', 'front', 'left']))

    def _draw_grid(self):
        glLineWidth(1.0)
        glColor4f(0.4, 0.4, 0.5, 0.4)
        grid_size = 50
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            glVertex3f(i, -grid_size, 0.0); glVertex3f(i, grid_size, 0.0)
            glVertex3f(-grid_size, i, 0.0); glVertex3f(grid_size, i, 0.0)
        glEnd()
        glLineWidth(1.0)

    def _draw_axes(self):
        glLineWidth(2.0)
        xmin, xmax, ymin, ymax, zmin, zmax = (0, 0, 0, 0, 0, 0)
        if self.model:
            xmin, xmax, ymin, ymax, zmin, zmax = self.model.get_bounds()
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin, 1) + 2
        axis_len = max_range / 2 + 2
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0); glVertex3f(axis_len, 0.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, axis_len, 0.0)
        glColor3f(0.3, 0.6, 1.0)
        glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 0.0, axis_len)
        glEnd()
        glLineWidth(1.0)

    def _draw_core_indicator(self):
        cx, cy, cz = float(self.core_x), float(self.core_y), float(self.core_z)
        glLineWidth(2.0)
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(cx - 1.5, cy, cz); glVertex3f(cx + 1.5, cy, cz)
        glVertex3f(cx, cy - 1.5, cz); glVertex3f(cx, cy + 1.5, cz)
        glVertex3f(cx, cy, cz - 1.5); glVertex3f(cx, cy, cz + 1.5)
        glEnd()
        segments = 32
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = cx + 0.5 * np.cos(angle)
            y = cy + 0.5 * np.sin(angle)
            glVertex3f(x, y, cz)
        glEnd()
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = cx + 0.5 * np.cos(angle)
            z = cz + 0.5 * np.sin(angle)
            glVertex3f(x, cy, z)
        glEnd()
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            y = cy + 0.5 * np.cos(angle)
            z = cz + 0.5 * np.sin(angle)
            glVertex3f(cx, y, z)
        glEnd()
        glLineWidth(1.0)

    def _draw_mesh_wireframe(self):
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
        glLineWidth(1.0)
        glColor4f(1.0, 1.0, 1.0, 0.35)
        glBegin(GL_LINES)
        for face in self.mesh_faces:
            if len(face) < 2:
                continue
            for i in range(len(face)):
                a = self.mesh_vertices[face[i]]
                b = self.mesh_vertices[face[(i + 1) % len(face)]]
                glVertex3f(a[0], a[1], a[2])
                glVertex3f(b[0], b[1], b[2])
        glEnd()
        glLineWidth(1.0)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_active = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_active and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.rotation_y += delta.x() * 0.5
            self.rotation_x += delta.y() * 0.5
            self.rotation_x = max(-90.0, min(90.0, self.rotation_x))
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_active = False
            self.last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(50.0, self.zoom))
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.zoom = 1.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.update()

    def set_core_position(self, x, y, z):
        self.core_x = int(x)
        self.core_y = int(y)
        self.core_z = int(z)
        self.update()

    def set_front_direction(self, direction):
        self.front_direction = (
            int(direction[0]),
            int(direction[1]),
            int(direction[2]),
        )
        self.update()

    def get_core_position(self):
        return self.core_x, self.core_y, self.core_z

    def set_model(self, model):
        self.model = model
        self.update_voxel_cache()
        self.updateGeometry()
        try:
            self._auto_zoom()
        except Exception:
            pass
        if model:
            self.core_x = model.core_x
            self.core_y = model.core_y
            self.core_z = model.core_z
            fd = getattr(model, 'front_direction', None)
            if fd is not None:
                self.front_direction = (int(fd[0]), int(fd[1]), int(fd[2]))
        self.update()

    def update_voxel_cache(self):
        self.voxel_color_cache = {}
        if self.model is None:
            return
        for (x, y, z), info in self.model.voxels.items():
            ci = info.get('color', 1)
            color = self.model.palette.get(ci, (128, 128, 128, 255))
            self.voxel_color_cache[(x, y, z)] = color

    def set_mesh_wireframe(self, vertices, faces):
        self.mesh_vertices = np.array(vertices, dtype=float) if vertices is not None else None
        self.mesh_faces = faces if faces is not None else None
        self.update()

    def _auto_zoom(self):
        if self.model is None:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self.model.get_bounds()
        if xmin > xmax:
            return
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if max_range <= 0:
            return
        fov_rad = np.radians(45.0)
        desired_dist = max_range / (2.0 * np.tan(fov_rad / 2.0))
        current_dist = 50.0 / self.zoom if self.zoom > 0 else 50.0
        if current_dist > 0:
            self.zoom = max(0.1, 50.0 / desired_dist)
        else:
            self.zoom = 1.0
