from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QFileDialog, QGroupBox,
    QCheckBox, QMessageBox, QScrollArea, QSplitter, QComboBox, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QCloseEvent
import sys
import os
import zipfile
import struct
import zlib
import numpy as np
from io import BytesIO

from src.formats.vox_loader import load_vox
from src.formats.vxl_loader import load_vxl
from src.formats.mesh_loader import load_vox_from_mesh, scan_mesh, progressive_scan_mesh, load_obj, load_ply, load_stl
from src.formats.text_loader import load_csv, load_json
from src.formats.starmade_exporter import export_starmade_obj
from src.formats.starmade_sment_exporter import export_starmade_sment, export_starmade_dir
from src.gui.opengl_widget import VoxelGLWidget


class LoaderThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filename, hollow=True, thickness=1):
        super().__init__()
        self.filename = filename
        self.hollow = hollow
        self.thickness = thickness

    def run(self):
        try:
            ext = os.path.splitext(self.filename)[1].lower()
            if ext == '.vox':
                model = load_vox(self.filename)
            elif ext == '.vxl':
                model = load_vxl(self.filename)
            elif ext in ['.obj', '.ply', '.stl']:
                model = load_vox_from_mesh(
                    self.filename,
                    resolution=32,
                    hollow=self.hollow,
                    thickness=self.thickness,
                )
            elif ext == '.csv':
                model = load_csv(self.filename)
            elif ext == '.json':
                model = load_json(self.filename)
            else:
                raise ValueError(f"Unsupported format: {ext}")
            self.finished.emit(model)
        except Exception as e:
            self.error.emit(str(e))


class MeshImportThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filename, resolution, width, height, depth, rotation_x, rotation_y, rotation_z, hollow=True, thickness=1, vertices=None, faces=None):
        super().__init__()
        self.filename = filename
        self.resolution = resolution
        self.width = width
        self.height = height
        self.depth = depth
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z
        self.hollow = hollow
        self.thickness = thickness
        self.vertices = vertices
        self.faces = faces

    def run(self):
        try:
            if self.filename == "live_preview" and self.vertices is not None and self.faces is not None:
                model = load_vox_from_mesh(
                    vertices=self.vertices,
                    faces=self.faces,
                    width=self.width,
                    height=self.height,
                    depth=self.depth,
                    resolution=self.resolution,
                    rotation_x=self.rotation_x,
                    rotation_y=self.rotation_y,
                    rotation_z=self.rotation_z,
                    hollow=self.hollow,
                    thickness=self.thickness,
                )
            else:
                model = load_vox_from_mesh(
                    self.filename,
                    width=self.width,
                    height=self.height,
                    depth=self.depth,
                    resolution=self.resolution,
                    rotation_x=self.rotation_x,
                    rotation_y=self.rotation_y,
                    rotation_z=self.rotation_z,
                    hollow=self.hollow,
                    thickness=self.thickness,
                )
            self.finished.emit(model)
        except Exception as e:
            self.error.emit(str(e))


class ScanThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filename, rotation_x, rotation_y, rotation_z):
        super().__init__()
        self.filename = filename
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z

    def run(self):
        try:
            result = scan_mesh(
                self.filename,
                rotation_x=self.rotation_x,
                rotation_y=self.rotation_y,
                rotation_z=self.rotation_z,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ProgressiveScanThread(QThread):
    progress = pyqtSignal(object, int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filename, resolution, width, height, depth, rotation_x, rotation_y, rotation_z, hollow=True, thickness=1):
        super().__init__()
        self.filename = filename
        self.resolution = resolution
        self.width = width
        self.height = height
        self.depth = depth
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z
        self.hollow = hollow
        self.thickness = thickness
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            ext = os.path.splitext(self.filename)[1].lower()
            if ext == '.obj':
                vertices, faces = load_obj(self.filename)
            elif ext == '.ply':
                vertices, faces = load_ply(self.filename)
            elif ext == '.stl':
                vertices, faces = load_stl(self.filename)
            else:
                raise ValueError(f"Unsupported mesh format: {ext}")

            if not vertices or not faces:
                raise ValueError("Mesh has no geometry")

            final_model = None
            for model, pass_idx, total_passes, pass_name in progressive_scan_mesh(
                vertices, faces,
                target_resolution=self.resolution,
                width=self.width,
                height=self.height,
                depth=self.depth,
                rotation_x=self.rotation_x,
                rotation_y=self.rotation_y,
                rotation_z=self.rotation_z,
                hollow=self.hollow,
                thickness=self.thickness,
            ):
                if self._abort:
                    return
                final_model = model
                self.progress.emit(model, pass_idx, total_passes, pass_name)

            if final_model is not None:
                self.finished.emit(final_model)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StarMade Voxelizer")
        self.setGeometry(100, 100, 1200, 800)

        self.model = None
        self._base_model = None
        self.current_file = None
        self.last_export_path = None
        self.scan_thread = None
        self.mesh_vertices = None
        self.mesh_faces = None
        self.live_preview_timer = QTimer()
        self.live_preview_timer.setSingleShot(True)
        self.live_preview_timer.timeout.connect(self._do_live_preview)

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left_panel = QWidget()
        left_layout_panel = QVBoxLayout(left_panel)
        left_layout_panel.setContentsMargins(8, 8, 8, 8)

        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Voxel Model")
        self.load_btn.clicked.connect(self._load_file)
        file_layout.addWidget(self.load_btn)

        self.import_mesh_btn = QPushButton("Import 3D Model (.obj / .ply / .stl)")
        self.import_mesh_btn.clicked.connect(self._import_mesh)
        file_layout.addWidget(self.import_mesh_btn)

        self.scan_btn = QPushButton("Scan 3D Model Hull")
        self.scan_btn.clicked.connect(self._scan_mesh)
        file_layout.addWidget(self.scan_btn)

        self.progressive_scan_btn = QPushButton("Progressive Scan")
        self.progressive_scan_btn.clicked.connect(self._progressive_scan)
        file_layout.addWidget(self.progressive_scan_btn)

        self.scan_result_label = QLabel("Scan: no file selected")
        self.scan_result_label.setWordWrap(True)
        self.scan_result_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.scan_result_label)

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Max Res:"))
        self.max_res_spin = QSpinBox()
        self.max_res_spin.setRange(4, 1024)
        self.max_res_spin.setValue(64)
        self.max_res_spin.setSingleStep(4)
        res_layout.addWidget(self.max_res_spin)
        file_layout.addLayout(res_layout)

        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("W (m):"))
        self.dim_w_spin = QSpinBox()
        self.dim_w_spin.setRange(0, 2048)
        self.dim_w_spin.setValue(0)
        dim_layout.addWidget(self.dim_w_spin)

        dim_layout.addWidget(QLabel("H (m):"))
        self.dim_h_spin = QSpinBox()
        self.dim_h_spin.setRange(0, 2048)
        self.dim_h_spin.setValue(0)
        dim_layout.addWidget(self.dim_h_spin)

        dim_layout.addWidget(QLabel("D (m):"))
        self.dim_d_spin = QSpinBox()
        self.dim_d_spin.setRange(0, 2048)
        self.dim_d_spin.setValue(0)
        dim_layout.addWidget(self.dim_d_spin)

        self.use_dims_chk = QCheckBox("Use exact dims")
        self.use_dims_chk.setChecked(False)
        dim_layout.addWidget(self.use_dims_chk)

        self.auto_rescan_chk = QCheckBox("Auto rescan on dim change")
        self.auto_rescan_chk.setChecked(False)
        dim_layout.addWidget(self.auto_rescan_chk)

        dim_layout.setStretch(0, 0)
        dim_layout.setStretch(1, 1)
        dim_layout.setStretch(2, 0)
        dim_layout.setStretch(3, 1)
        dim_layout.setStretch(4, 0)
        dim_layout.setStretch(5, 1)
        dim_layout.setStretch(6, 0)
        dim_layout.setStretch(7, 1)

        hollow_layout = QHBoxLayout()
        self.hollow_chk = QCheckBox("Hull only")
        self.hollow_chk.setChecked(True)
        hollow_layout.addWidget(self.hollow_chk)

        hollow_layout.addWidget(QLabel("Thickness:"))
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 8)
        self.thickness_spin.setValue(1)
        hollow_layout.addWidget(self.thickness_spin)

        self.cleanup_chk = QCheckBox("Cleanup")
        self.cleanup_chk.setChecked(True)
        hollow_layout.addWidget(self.cleanup_chk)

        file_layout.addLayout(dim_layout)
        file_layout.addLayout(hollow_layout)

        self.max_res_spin.valueChanged.connect(self._on_live_setting_changed)
        self.dim_w_spin.valueChanged.connect(self._on_live_setting_changed)
        self.dim_h_spin.valueChanged.connect(self._on_live_setting_changed)
        self.dim_d_spin.valueChanged.connect(self._on_live_setting_changed)
        self.use_dims_chk.toggled.connect(self._on_live_setting_changed)
        self.hollow_chk.toggled.connect(self._on_live_setting_changed)
        self.thickness_spin.valueChanged.connect(self._on_live_setting_changed)
        self.cleanup_chk.toggled.connect(self._on_live_setting_changed)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        self.voxel_count_label = QLabel("Voxels: 0")
        file_layout.addWidget(self.voxel_count_label)

        self.dims_label = QLabel("Dimensions: 0 x 0 x 0 m")
        file_layout.addWidget(self.dims_label)

        file_group.setLayout(file_layout)
        left_layout_panel.addWidget(file_group)

        core_group = QGroupBox("Core Position")
        core_layout = QVBoxLayout()

        core_input_layout = QHBoxLayout()
        core_input_layout.addWidget(QLabel("X:"))
        self.core_x_spin = QSpinBox()
        self.core_x_spin.setRange(-10000, 10000)
        self.core_x_spin.valueChanged.connect(self._on_core_changed)
        core_input_layout.addWidget(self.core_x_spin)

        core_input_layout.addWidget(QLabel("Y:"))
        self.core_y_spin = QSpinBox()
        self.core_y_spin.setRange(-10000, 10000)
        self.core_y_spin.valueChanged.connect(self._on_core_changed)
        core_input_layout.addWidget(self.core_y_spin)

        core_input_layout.addWidget(QLabel("Z:"))
        self.core_z_spin = QSpinBox()
        self.core_z_spin.setRange(-10000, 10000)
        self.core_z_spin.valueChanged.connect(self._on_core_changed)
        core_input_layout.addWidget(self.core_z_spin)

        core_layout.addLayout(core_input_layout)

        front_layout = QHBoxLayout()
        front_layout.addWidget(QLabel("Front:"))
        self.front_dir_combo = QComboBox()
        self.front_dir_combo.addItems(["-X", "+X", "-Y", "+Y", "-Z", "+Z"])
        self.front_dir_combo.setCurrentText("-Z")
        self.front_dir_combo.currentTextChanged.connect(self._on_front_changed)
        front_layout.addWidget(self.front_dir_combo)
        core_layout.addLayout(front_layout)

        core_btn_layout = QHBoxLayout()
        self.center_core_btn = QPushButton("Center Core")
        self.center_core_btn.clicked.connect(self._center_core)
        core_btn_layout.addWidget(self.center_core_btn)

        self.reset_core_btn = QPushButton("Reset Core")
        self.reset_core_btn.clicked.connect(self._reset_core)
        core_btn_layout.addWidget(self.reset_core_btn)

        core_layout.addLayout(core_btn_layout)
        core_group.setLayout(core_layout)
        left_layout_panel.addWidget(core_group)

        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout()

        self.flip_x_btn = QPushButton("Flip X (Left/Right)")
        self.flip_x_btn.clicked.connect(self._flip_x)
        tools_layout.addWidget(self.flip_x_btn)

        self.flip_y_btn = QPushButton("Flip Y (Forward/Back)")
        self.flip_y_btn.clicked.connect(self._flip_y)
        tools_layout.addWidget(self.flip_y_btn)

        self.flip_z_btn = QPushButton("Flip Z (Up/Down)")
        self.flip_z_btn.clicked.connect(self._flip_z)
        tools_layout.addWidget(self.flip_z_btn)

        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(QLabel("Rotate X:"))
        self.rot_x_spin = QSpinBox()
        self.rot_x_spin.setRange(0, 270)
        self.rot_x_spin.setValue(0)
        self.rot_x_spin.setSingleStep(90)
        self.rot_x_spin.valueChanged.connect(self._on_live_setting_changed)
        rotate_layout.addWidget(self.rot_x_spin)

        rotate_layout.addWidget(QLabel("Y:"))
        self.rot_y_spin = QSpinBox()
        self.rot_y_spin.setRange(0, 270)
        self.rot_y_spin.setValue(0)
        self.rot_y_spin.setSingleStep(90)
        self.rot_y_spin.valueChanged.connect(self._on_live_setting_changed)
        rotate_layout.addWidget(self.rot_y_spin)

        rotate_layout.addWidget(QLabel("Z:"))
        self.rot_z_spin = QSpinBox()
        self.rot_z_spin.setRange(0, 270)
        self.rot_z_spin.setValue(0)
        self.rot_z_spin.setSingleStep(90)
        self.rot_z_spin.valueChanged.connect(self._on_live_setting_changed)
        rotate_layout.addWidget(self.rot_z_spin)

        tools_layout.addLayout(rotate_layout)

        self.apply_rot_btn = QPushButton("Apply Rotation to Model")
        self.apply_rot_btn.clicked.connect(self._apply_rotation)
        tools_layout.addWidget(self.apply_rot_btn)

        self.clear_btn = QPushButton("Clear Model")
        self.clear_btn.clicked.connect(self._clear_model)
        tools_layout.addWidget(self.clear_btn)

        tools_group.setLayout(tools_layout)
        left_layout_panel.addWidget(tools_group)

        view_group = QGroupBox("View Settings")
        view_layout = QVBoxLayout()

        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.toggled.connect(self._on_view_changed)
        view_layout.addWidget(self.show_grid_cb)

        self.show_axes_cb = QCheckBox("Show Axes")
        self.show_axes_cb.setChecked(True)
        self.show_axes_cb.toggled.connect(self._on_view_changed)
        view_layout.addWidget(self.show_axes_cb)

        self.show_core_cb = QCheckBox("Show Core Indicator")
        self.show_core_cb.setChecked(True)
        self.show_core_cb.toggled.connect(self._on_view_changed)
        view_layout.addWidget(self.show_core_cb)

        self.show_faces_cb = QCheckBox("Show Faces")
        self.show_faces_cb.setChecked(True)
        self.show_faces_cb.toggled.connect(self._on_view_changed)
        view_layout.addWidget(self.show_faces_cb)

        self.show_wireframe_cb = QCheckBox("Show Wireframe")
        self.show_wireframe_cb.toggled.connect(self._on_view_changed)
        view_layout.addWidget(self.show_wireframe_cb)

        self.show_mesh_wireframe_cb = QCheckBox("Show Original Mesh Wireframe")
        self.show_mesh_wireframe_cb.setChecked(False)
        self.show_mesh_wireframe_cb.toggled.connect(self._on_view_changed)
        view_layout.addWidget(self.show_mesh_wireframe_cb)

        self.live_preview_cb = QCheckBox("Live Preview")
        self.live_preview_cb.setChecked(False)
        self.live_preview_cb.setToolTip("Re-voxelize automatically when settings change")
        view_layout.addWidget(self.live_preview_cb)

        self.smooth_cb = QCheckBox("Smooth Blocks")
        self.smooth_cb.setChecked(False)
        self.smooth_cb.setToolTip("Use slabs, wedges, corners to smooth the hull")
        self.smooth_cb.toggled.connect(self._on_smooth_toggled)
        view_layout.addWidget(self.smooth_cb)

        view_group.setLayout(view_layout)
        left_layout_panel.addWidget(view_group)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        self.export_obj_btn = QPushButton("Export StarMade OBJ")
        self.export_obj_btn.clicked.connect(self._export_obj)
        export_layout.addWidget(self.export_obj_btn)

        self.export_sment_btn = QPushButton("Export StarMade Blueprint (.sment)")
        self.export_sment_btn.clicked.connect(self._export_sment)
        export_layout.addWidget(self.export_sment_btn)

        self.export_dir_btn = QPushButton("Export to Blueprints Folder")
        self.export_dir_btn.clicked.connect(self._export_to_blueprints_dir)
        export_layout.addWidget(self.export_dir_btn)

        self.export_label = QLabel("")
        export_layout.addWidget(self.export_label)

        export_group.setLayout(export_layout)
        left_layout_panel.addWidget(export_group)

        left_layout_panel.addStretch()

        left_scroll.setWidget(left_panel)
        left_widget.layout().addWidget(left_scroll)

        self.gl_widget = VoxelGLWidget()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.gl_widget)

        main_layout.addWidget(splitter)

    def _scan_mesh(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Scan 3D Model Hull", "",
            "3D Models (*.obj *.ply *.stl);;All Files (*.*)"
        )
        if not filename:
            return

        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()

        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        self.scan_result_label.setText("Scanning hull geometry...")
        self.scan_result_label.setStyleSheet("color: orange;")

        self.scan_thread = ScanThread(filename, rot_x, rot_y, rot_z)
        self.scan_thread.finished.connect(self._on_scan_complete)
        self.scan_thread.error.connect(self._on_scan_error)
        self.scan_thread.start()

    def _on_scan_complete(self, result):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan 3D Model Hull")

        w = int(result['bbox_size'][0])
        h = int(result['bbox_size'][1])
        d = int(result['bbox_size'][2])

        suggested = result['suggested_resolution']
        self.max_res_spin.setValue(suggested)

        if self.use_dims_chk.isChecked():
            self.dim_w_spin.setValue(max(1, w))
            self.dim_h_spin.setValue(max(1, h))
            self.dim_d_spin.setValue(max(1, d))

        text = (
            f"Vertices: {result['vertices']} | Triangles: {result['faces']}\n"
            f"BBox: {w} x {h} x {d}\n"
            f"Min feature: {result['smallest_feature']:.3f} | Max feature: {result['largest_feature']:.3f}\n"
            f"Suggested resolution: {suggested}"
        )
        self.scan_result_label.setText(text)
        self.scan_result_label.setStyleSheet("color: green;")

        if self.scan_thread and hasattr(self.scan_thread, 'filename'):
            self._load_mesh_for_wireframe(self.scan_thread.filename)

        if self.live_preview_cb.isChecked():
            self._schedule_live_preview()

    def _load_mesh_for_wireframe(self, filename):
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.obj':
                vertices, faces = load_obj(filename)
            elif ext == '.ply':
                vertices, faces = load_ply(filename)
            elif ext == '.stl':
                vertices, faces = load_stl(filename)
            else:
                return

            vertices = np.array(vertices, dtype=float)
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            center = (bbox_min + bbox_max) / 2.0
            vertices = vertices - center

            rot_x = self.rot_x_spin.value()
            rot_y = self.rot_y_spin.value()
            rot_z = self.rot_z_spin.value()
            for _ in range(rot_z // 90):
                vertices[:, [0, 1]] = np.column_stack((-vertices[:, 1], vertices[:, 0]))
            for _ in range(rot_y // 90):
                vertices[:, [0, 2]] = np.column_stack((vertices[:, 2], -vertices[:, 0]))
            for _ in range(rot_x // 90):
                vertices[:, [1, 2]] = np.column_stack((-vertices[:, 2], vertices[:, 1]))

            resolution = self.max_res_spin.value()
            if self.use_dims_chk.isChecked():
                width = self.dim_w_spin.value() or None
                height = self.dim_h_spin.value() or None
                depth = self.dim_d_spin.value() or None
            else:
                width = None
                height = None
                depth = None

            bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
            if width is None or height is None or depth is None:
                max_dim = bbox_size.max()
                scale = (resolution - 1) / max_dim if max_dim > 0 else 1.0
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

            self.mesh_vertices = vertices
            self.mesh_faces = faces
            self.gl_widget.set_mesh_wireframe(self.mesh_vertices, self.mesh_faces)
        except Exception:
            pass

    def _on_scan_error(self, error_msg):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan 3D Model Hull")
        self.scan_result_label.setText(f"Scan failed: {error_msg}")
        self.scan_result_label.setStyleSheet("color: red;")

    def _progressive_scan(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Progressive Scan", "",
            "3D Models (*.obj *.ply *.stl);;All Files (*.*)"
        )
        if not filename:
            return

        resolution = self.max_res_spin.value()
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        hollow = self.hollow_chk.isChecked()
        thickness = self.thickness_spin.value()

        if self.use_dims_chk.isChecked():
            width = self.dim_w_spin.value() or None
            height = self.dim_h_spin.value() or None
            depth = self.dim_d_spin.value() or None
        else:
            width = None
            height = None
            depth = None

        self.current_file = filename
        self.scan_btn.setEnabled(False)
        self.progressive_scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        self.progressive_scan_btn.setText("Scanning...")
        self.scan_result_label.setText("Starting progressive scan...")
        self.scan_result_label.setStyleSheet("color: orange;")

        self.prog_thread = ProgressiveScanThread(filename, resolution, width, height, depth, rot_x, rot_y, rot_z, hollow, thickness)
        self.prog_thread.progress.connect(self._on_progressive_progress)
        self.prog_thread.finished.connect(self._on_progressive_complete)
        self.prog_thread.error.connect(self._on_progressive_error)
        self.prog_thread.start()

    def _on_progressive_progress(self, model, pass_idx, total_passes, pass_name):
        self.model = model
        fd = getattr(model, 'front_direction', (0, 0, -1))
        mapping = {
            (0, 0, -1): '-Z', (0, 0, 1): '+Z',
            (0, -1, 0): '-Y', (0, 1, 0): '+Y',
            (-1, 0, 0): '-X', (1, 0, 0): '+X',
        }
        self.front_dir_combo.blockSignals(True)
        self.front_dir_combo.setCurrentText(mapping.get(fd, '-Z'))
        self.front_dir_combo.blockSignals(False)
        self.gl_widget.set_model(model)
        self._load_mesh_for_wireframe(self.current_file)

        x1, x2, y1, y2, z1, z2 = model.get_bounds()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        d = z2 - z1 + 1

        self.voxel_count_label.setText(f"Voxels: {len(model.voxels)} ({pass_name})")
        self.dims_label.setText(f"Dimensions: {w} x {h} x {d} m")
        self.scan_result_label.setText(f"{pass_name}")
        self.scan_result_label.setStyleSheet("color: orange;")

        self.gl_widget.update()

    def _on_progressive_complete(self, model, store_base=True):
        self.model = model
        if store_base:
            self._base_model = model
        fd = getattr(model, 'front_direction', (0, 0, -1))
        mapping = {
            (0, 0, -1): '-Z', (0, 0, 1): '+Z',
            (0, -1, 0): '-Y', (0, 1, 0): '+Y',
            (-1, 0, 0): '-X', (1, 0, 0): '+X',
        }
        self.front_dir_combo.blockSignals(True)
        self.front_dir_combo.setCurrentText(mapping.get(fd, '-Z'))
        self.front_dir_combo.blockSignals(False)
        self.gl_widget.set_model(model)
        self._load_mesh_for_wireframe(self.current_file)

        x1, x2, y1, y2, z1, z2 = model.get_bounds()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        d = z2 - z1 + 1

        self.voxel_count_label.setText(f"Voxels: {len(model.voxels)}")
        self.dims_label.setText(f"Dimensions: {w} x {h} x {d} m")
        self.scan_result_label.setText("Progressive scan complete")
        self.scan_result_label.setStyleSheet("color: green;")

        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan 3D Model Hull")
        self.progressive_scan_btn.setEnabled(True)
        self.progressive_scan_btn.setText("Progressive Scan")
        self.gl_widget.update()

    def _on_progressive_error(self, error_msg):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan 3D Model Hull")
        self.progressive_scan_btn.setEnabled(True)
        self.progressive_scan_btn.setText("Progressive Scan")
        self.scan_result_label.setText(f"Scan failed: {error_msg}")
        self.scan_result_label.setStyleSheet("color: red;")

    def _on_live_setting_changed(self):
        if self.live_preview_cb.isChecked() and self.mesh_vertices is not None:
            self._schedule_live_preview()
        elif self.auto_rescan_chk.isChecked() and self._base_model is not None and self.use_dims_chk.isChecked():
            self._rescan_with_new_dims()

    def _rescan_with_new_dims(self):
        if self._base_model is None or not hasattr(self._base_model, 'voxels'):
            return
        width = self.dim_w_spin.value() or None
        height = self.dim_h_spin.value() or None
        depth = self.dim_d_spin.value() or None
        if width is None or height is None or depth is None:
            return
        resolution = self.max_res_spin.value()
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        hollow = self.hollow_chk.isChecked()
        thickness = self.thickness_spin.value()
        self.scan_btn.setEnabled(False)
        self.progressive_scan_btn.setEnabled(False)
        self.scan_result_label.setText("Rescanning with new dimensions...")
        self.scan_result_label.setStyleSheet("color: orange;")
        self.prog_thread = ProgressiveScanThread(
            self.current_file, resolution, width, height, depth,
            rot_x, rot_y, rot_z, hollow, thickness
        )
        self.prog_thread.progress.connect(self._on_progressive_progress)
        self.prog_thread.finished.connect(lambda model: self._on_progressive_complete(model, store_base=True))
        self.prog_thread.error.connect(self._on_progressive_error)
        self.prog_thread.start()

    def _on_smooth_toggled(self, checked):
        if self._base_model is None or not hasattr(self._base_model, 'voxels'):
            return
        if checked:
            from src.block_shapes import optimize_blocks
            smooth_model = optimize_blocks(self._base_model)
        else:
            smooth_model = self._base_model
        self.model = smooth_model
        self.gl_widget.set_model(smooth_model)
        x1, x2, y1, y2, z1, z2 = smooth_model.get_bounds()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        d = z2 - z1 + 1
        self.voxel_count_label.setText(f"Voxels: {len(smooth_model.voxels)}")
        self.dims_label.setText(f"Dimensions: {w} x {h} x {d} m")
        self.gl_widget.update()

    def _schedule_live_preview(self):
        if not self.live_preview_cb.isChecked():
            return
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
        self.live_preview_timer.stop()
        self.live_preview_timer.start(400)

    def _do_live_preview(self):
        if not self.live_preview_cb.isChecked():
            return
        if self.mesh_vertices is None or self.mesh_faces is None:
            return

        resolution = self.max_res_spin.value()
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        hollow = self.hollow_chk.isChecked()
        thickness = self.thickness_spin.value()

        if self.use_dims_chk.isChecked():
            width = self.dim_w_spin.value() or None
            height = self.dim_h_spin.value() or None
            depth = self.dim_d_spin.value() or None
        else:
            width = None
            height = None
            depth = None

        self.current_file = "live_preview"
        self.file_label.setText("Live preview...")
        self.load_btn.setEnabled(False)
        self.import_mesh_btn.setEnabled(False)
        self.export_label.setText("")

        self.mesh_thread = MeshImportThread(
            "live_preview", resolution, width, height, depth,
            rot_x, rot_y, rot_z, hollow, thickness,
            vertices=self.mesh_vertices, faces=self.mesh_faces
        )
        self.mesh_thread.finished.connect(self._on_live_preview_complete)
        self.mesh_thread.error.connect(self._on_live_preview_error)
        self.mesh_thread.start()

    def _on_live_preview_complete(self, model):
        self.model = model
        self._base_model = model
        fd = getattr(model, 'front_direction', (0, 0, -1))
        mapping = {
            (0, 0, -1): '-Z', (0, 0, 1): '+Z',
            (0, -1, 0): '-Y', (0, 1, 0): '+Y',
            (-1, 0, 0): '-X', (1, 0, 0): '+X',
        }
        rev = {v: k for k, v in mapping.items()}
        self.front_dir_combo.blockSignals(True)
        self.front_dir_combo.setCurrentText(mapping.get(fd, '-Z'))
        self.front_dir_combo.blockSignals(False)
        self.gl_widget.set_model(model)

        x1, x2, y1, y2, z1, z2 = model.get_bounds()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        d = z2 - z1 + 1

        self.voxel_count_label.setText(f"Voxels: {len(model.voxels)}")
        self.dims_label.setText(f"Dimensions: {w} x {h} x {d} m")

        self.core_x_spin.blockSignals(True)
        self.core_y_spin.blockSignals(True)
        self.core_z_spin.blockSignals(True)
        self.core_x_spin.setValue(model.core_x)
        self.core_y_spin.setValue(model.core_y)
        self.core_z_spin.setValue(model.core_z)
        self.core_x_spin.blockSignals(False)
        self.core_y_spin.blockSignals(False)
        self.core_z_spin.blockSignals(False)

        self.gl_widget.set_core_position(model.core_x, model.core_y, model.core_z)

        self.load_btn.setEnabled(True)
        self.load_btn.setText("Load Voxel Model")
        self.import_mesh_btn.setEnabled(True)
        self.import_mesh_btn.setText("Import 3D Model (.obj / .ply / .stl)")
        self.file_label.setText("Live preview ready")

    def _on_live_preview_error(self, error_msg):
        self.load_btn.setEnabled(True)
        self.load_btn.setText("Load Voxel Model")
        self.import_mesh_btn.setEnabled(True)
        self.import_mesh_btn.setText("Import 3D Model (.obj / .ply / .stl)")
        self.file_label.setText(f"Preview error: {error_msg}")

    def _load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Voxel Model", "",
            "Voxel Files (*.vox *.vxl *.csv *.json);;All Files (*.*)"
        )
        if not filename:
            return
        self._do_load(filename)

    def _import_mesh(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import 3D Model", "",
            "3D Models (*.obj *.ply *.stl);;All Files (*.*)"
        )
        if not filename:
            return

        resolution = self.max_res_spin.value()
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        hollow = self.hollow_chk.isChecked()
        thickness = self.thickness_spin.value()

        if self.use_dims_chk.isChecked():
            width = self.dim_w_spin.value() or None
            height = self.dim_h_spin.value() or None
            depth = self.dim_d_spin.value() or None
        else:
            width = None
            height = None
            depth = None

        self.current_file = filename
        self.file_label.setText(f"Importing: {os.path.basename(filename)}")
        self.load_btn.setEnabled(False)
        self.import_mesh_btn.setEnabled(False)
        self.load_btn.setText("Loading...")
        self.import_mesh_btn.setText("Importing...")

        self._load_mesh_for_wireframe(filename)

        self.mesh_thread = MeshImportThread(filename, resolution, width, height, depth, rot_x, rot_y, rot_z, hollow, thickness)
        self.mesh_thread.finished.connect(self._on_load_complete)
        self.mesh_thread.error.connect(self._on_load_error)
        self.mesh_thread.start()

    def _do_load(self, filename):
        self.current_file = filename
        self.file_label.setText(os.path.basename(filename))
        self.load_btn.setEnabled(False)
        self.load_btn.setText("Loading...")
        self.import_mesh_btn.setEnabled(False)

        hollow = self.hollow_chk.isChecked()
        thickness = self.thickness_spin.value()

        self.loader_thread = LoaderThread(filename, hollow, thickness)
        self.loader_thread.finished.connect(self._on_load_complete)
        self.loader_thread.error.connect(self._on_load_error)
        self.loader_thread.start()

    def _on_load_complete(self, model):
        self.model = model
        self._base_model = model
        fd = getattr(model, 'front_direction', (0, 0, -1))
        mapping = {
            (0, 0, -1): '-Z', (0, 0, 1): '+Z',
            (0, -1, 0): '-Y', (0, 1, 0): '+Y',
            (-1, 0, 0): '-X', (1, 0, 0): '+X',
        }
        self.front_dir_combo.blockSignals(True)
        self.front_dir_combo.setCurrentText(mapping.get(fd, '-Z'))
        self.front_dir_combo.blockSignals(False)
        self.gl_widget.set_model(model)

        x1, x2, y1, y2, z1, z2 = model.get_bounds()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        d = z2 - z1 + 1

        self.voxel_count_label.setText(f"Voxels: {len(model.voxels)}")
        self.dims_label.setText(f"Dimensions: {w} x {h} x {d} m")

        self.core_x_spin.blockSignals(True)
        self.core_y_spin.blockSignals(True)
        self.core_z_spin.blockSignals(True)

        self.core_x_spin.setValue(model.core_x)
        self.core_y_spin.setValue(model.core_y)
        self.core_z_spin.setValue(model.core_z)

        self.core_x_spin.blockSignals(False)
        self.core_y_spin.blockSignals(False)
        self.core_z_spin.blockSignals(False)

        self.gl_widget.set_core_position(model.core_x, model.core_y, model.core_z)

        self.load_btn.setEnabled(True)
        self.load_btn.setText("Load Voxel Model")
        self.import_mesh_btn.setEnabled(True)
        self.import_mesh_btn.setText("Import 3D Model (.obj / .ply / .stl)")
        self.export_label.setText("")

    def _on_load_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Failed to load file:\n{error_msg}")
        self.load_btn.setEnabled(True)
        self.load_btn.setText("Load Voxel Model")
        self.import_mesh_btn.setEnabled(True)
        self.import_mesh_btn.setText("Import 3D Model (.obj / .ply / .stl)")

    def _on_core_changed(self):
        if self.model is None:
            return
        cx = self.core_x_spin.value()
        cy = self.core_y_spin.value()
        cz = self.core_z_spin.value()
        self.model.core_x = cx
        self.model.core_y = cy
        self.model.core_z = cz
        self.gl_widget.set_core_position(cx, cy, cz)

    def _on_front_changed(self, text):
        if self.model is None:
            return
        mapping = {
            '-X': (-1, 0, 0),
            '+X': (1, 0, 0),
            '-Y': (0, -1, 0),
            '+Y': (0, 1, 0),
            '-Z': (0, 0, -1),
            '+Z': (0, 0, 1),
        }
        self.model.front_direction = mapping.get(text, (0, 0, -1))
        self.gl_widget.set_front_direction(mapping.get(text, (0, 0, -1)))

    def _center_core(self):
        if self.model is None:
            return
        x1, x2, y1, y2, z1, z2 = self.model.get_bounds()
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cz = (z1 + z2) // 2

        self.core_x_spin.setValue(cx)
        self.core_y_spin.setValue(cy)
        self.core_z_spin.setValue(cz)

    def _reset_core(self):
        self.core_x_spin.setValue(0)
        self.core_y_spin.setValue(0)
        self.core_z_spin.setValue(0)

    def _flip_x(self):
        if self.model is None:
            return
        xmin, xmax, _, _, _, _ = self.model.get_bounds()
        max_x = xmax - xmin
        new_voxels = {}
        for (x, y, z), info in self.model.voxels.items():
            new_x = max_x - (x - xmin)
            new_voxels[(new_x, y, z)] = info
        self.model.voxels = new_voxels
        self.model.core_x = max_x - (self.model.core_x - xmin)
        self.core_x_spin.setValue(self.model.core_x)
        self.gl_widget.update_voxel_cache()
        self.gl_widget.update()

    def _flip_y(self):
        if self.model is None:
            return
        _, _, ymin, ymax, _, _ = self.model.get_bounds()
        max_y = ymax - ymin
        new_voxels = {}
        for (x, y, z), info in self.model.voxels.items():
            new_y = max_y - (y - ymin)
            new_voxels[(x, new_y, z)] = info
        self.model.voxels = new_voxels
        self.model.core_y = max_y - (self.model.core_y - ymin)
        self.core_y_spin.setValue(self.model.core_y)
        self.gl_widget.update_voxel_cache()
        self.gl_widget.update()

    def _flip_z(self):
        if self.model is None:
            return
        _, _, _, _, zmin, zmax = self.model.get_bounds()
        max_z = zmax - zmin
        new_voxels = {}
        for (x, y, z), info in self.model.voxels.items():
            new_z = max_z - (z - zmin)
            new_voxels[(x, y, new_z)] = info
        self.model.voxels = new_voxels
        self.model.core_z = max_z - (self.model.core_z - zmin)
        self.core_z_spin.setValue(self.model.core_z)
        self.gl_widget.update_voxel_cache()
        self.gl_widget.update()

    def _apply_rotation(self):
        if self.model is None:
            return
        rx = self.rot_x_spin.value()
        ry = self.rot_y_spin.value()
        rz = self.rot_z_spin.value()
        if rx == 0 and ry == 0 and rz == 0:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = self.model.get_bounds()

        offset_x = xmin
        offset_y = ymin
        offset_z = zmin

        new_voxels = {}
        for (x, y, z), info in self.model.voxels.items():
            nx = x - offset_x
            ny = y - offset_y
            nz = z - offset_z

            for _ in range(rz // 90):
                nx, ny = -ny, nx
            for _ in range(ry // 90):
                nx, nz = nz, -nx
            for _ in range(rx // 90):
                ny, nz = -nz, ny

            new_voxels[(nx + offset_x, ny + offset_y, nz + offset_z)] = info

        self.model.voxels = new_voxels

        cx = self.model.core_x - offset_x
        cy = self.model.core_y - offset_y
        cz = self.model.core_z - offset_z
        for _ in range(rz // 90):
            cx, cy = -cy, cx
        for _ in range(ry // 90):
            cx, cz = cz, -cx
        for _ in range(rx // 90):
            cy, cz = -cz, cy
        self.model.core_x = cx + offset_x
        self.model.core_y = cy + offset_y
        self.model.core_z = cz + offset_z

        self._base_model = self.model
        self.core_x_spin.setValue(self.model.core_x)
        self.core_y_spin.setValue(self.model.core_y)
        self.core_z_spin.setValue(self.model.core_z)

        self.gl_widget.update_voxel_cache()
        self.gl_widget.update()

        x1, x2, y1, y2, z1, z2 = self.model.get_bounds()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        d = z2 - z1 + 1
        self.dims_label.setText(f"Dimensions: {w} x {h} x {d} m")

    def _clear_model(self):
        self.model = None
        self._base_model = None
        self.gl_widget.set_model(None)
        self.voxel_count_label.setText("Voxels: 0")
        self.dims_label.setText("Dimensions: 0 x 0 x 0")
        self.file_label.setText("No file loaded")
        self.current_file = None

    def _on_view_changed(self):
        self.gl_widget.show_grid = self.show_grid_cb.isChecked()
        self.gl_widget.show_axes = self.show_axes_cb.isChecked()
        self.gl_widget.show_core_indicator = self.show_core_cb.isChecked()
        self.gl_widget.show_faces = self.show_faces_cb.isChecked()
        self.gl_widget.show_wireframe = self.show_wireframe_cb.isChecked()
        self.gl_widget.show_mesh_wireframe = self.show_mesh_wireframe_cb.isChecked()
        self.gl_widget.update()

    def _export_obj(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model loaded to export!")
            return

        default_name = "model.obj"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base}_starmade.obj"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export StarMade OBJ", default_name,
            "OBJ Files (*.obj);;All Files (*.*)"
        )
        if filename:
            try:
                export_starmade_obj(self.model, filename)
                self.export_label.setText(f"Exported: {os.path.basename(filename)}")
                QMessageBox.information(self, "Success", f"Model exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _export_sment(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model loaded to export!")
            return

        default_name = "blueprint.sment"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base}.sment"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export StarMade Blueprint", default_name,
            "StarMade Blueprint (*.sment);;All Files (*.*)"
        )
        if filename:
            try:
                export_starmade_sment(self.model, filename)
                self.export_label.setText(f"Exported: {os.path.basename(filename)}")
                QMessageBox.information(self, "Success", f"StarMade blueprint exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _export_to_blueprints_dir(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model loaded to export!")
            return

        default_name = "MyShip"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = base

        blueprint_name, ok = QInputDialog.getText(
            self, "Blueprint Name", "Enter blueprint name:", text=default_name
        )
        if not ok or not blueprint_name.strip():
            return

        blueprint_name = blueprint_name.strip()
        base_dir = os.path.join(os.environ.get('APPDATA', ''), 'starmade-launcher', 'blueprints')
        output_dir = os.path.join(base_dir, blueprint_name)

        if os.path.exists(output_dir):
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"Blueprint folder already exists:\n{output_dir}\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            os.makedirs(output_dir, exist_ok=True)
            export_starmade_dir(self.model, base_dir, blueprint_name)
            self.export_label.setText(f"Exported to: {base_dir}")
            QMessageBox.information(self, "Success", f"Blueprint exported to:\n{output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def closeEvent(self, event: QCloseEvent):
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
