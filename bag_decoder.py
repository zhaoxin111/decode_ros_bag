"""ROS 2 SQLite bag decoding and Odin1 camera/geometry primitives.

Conventions: column vectors; T_A_B maps B into A; distances in meters;
quaternions xyzw; camera optical axes right/down/forward.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import struct
from urllib.parse import quote

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp
import yaml


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def json_message(value):
    """Lossless JSON representation for small ROS messages, including covariance."""
    if hasattr(value, '__dataclass_fields__'):
        return {k: json_message(getattr(value, k)) for k in value.__dataclass_fields__
                if not k.startswith('_') and k not in ('ClassVar',)}
    if isinstance(value, np.ndarray):
        return json_message(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_message(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_message(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class MessageRef:
    file_index: int
    row_id: int
    topic: str
    msgtype: str
    bag_ns: int
    stamp_ns: int | None

    @property
    def key(self):
        return f'{self.file_index:03d}_{self.row_id:09d}'

    def provenance(self, files):
        return dict(bag_file=files[self.file_index].name, message_id=self.row_id,
                    topic=self.topic, bag_timestamp_ns=self.bag_ns,
                    timestamp_ns=self.stamp_ns)


class SqliteBag:
    """Header-only indexing avoids deserializing gigabytes just to find timestamps.

    Uses ROS 2's standard CDR header and rosbags' standard message definitions.
    Payloads are fetched by SQLite primary key, once selected. All DBs are read-only.
    """
    def __init__(self, input_path):
        self.input_path = Path(input_path).resolve()
        self.directory = self.input_path / 'bag' if (self.input_path / 'bag').is_dir() else self.input_path
        if not self.directory.is_dir():
            raise ValueError(f'Not a recording/bag directory: {self.input_path}')
        metadata = self.directory / 'metadata.yaml'
        if metadata.exists():
            self.metadata = yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']
            if self.metadata.get('storage_identifier') != 'sqlite3':
                raise ValueError('Only ROS 2 sqlite3 bags are supported')
            if self.metadata.get('compression_mode') not in (None, '', 'NONE'):
                raise ValueError('Compressed bag storage is not supported; decompress the bag first')
            self.files = [(self.directory / p).resolve() for p in self.metadata['relative_file_paths']]
        else:
            self.metadata = {}
            self.files = sorted(self.directory.glob('*.db3'))
        if not self.files:
            raise ValueError('No .db3 bag files found')
        self.connections = []
        self.topics = {}
        self.topic_ids = []
        self.types = get_typestore(Stores.ROS2_HUMBLE)
        try:
            for path in self.files:
                db = sqlite3.connect(f'file:{quote(str(path))}?mode=ro', uri=True)
                self.connections.append(db)
                topics = {}
                for tid, name, typ, fmt in db.execute('SELECT id,name,type,serialization_format FROM topics'):
                    if fmt != 'cdr':
                        raise ValueError(f'Unsupported serialization {fmt} on {name}')
                    if name in self.topics and self.topics[name] != typ:
                        raise ValueError(f'Topic type changes across split bags: {name}')
                    self.topics[name] = typ
                    topics[tid] = (name, typ)
                self.topic_ids.append(topics)
        except Exception:
            self.close()
            raise

    def close(self):
        for db in self.connections:
            db.close()

    def index(self, topics):
        refs = {topic: [] for topic in topics if topic in self.topics}
        for file_index, (db, ids) in enumerate(zip(self.connections, self.topic_ids)):
            wanted = [tid for tid, (name, _) in ids.items() if name in refs]
            if not wanted:
                continue
            query = ('SELECT id,topic_id,timestamp,substr(data,1,12) FROM messages '
                     f'WHERE topic_id IN ({",".join("?" for _ in wanted)})')
            for row, tid, bag_ns, head in db.execute(query, wanted):
                topic, typ = ids[tid]
                header_time = None
                if typ in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage',
                           'sensor_msgs/msg/PointCloud2', 'sensor_msgs/msg/Imu', 'nav_msgs/msg/Odometry'):
                    if len(head) != 12 or head[:2] not in (b'\x00\x01', b'\x00\x00'):
                        raise ValueError(f'Invalid CDR header: {topic} file {file_index} row {row}')
                    sec, nsec = struct.unpack('<iI' if head[1] else '>iI', head[4:12])
                    if nsec >= 1_000_000_000:
                        raise ValueError(f'Invalid header timestamp: {topic} row {row}')
                    header_time = sec * 1_000_000_000 + nsec
                refs[topic].append(MessageRef(file_index, row, topic, typ, int(bag_ns), header_time))
            print(f'Indexed {file_index + 1}/{len(self.files)}: {self.files[file_index].name}', flush=True)
        for topic in refs:
            refs[topic].sort(key=lambda r: (r.stamp_ns if r.stamp_ns is not None else r.bag_ns,
                                           r.bag_ns, r.file_index, r.row_id))
        return refs

    def read(self, ref):
        data = self.connections[ref.file_index].execute(
            'SELECT data FROM messages WHERE id=?', (ref.row_id,)).fetchone()[0]
        return self.types.deserialize_cdr(data, ref.msgtype)


POINT_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def point_array(msg):
    """Decode all fields, preserving count, endian, unaligned offsets and row padding."""
    names, formats, offsets = [], [], []
    for field in msg.fields:
        if field.name in names or field.datatype not in POINT_TYPES or field.count < 1:
            raise ValueError(f'Invalid/unsupported PointCloud2 field: {field}')
        dt = np.dtype(('>' if msg.is_bigendian else '<') + POINT_TYPES[field.datatype])
        if field.offset < 0 or field.offset + dt.itemsize * field.count > msg.point_step:
            raise ValueError(f'PointCloud2 field exceeds point_step: {field.name}')
        names.append(field.name)
        formats.append(dt if field.count == 1 else (dt, (field.count,)))
        offsets.append(field.offset)
    if not {'x', 'y', 'z'}.issubset(names):
        raise ValueError('PointCloud2 must have x, y, z fields')
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=msg.point_step))
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.height * msg.row_step:
        raise ValueError('Invalid PointCloud2 layout/data length')
    view = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step, msg.point_step))
    # Structured copies may leave padding bytes uninitialized. Zero padding explicitly;
    # original padding remains available in the separately saved raw message data.
    result = np.zeros((msg.height, msg.width), dtype=dtype)
    for name in names:
        result[name] = view[name]
    return result


def cloud_xyz(points):
    flat = points.reshape(-1)
    return np.column_stack([flat[n] for n in ('x', 'y', 'z')]).astype(np.float64)


def save_raw_cloud(path, msg, points=None):
    points = point_array(msg) if points is None else points
    # Original bytes preserve row padding and non-field bytes as well as decoded fields.
    np.savez_compressed(path, points=points, data=np.asarray(msg.data, dtype=np.uint8),
                        height=msg.height, width=msg.width, point_step=msg.point_step,
                        row_step=msg.row_step, is_bigendian=msg.is_bigendian,
                        is_dense=msg.is_dense, frame_id=msg.header.frame_id,
                        timestamp_ns=stamp_ns(msg),
                        fields_json=json.dumps([dict(name=f.name, offset=f.offset,
                                                     datatype=f.datatype, count=f.count) for f in msg.fields]))


def decode_image(msg):
    if msg.__msgtype__ == 'sensor_msgs/msg/CompressedImage':
        image = cv2.imdecode(np.asarray(msg.data), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('Cannot decode compressed image')
        return image
    encodings = {'bgr8': ('u1', 3), 'rgb8': ('u1', 3), 'mono8': ('u1', 1),
                 'bgra8': ('u1', 4), 'rgba8': ('u1', 4), 'mono16': ('u2', 1),
                 '16UC1': ('u2', 1), '32FC1': ('f4', 1), '64FC1': ('f8', 1)}
    if msg.encoding not in encodings:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    kind, channels = encodings[msg.encoding]
    dtype = np.dtype(('>' if msg.is_bigendian else '<') + kind)
    itemsize = dtype.itemsize
    if msg.step < msg.width * channels * itemsize or len(msg.data) < msg.height * msg.step:
        raise ValueError('Invalid Image step/data length')
    image = np.ndarray((msg.height, msg.width, channels), dtype=dtype, buffer=msg.data,
                       strides=(msg.step, channels * itemsize, itemsize)).astype(dtype.newbyteorder('='))
    if channels == 1:
        image = image[..., 0]
    elif msg.encoding in ('rgb8', 'rgba8'):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGRA)
    return image


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Failed to write image: {path}')


def transform_points(transform, xyz):
    return np.asarray(xyz) @ transform[:3, :3].T + transform[:3, 3]


def validate_transform(transform, name='transform'):
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f'Invalid homogeneous {name}')
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=2e-4) or np.linalg.det(t[:3, :3]) < 0:
        raise ValueError(f'Invalid rotation in {name}')
    return t


class PoseUnavailable(ValueError):
    pass


class PoseTrack:
    def __init__(self, refs, bag, scale=1.0):
        self.topic = refs[0].topic if refs else ''
        self.frame = self.child = None
        times, xyz, quats = [], [], []
        self.rejected = Counter()
        for ref in refs:
            msg = bag.read(ref)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pos, quat = np.array([p.x, p.y, p.z]) * scale, np.array([q.x, q.y, q.z, q.w])
            if not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
                self.rejected['invalid_pose'] += 1
                continue
            if times and ref.stamp_ns == times[-1]:
                self.rejected['duplicate_timestamp'] += 1
                continue
            frame, child = msg.header.frame_id, msg.child_frame_id
            if not frame or not child:
                raise ValueError(f'Empty odometry frames on {self.topic}')
            if self.frame is not None and (frame, child) != (self.frame, self.child):
                raise ValueError(f'Odometry frame changed on {self.topic}')
            self.frame, self.child = frame, child
            times.append(ref.stamp_ns)
            xyz.append(pos)
            quats.append(quat / np.linalg.norm(quat))
        self.times = np.asarray(times, dtype=np.int64)
        self.xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        self.quats = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
        self.origin = int(self.times[0]) if len(times) else 0
        self.seconds = (self.times - self.origin) * 1e-9
        self.rotations = Rotation.from_quat(self.quats) if len(times) else None
        self.slerp = Slerp(self.seconds, self.rotations) if len(times) > 1 else None

    def evaluate(self, times_ns, max_gap_ns):
        times_ns = np.atleast_1d(np.asarray(times_ns, dtype=np.int64))
        if not len(self.times):
            raise PoseUnavailable(f'No valid poses on {self.topic}')
        hi = np.searchsorted(self.times, times_ns)
        candidate = np.minimum(hi, len(self.times) - 1)
        exact = self.times[candidate] == times_ns
        lo = np.maximum(hi - 1, 0)
        hi = candidate
        lo[exact] = hi[exact]
        valid = ((times_ns >= self.times[0]) & (times_ns <= self.times[-1]) &
                 (self.times[hi] - self.times[lo] <= max_gap_ns))
        if not valid.all():
            raise PoseUnavailable(f'Outside pose coverage or gap > {max_gap_ns / 1e6:g} ms on {self.topic}')
        gap = self.times[hi] - self.times[lo]
        alpha = np.divide(times_ns - self.times[lo], gap, out=np.zeros(len(gap)), where=gap != 0)
        pos = self.xyz[lo] * (1 - alpha[:, None]) + self.xyz[hi] * alpha[:, None]
        if self.slerp is None:
            rot = np.repeat(self.rotations.as_matrix(), len(times_ns), axis=0)
        else:
            rot = self.slerp((times_ns - self.origin) * 1e-9).as_matrix()
        matrices = np.repeat(np.eye(4)[None], len(times_ns), axis=0)
        matrices[:, :3, :3] = rot
        matrices[:, :3, 3] = pos
        brackets = np.column_stack((self.times[lo], self.times[hi]))
        return matrices, brackets

    def summary(self):
        dt = np.diff(self.times) * 1e-9
        speed = np.linalg.norm(np.diff(self.xyz, axis=0), axis=1) / dt if len(dt) else []
        return dict(topic=self.topic, frame_id=self.frame, child_frame_id=self.child,
                    count=len(self.times), rejected=dict(self.rejected),
                    time_range_ns=[int(self.times[0]), int(self.times[-1])] if len(self.times) else [],
                    gap_ms=distribution(dt * 1000), speed_m_s=distribution(speed))


def choose_track(tracks, times_ns, max_gap_ns):
    reasons = []
    for track in tracks:
        try:
            matrices, brackets = track.evaluate(times_ns, max_gap_ns)
            return track, matrices, brackets
        except PoseUnavailable as exc:
            reasons.append(str(exc))
    raise PoseUnavailable('; '.join(reasons))


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {name: float(v) for name, v in zip(('min', 'median', 'p95', 'max'),
                                            np.percentile(values, [0, 50, 95, 100]))}


class FishPoly:
    """theta_d = theta + k2*theta**2 + ... + k7*theta**7 (not OpenCV fisheye)."""
    def __init__(self, calibration, width=800, height=648, fov=90.0):
        self.source = calibration
        self.cam = calibration['cam_0']
        if self.cam['cam_model'] != 'FishPoly':
            raise ValueError('Only Odin1 FishPoly calibration is supported')
        if self.cam.get('p1', 0) != 0 or self.cam.get('p2', 0) != 0:
            raise ValueError('Nonzero tangential distortion is not supported')
        self.t_camera_lidar = validate_transform(calibration['Tcl_0'], 'Tcl_0')
        self.width, self.height = width, height
        f = width / (2 * np.tan(np.deg2rad(fov) / 2))
        self.k = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
        v, u = np.mgrid[:height, :width]
        rays = np.stack([(u - self.k[0, 2]) / f, (v - self.k[1, 2]) / f, np.ones_like(u)], axis=-1)
        uv = self.project_original(rays.reshape(-1, 3)).reshape(height, width, 2)
        self.map_x, self.map_y = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
        self.valid = ((self.map_x >= 0) & (self.map_y >= 0) &
                      (self.map_x < self.cam['image_width'] - 1) &
                      (self.map_y < self.cam['image_height'] - 1))

    def project_original(self, xyz):
        x, y, z = np.asarray(xyz).T
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        td = theta.copy()
        for power in range(2, 8):
            td += self.cam[f'k{power}'] * theta ** power
        scale = np.divide(td, r, out=np.zeros_like(r), where=r > 1e-12)
        xd, yd = x * scale, y * scale
        return np.column_stack((self.cam['A11'] * xd + self.cam['A12'] * yd + self.cam['u0'],
                                self.cam['A22'] * yd + self.cam['v0']))

    def project_rectified(self, xyz):
        xyz = np.asarray(xyz)
        uvw = xyz @ self.k.T
        return uvw[:, :2] / np.where(np.abs(uvw[:, 2:3]) > 1e-15, uvw[:, 2:3], np.nan)

    def rectify(self, image):
        if image.shape[:2] != (self.cam['image_height'], self.cam['image_width']):
            raise ValueError(f'Image size {image.shape[:2]} disagrees with calibration')
        out = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[~self.valid] = 0
        return out


def rasterize(uv, xyz, width, height, mask=None):
    """Float32 camera-Z sparse depth; nearest Z per rounded pixel; NaN is missing."""
    good = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    good &= (uv[:, 0] >= -.5) & (uv[:, 0] < width - .5) & (uv[:, 1] >= -.5) & (uv[:, 1] < height - .5)
    pixels = np.floor(uv[good] + .5).astype(np.int64)
    depth = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(depth, pixels[:, 1] * width + pixels[:, 0], xyz[good, 2].astype(np.float32))
    depth[~np.isfinite(depth)] = np.nan
    depth = depth.reshape(height, width)
    if mask is not None:
        depth[~mask] = np.nan
    return depth


def overlay_depth(image, depth):
    image = image.copy()
    valid = np.isfinite(depth)
    values = depth[valid]
    if not len(values):
        return image
    low, high = np.percentile(values, [2, 98])
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip((values - low) / max(high - low, .01) * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    # Enlarge only in the visualization, never in the exported depth.
    expanded = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    colored = np.zeros_like(image)
    colored[valid] = colors[valid]
    colored = cv2.dilate(colored, np.ones((3, 3), np.uint8))
    image[expanded] = cv2.addWeighted(image, .25, colored, .75, 0)[expanded]
    return image


def save_ply(path, xyz):
    xyz = np.asarray(xyz, dtype='<f4')
    header = ('ply\nformat binary_little_endian 1.0\ncomment camera optical coordinates, meters\n'
              f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        xyz.tofile(stream)
