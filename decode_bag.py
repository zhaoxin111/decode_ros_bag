#!/usr/bin/env python3
"""Export synchronized Odin1 image / point cloud / pose samples from a ROS 2 bag."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import yaml

from bag_decoder import (
    FishPoly, PoseTrack, PoseUnavailable, SqliteBag, choose_track, cloud_xyz,
    decode_image, distribution, dump_json, json_message, overlay_depth, point_array,
    rasterize, save_ply, save_raw_cloud, stamp_ns, transform_points, validate_transform,
    write_image,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', type=Path, required=True, help='Recording directory or bag/ containing split .db3 files')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    p.add_argument('--calib', type=Path, help='Odin1 calib.yaml; discovered beside bag by default')
    p.add_argument('--image-topic', help='Default: compressed RGB, falling back to raw Image if absent/empty')
    p.add_argument('--cloud-topic', default='/odin1/cloud_raw')
    p.add_argument('--pose-topic', default='/odin1/odometry_highfreq')
    p.add_argument('--fallback-pose-topic', default='/odin1/odometry', help='Fallback for gaps; empty string disables')
    p.add_argument('--pose-translation-scale', type=float, default=1.0, help='Multiply odometry translation to obtain meters')
    p.add_argument('--base-lidar-transform', type=Path, help='YAML 4x4 matrix T_base_lidar; otherwise cloud frame must equal odometry child')
    p.add_argument('--width', type=int, default=800)
    p.add_argument('--height', type=int, default=648)
    p.add_argument('--fov', type=float, default=90.0, help='Rectified horizontal field of view, degrees')
    p.add_argument('--max-delta-ms', type=float, default=50.0, help='Maximum absolute image/cloud header-time difference')
    p.add_argument('--max-pose-gap-ms', type=float, default=250.0)
    p.add_argument('--motion-compensation', choices=['frame', 'point', 'none'], default='frame')
    p.add_argument('--point-time-reference', choices=['start', 'end'], help='Required for point mode: cloud header denotes scan start/end')
    p.add_argument('--point-time-unit', choices=['s', 'ms', 'us', 'ns'], help='Required for point mode: offset_time unit; explicitly confirm with driver/SDK')
    p.add_argument('--confidence', type=float, default=35.0)
    p.add_argument('--min-range', type=float, default=0.1, help='Minimum Euclidean range in the source lidar frame, meters')
    p.add_argument('--start', type=float, default=0.0, help='Start seconds relative to first image header time')
    p.add_argument('--end', type=float, help='End seconds relative to first image header time, inclusive')
    p.add_argument('--stride', type=int, default=1, help='Select every Nth image within the requested interval')
    p.add_argument('--max-frames', type=int, help='Stop after this many successful samples')
    p.add_argument('--ply', action='store_true', help='Also export aligned camera-frame point clouds as binary PLY')
    p.add_argument('--preview-count', type=int, default=8, help='Number of evenly spaced projection checks; zero disables')
    p.add_argument('--export-imu', action='store_true')
    p.add_argument('--export-tf', action='store_true')
    p.add_argument('--export-trajectories', action='store_true', help='Export both original odometry streams as JSONL and TUM')
    p.add_argument('--export-extra-clouds', action='store_true', help='Export all remaining PointCloud2 topics as raw NPZ')
    p.add_argument('--export-depth-products', action='store_true', help='Export recorded non-RGB Image topics as NPY (no unit conversion)')
    return p


def validate_args(args):
    if args.width < 1 or args.height < 1 or not 0 < args.fov < 180:
        raise ValueError('Image dimensions must be positive and FOV must be between 0 and 180 degrees')
    if args.max_delta_ms < 0 or args.max_pose_gap_ms <= 0 or args.min_range < 0:
        raise ValueError('Invalid time/range threshold')
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1) or args.preview_count < 0:
        raise ValueError('stride/max-frames must be positive and preview-count must be nonnegative')
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError('Require 0 <= start <= end')
    if not np.isfinite(args.pose_translation_scale) or args.pose_translation_scale <= 0:
        raise ValueError('pose-translation-scale must be finite and positive')
    for name in ('max_delta_ms', 'max_pose_gap_ms', 'min_range', 'confidence', 'start', 'end'):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if args.motion_compensation == 'point' and (args.point_time_reference is None or args.point_time_unit is None):
        raise ValueError('Point compensation needs explicit --point-time-reference and --point-time-unit; '
                         'confirm the time convention and that source XYZ is not already deskewed')
    if args.motion_compensation != 'point' and (args.point_time_reference or args.point_time_unit):
        raise ValueError('Point time arguments require --motion-compensation point')
    if args.output.exists():
        raise ValueError(f'Output already exists: {args.output}; choose a new directory')


def calibrate(args, bag):
    candidates = [bag.directory / 'calib.yaml', bag.directory.parent / 'calib.yaml']
    path = args.calib or next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError('No calib.yaml found; supply --calib')
    camera = FishPoly(yaml.safe_load(path.read_text()), args.width, args.height, args.fov)
    t_base_lidar = np.eye(4)
    if args.base_lidar_transform:
        t_base_lidar = validate_transform(yaml.safe_load(args.base_lidar_transform.read_text()), 'T_base_lidar')
    return path.resolve(), camera, t_base_lidar


def select_topics(args, bag):
    if args.image_topic is None:
        args.image_topic = ('/odin1/image/compressed' if '/odin1/image/compressed' in bag.topics else '/odin1/image')
    wanted = {args.image_topic, args.cloud_topic, args.pose_topic, args.fallback_pose_topic}
    # Index raw RGB as well, so a recorded but empty compressed stream can fall back.
    if not getattr(args, '_explicit_image_topic', False):
        wanted.add('/odin1/image')
    for topic, typ in bag.topics.items():
        if (args.export_imu and typ == 'sensor_msgs/msg/Imu' or
            args.export_tf and typ == 'tf2_msgs/msg/TFMessage' or
            args.export_trajectories and typ == 'nav_msgs/msg/Odometry' or
            args.export_extra_clouds and typ == 'sensor_msgs/msg/PointCloud2' or
            args.export_depth_products and typ == 'sensor_msgs/msg/Image' and topic not in
                ('/odin1/image', '/odin1/image/undistorted')):
            wanted.add(topic)
    return wanted - {None, ''}


def load_tracks(args, refs, bag):
    tracks = []
    for topic in dict.fromkeys((args.pose_topic, args.fallback_pose_topic)):
        if refs.get(topic):
            if bag.topics[topic] != 'nav_msgs/msg/Odometry':
                raise ValueError(f'Pose topic must be nav_msgs/msg/Odometry: {topic}')
            track = PoseTrack(refs[topic], bag, args.pose_translation_scale)
            if len(track.times):
                tracks.append(track)
    if not tracks:
        raise ValueError('No usable odometry on the selected pose topics')
    if len({(t.frame, t.child) for t in tracks}) > 1:
        raise ValueError('Primary and fallback odometry use different frames; choose a single consistent pose source')
    return tracks


def track_agreement(tracks, max_gap_ns):
    if len(tracks) < 2:
        return None
    a, b = tracks[:2]
    distances, angles = [], []
    for t in a.times:
        try:
            ma, _ = a.evaluate([t], max_gap_ns)
            mb, _ = b.evaluate([t], max_gap_ns)
        except PoseUnavailable:
            continue
        distances.append(np.linalg.norm(ma[0, :3, 3] - mb[0, :3, 3]))
        rot = ma[0, :3, :3].T @ mb[0, :3, :3]
        angles.append(np.rad2deg(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    return dict(overlapping_samples=len(distances), translation_difference_m=distribution(distances),
                rotation_difference_deg=distribution(angles))


def nearest_ref(refs, times, target):
    at = int(np.searchsorted(times, target))
    candidates = [i for i in (at - 1, at) if 0 <= i < len(refs)]
    # Deterministic ties favor the earlier cloud.
    best = min(candidates, key=lambda i: (abs(int(times[i]) - target), int(times[i]), i))
    return refs[best]


def point_times(msg, points, args):
    if 'offset_time' not in points.dtype.names:
        raise ValueError('Point compensation requires offset_time in PointCloud2')
    offsets = points.reshape(-1)['offset_time'].astype(np.float64)
    factor = {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1}[args.point_time_unit]
    offsets_ns = offsets * factor
    if not np.isfinite(offsets_ns).all() or (offsets_ns < 0).any() or (offsets_ns > 10e9).any():
        raise ValueError('offset_time must contain finite, nonnegative offsets <= 10 s; check the supplied unit')
    offsets_ns = np.rint(offsets_ns).astype(np.int64)
    if args.point_time_reference == 'end':
        offsets_ns -= offsets_ns.max(initial=0)
    return stamp_ns(msg) + offsets_ns


def process_geometry(args, msg, points, image_ns, tracks, camera, t_base_lidar):
    xyz = cloud_xyz(points)
    ranges = np.linalg.norm(xyz, axis=1)
    good = np.isfinite(xyz).all(axis=1) & (ranges >= args.min_range) & (ranges > 0)
    finite_range_count = int(good.sum())
    flat = points.reshape(-1)
    if 'confidence' in flat.dtype.names:
        good &= np.isfinite(flat['confidence']) & (flat['confidence'] >= args.confidence)
    indices = np.flatnonzero(good)
    xyz = xyz[good]
    if args.motion_compensation == 'point':
        all_point_ns = point_times(msg, points, args)
        # Require coverage of the entire scan, including points filtered from model input.
        unique, inverse = np.unique(all_point_ns, return_inverse=True)
        query_times = np.concatenate(([image_ns], unique))
    elif args.motion_compensation == 'frame':
        query_times = np.array([image_ns, stamp_ns(msg)], dtype=np.int64)
    else:
        query_times = np.array([image_ns], dtype=np.int64)
    track, matrices, brackets = choose_track(tracks, query_times, round(args.max_pose_gap_ms * 1e6))
    if msg.header.frame_id != track.child and not args.base_lidar_transform:
        raise ValueError(f'Cloud frame {msg.header.frame_id!r} != pose child {track.child!r}; '
                         'supply the verified --base-lidar-transform')
    t_world_lidar_image = matrices[0] @ t_base_lidar
    t_world_camera = t_world_lidar_image @ np.linalg.inv(camera.t_camera_lidar)
    original_camera_xyz = transform_points(camera.t_camera_lidar, xyz)
    if args.motion_compensation == 'frame':
        transform = camera.t_camera_lidar @ np.linalg.inv(t_world_lidar_image) @ matrices[1] @ t_base_lidar
        aligned = transform_points(transform, xyz)
    elif args.motion_compensation == 'point':
        transforms = (camera.t_camera_lidar @ np.linalg.inv(t_world_lidar_image))[None] @ matrices[1:] @ t_base_lidar
        per_point = transforms[inverse[good]]
        aligned = np.einsum('nij,nj->ni', per_point[:, :3, :3], xyz) + per_point[:, :3, 3]
    else:
        aligned = original_camera_xyz.copy()
    # Only finite camera-front points are model inputs; raw clouds remain untouched.
    valid = np.isfinite(aligned).all(axis=1) & (aligned[:, 2] > 0)
    stats = dict(raw_points=len(flat), finite_range_points=finite_range_count,
                 filtered_points=len(indices), camera_front_points=int(valid.sum()),
                 confidence_filter_applied='confidence' in flat.dtype.names,
                 compensation_displacement_m=distribution(np.linalg.norm(aligned - original_camera_xyz, axis=1)))
    if 'offset_time' in flat.dtype.names:
        off = flat['offset_time']
        if off.size and np.isfinite(off).all():
            stats['offset_time_native_range'] = [float(off.min()), float(off.max())]
    provenance = dict(topic=track.topic, image_bracket_ns=brackets[0].tolist(),
                      source_time_range_ns=[int(query_times.min()), int(query_times.max())],
                      maximum_used_pose_gap_ms=float(np.max(brackets[:, 1] - brackets[:, 0]) / 1e6))
    return aligned[valid], indices[valid], original_camera_xyz[valid], t_world_camera, stats, provenance


def save_original(path_base, msg, image):
    if msg.__msgtype__ == 'sensor_msgs/msg/CompressedImage':
        data = np.asarray(msg.data, dtype=np.uint8).tobytes()
        if data.startswith(b'\xff\xd8'):
            path = path_base.with_suffix('.jpg')
        elif data.startswith(b'\x89PNG'):
            path = path_base.with_suffix('.png')
        else:
            path = path_base.with_suffix('.png')
            write_image(path, image)
            return path
        path.write_bytes(data)
    else:
        path = path_base.with_suffix('.png')
        write_image(path, image)
    return path


def save_preview(output, name, image, rectified, xyz, uncompensated, depth, camera):
    shape = image.shape[:2]
    original_depth = rasterize(camera.project_original(xyz), xyz, shape[1], shape[0])
    before = rasterize(camera.project_rectified(uncompensated), uncompensated,
                       camera.width, camera.height, camera.valid)
    out = output / 'previews'
    out.mkdir(exist_ok=True)
    write_image(out / f'{name}_original_overlay.jpg', overlay_depth(image, original_depth))
    write_image(out / f'{name}_rectified_overlay.jpg', overlay_depth(rectified, depth))
    panels = []
    for title, panel in [('RGB', rectified), ('before frame/point compensation', overlay_depth(rectified, before)),
                         ('after compensation', overlay_depth(rectified, depth))]:
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    write_image(out / f'{name}_comparison.jpg', np.concatenate(panels, axis=1))


def generate_previews(output, camera, count):
    """Choose successful samples, including endpoints and a frame with visible motion."""
    if count == 0:
        return []
    entries = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
    count = min(count, len(entries))
    if count == 0:
        return []
    selected = np.linspace(0, len(entries) - 1, count, dtype=int).tolist()
    if count >= 4:
        most_motion = max(range(len(entries)), key=lambda i:
                          entries[i]['stats']['compensation_displacement_m']['p95'])
        if most_motion not in selected:
            selected[-2] = most_motion
    previews = []
    for index in sorted(selected):
        entry = entries[index]
        files = entry['files']
        image = cv2.imread(str(output / files['image_original']))
        rectified = cv2.imread(str(output / files['image_rectified']))
        with np.load(output / files['cloud_aligned'], allow_pickle=False) as aligned:
            xyz = aligned['xyz']
            indices = aligned['source_indices']
        with np.load(output / files['cloud_raw'], allow_pickle=False) as raw:
            original = cloud_xyz(raw['points'])[indices]
        uncompensated = transform_points(camera.t_camera_lidar, original)
        depth = np.load(output / files['depth'], allow_pickle=False)
        save_preview(output, entry['id'], image, rectified, xyz, uncompensated, depth, camera)
        previews.append(f'previews/{entry["id"]}_comparison.jpg')
    return previews


def safe_topic_name(topic):
    # Encode rather than replace: /a_b and /a/b must never collide.
    from urllib.parse import quote
    return quote(topic, safe='')


def export_extras(args, bag, refs, output):
    exports = {}
    for topic, messages in refs.items():
        typ = bag.topics[topic]
        kind = None
        if args.export_imu and typ == 'sensor_msgs/msg/Imu':
            kind = 'imu'
        elif args.export_tf and typ == 'tf2_msgs/msg/TFMessage':
            kind = 'tf'
        elif args.export_trajectories and typ == 'nav_msgs/msg/Odometry':
            kind = 'trajectories'
        elif args.export_extra_clouds and typ == 'sensor_msgs/msg/PointCloud2' and topic != args.cloud_topic:
            kind = 'clouds'
        elif args.export_depth_products and typ == 'sensor_msgs/msg/Image' and topic not in (
                args.image_topic, '/odin1/image', '/odin1/image/undistorted'):
            kind = 'depth_products'
        if kind is None:
            continue
        directory = output / 'extras' / kind / safe_topic_name(topic)
        directory.mkdir(parents=True)
        print(f'Export optional {topic}: {len(messages)} messages', flush=True)
        with (directory / 'messages.jsonl').open('w') as stream:
            tum = (directory / 'trajectory.tum').open('w') if kind == 'trajectories' else None
            try:
                if tum:
                    tum.write('# timestamp_seconds tx ty tz qx qy qz qw; translation meters after configured scale\n')
                for ref in messages:
                    msg = bag.read(ref)
                    entry = ref.provenance(bag.files)
                    if kind in ('imu', 'tf', 'trajectories'):
                        entry['message'] = json_message(msg)
                        if tum:
                            pos, q = msg.pose.pose.position, msg.pose.pose.orientation
                            timestamp = f'{ref.stamp_ns // 1_000_000_000}.{ref.stamp_ns % 1_000_000_000:09d}'
                            xyz = np.array([pos.x, pos.y, pos.z]) * args.pose_translation_scale
                            tum.write(timestamp + ' ' + ' '.join(f'{v:.12g}' for v in (*xyz, q.x, q.y, q.z, q.w)) + '\n')
                    elif kind == 'clouds':
                        path = directory / f'{ref.key}.npz'
                        save_raw_cloud(path, msg)
                        entry.update(file=path.name, frame_id=msg.header.frame_id)
                    else:
                        path = directory / f'{ref.key}.npy'
                        np.save(path, decode_image(msg), allow_pickle=False)
                        entry.update(file=path.name, encoding=msg.encoding, frame_id=msg.header.frame_id,
                                     unit='native recorded units; no conversion applied')
                    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
            finally:
                if tum:
                    tum.close()
        exports[topic] = dict(kind=kind, count=len(messages), index=str((directory / 'messages.jsonl').relative_to(output)))
    return exports


def validate_outputs(output, camera):
    count, valid_pixels = 0, []
    max_roundtrip = 0.0
    with (output / 'frames.jsonl').open() as stream:
        for line in stream:
            frame = json.loads(line)
            for key, value in frame['files'].items():
                if not (output / value).is_file():
                    raise ValueError(f'Missing output file: {key} {value}')
            image = cv2.imread(str(output / frame['files']['image_rectified']))
            if image is None or image.shape[:2] != (camera.height, camera.width):
                raise ValueError('Exported rectified image has invalid shape')
            depth = np.load(output / frame['files']['depth'], allow_pickle=False)
            if depth.shape != (camera.height, camera.width) or depth.dtype != np.float32:
                raise ValueError('Exported depth has invalid shape/type')
            if np.isinf(depth).any() or (depth[np.isfinite(depth)] <= 0).any():
                raise ValueError('Exported depth contains invalid positive-Z values')
            with np.load(output / frame['files']['cloud_aligned'], allow_pickle=False) as cloud:
                xyz = cloud['xyz']
                if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or (xyz[:, 2] <= 0).any():
                    raise ValueError('Invalid aligned cloud')
                reconstructed = rasterize(camera.project_rectified(xyz), xyz, camera.width, camera.height, camera.valid)
                if not np.allclose(depth, reconstructed, equal_nan=True, atol=1e-5):
                    raise ValueError('Saved aligned cloud does not reproduce the saved sparse depth')
                if len(xyz):
                    uv = camera.project_rectified(xyz)
                    # Independent pinhole unprojection checks the exported camera-Z convention.
                    back = np.column_stack(((uv[:, 0] - camera.k[0, 2]) / camera.k[0, 0] * xyz[:, 2],
                                            (uv[:, 1] - camera.k[1, 2]) / camera.k[1, 1] * xyz[:, 2], xyz[:, 2]))
                    max_roundtrip = max(max_roundtrip, float(np.abs(back - xyz).max()))
            validate_transform(frame['T_world_camera'], 'saved T_world_camera')
            count += 1
            valid_pixels.append(int(np.isfinite(depth).sum()))
    return dict(passed=True, checked_samples=count, depth_pixels=distribution(valid_pixels),
                pinhole_roundtrip_max_error_m=max_roundtrip,
                scope='File integrity, transforms, and numerical geometry only; visual alignment is a separate review')


def run(args):
    start_time = time.monotonic()
    args._explicit_image_topic = args.image_topic is not None
    validate_args(args)
    output = args.output.resolve()
    bag = SqliteBag(args.input)
    report = None
    try:
        calib_path, camera, t_base_lidar = calibrate(args, bag)
        wanted = select_topics(args, bag)
        print(f'Indexing sensor headers from {len(bag.files)} split files (no image/cloud payload scan)', flush=True)
        refs = bag.index(wanted)
        if not args._explicit_image_topic and not refs.get(args.image_topic) and refs.get('/odin1/image'):
            args.image_topic = '/odin1/image'
        for topic, types in ((args.image_topic, ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')),
                             (args.cloud_topic, ('sensor_msgs/msg/PointCloud2',))):
            if not refs.get(topic):
                raise ValueError(f'No messages on required topic: {topic}')
            if bag.topics[topic] not in types:
                raise ValueError(f'Unexpected type on {topic}: {bag.topics[topic]}')
        tracks = load_tracks(args, refs, bag)
        images, clouds = refs[args.image_topic], refs[args.cloud_topic]
        image_origin_ns = images[0].stamp_ns
        candidates = [r for r in images if r.stamp_ns >= image_origin_ns + round(args.start * 1e9) and
                      (args.end is None or r.stamp_ns <= image_origin_ns + round(args.end * 1e9))][::args.stride]
        if not candidates:
            raise ValueError('No images in requested time interval')
        # Limit candidates after successful synchronization, so --max-frames means exported samples.
        cloud_times = np.array([r.stamp_ns for r in clouds], dtype=np.int64)
        output.mkdir(parents=True, exist_ok=False)
        for sub in ('images_original', 'images_rectified', 'clouds_raw', 'clouds_aligned', 'depth'):
            (output / sub).mkdir()
        shutil.copyfile(calib_path, output / 'calib_original.yaml')
        write_image(output / 'rectified_valid_mask.png', camera.valid.astype(np.uint8) * 255)
        driver_commit = bag.directory.parent / 'driver_commit.txt'
        calibration = dict(
            schema_version=1, raw_camera=camera.cam, K_rectified=camera.k.tolist(),
            rectified_size=[args.width, args.height], rectified_horizontal_fov_deg=args.fov,
            rectified_valid_mask='rectified_valid_mask.png',
            T_camera_lidar=camera.t_camera_lidar.tolist(), T_base_lidar=t_base_lidar.tolist(),
            world_frame=tracks[0].frame, base_frame=tracks[0].child, camera_frame='camera_optical',
            conventions=dict(transform='T_A_B maps column-vector coordinates from B to A',
                             pose='T_world_camera maps camera to world at image header time',
                             camera_axes='x right, y down, z forward', distance_unit='meter',
                             timestamp_unit='integer nanosecond, sensor header clock; not Unix time',
                             quaternion_order='xyzw', depth='camera Z in meters; NaN missing'),
            assumptions=dict(Tcl_0='camera <- lidar, following Odin1 calibration convention',
                             pose_translation='ROS SI meters multiplied by pose_translation_scale',
                             pose_translation_scale=args.pose_translation_scale,
                             cloud_frame='same as pose child unless an explicit T_base_lidar is supplied',
                             rectification='same optical axes, no additional rotation',
                             point_time_semantics='operator-confirmed, not inferred from offset magnitude' if args.motion_compensation == 'point'
                             else 'not used; default frame compensation does not deskew within a scan'),
            recorded_driver_commit=driver_commit.read_text().strip() if driver_commit.exists() else None,
            projection_reference='https://github.com/manifoldsdk/odin_ros_driver/blob/main/include/polynomial_camera.hpp')
        dump_json(output / 'calibration.json', calibration)
        options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')}
        report = dict(schema_version=1, status='running', input=str(bag.directory), options=options,
                      input_files=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in bag.files],
                      indexed_topics={k: len(v) for k, v in refs.items()},
                      input_images=len(images), input_clouds=len(clouds), candidate_images=len(candidates),
                      pose_tracks=[t.summary() for t in tracks],
                      pose_track_agreement=track_agreement(tracks, round(args.max_pose_gap_ms * 1e6)),
                      selected_samples=0, skipped={}, previews=[], extras={},
                      limitations=['Frame compensation treats each raw scan as acquired at its header time; '
                                   'within-scan distortion remains unless explicit point mode is used.',
                                   'ROS SI odometry units and recorded frame labels are assumed; no unverified scale is inferred.',
                                   'Dynamic objects and calibration/time offsets can still cause image-cloud mismatch.'])
        report['dependencies'] = {}
        for dep in ('numpy', 'scipy', 'rosbags', 'PyYAML'):
            report['dependencies'][dep] = importlib.metadata.version(dep)
        report['dependencies']['opencv'] = cv2.__version__
        dump_json(output / 'report.json', report)
        selected, skipped, deltas, used, pose_sources = 0, Counter(), [], Counter(), Counter()
        cached_ref = cached_msg = cached_points = None
        with (output / 'frames.jsonl').open('w') as frames, (output / 'skipped.jsonl').open('w') as skip_stream:
            for candidate_index, image_ref in enumerate(candidates):
                cloud_ref = nearest_ref(clouds, cloud_times, image_ref.stamp_ns)
                delta = cloud_ref.stamp_ns - image_ref.stamp_ns
                reason = None
                if abs(delta) > round(args.max_delta_ms * 1e6):
                    reason = 'cloud_time_delta_exceeds_limit'
                else:
                    if cached_ref != cloud_ref:
                        cached_ref, cached_msg = cloud_ref, bag.read(cloud_ref)
                        cached_points = point_array(cached_msg)
                    try:
                        xyz, source_indices, uncompensated, pose, stats, pose_provenance = process_geometry(
                            args, cached_msg, cached_points, image_ref.stamp_ns, tracks, camera, t_base_lidar)
                        if not len(xyz):
                            reason = 'no_valid_camera_front_points'
                    except PoseUnavailable as exc:
                        reason = 'pose_unavailable'
                        pose_reason = str(exc)
                if reason:
                    skipped[reason] += 1
                    entry = dict(image=image_ref.provenance(bag.files), reason=reason, cloud_image_delta_ns=delta)
                    if reason == 'pose_unavailable':
                        entry['detail'] = pose_reason
                    skip_stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    continue
                image_msg = bag.read(image_ref)
                image = decode_image(image_msg)
                if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                    raise ValueError('Model RGB source must decode to a three-channel uint8 image')
                rectified = camera.rectify(image)
                name = f'{selected:06d}_{image_ref.stamp_ns}'
                original_path = save_original(output / 'images_original' / name, image_msg, image)
                rectified_path = output / 'images_rectified' / f'{name}.png'
                write_image(rectified_path, rectified)
                raw_path = output / 'clouds_raw' / f'{cloud_ref.key}_{cloud_ref.stamp_ns}.npz'
                if not raw_path.exists():
                    save_raw_cloud(raw_path, cached_msg, cached_points)
                aligned_path = output / 'clouds_aligned' / f'{name}.npz'
                # Rasterize the actual stored float32 XYZ so roundoff cannot change the saved pixel assignments.
                xyz = xyz.astype(np.float32)
                points_data = dict(xyz=xyz, source_indices=source_indices.astype(np.int64),
                                   timestamp_ns=np.int64(image_ref.stamp_ns),
                                   source_timestamp_ns=np.int64(cloud_ref.stamp_ns), frame_id='camera_optical')
                for field in cached_points.dtype.names:
                    if field not in ('x', 'y', 'z'):
                        points_data[f'source_{field}'] = cached_points.reshape(-1)[field][source_indices]
                np.savez_compressed(aligned_path, **points_data)
                depth = rasterize(camera.project_rectified(xyz), xyz, args.width, args.height, camera.valid)
                depth_path = output / 'depth' / f'{name}.npy'
                np.save(depth_path, depth, allow_pickle=False)
                files = dict(image_original=original_path, image_rectified=rectified_path,
                             cloud_raw=raw_path, cloud_aligned=aligned_path, depth=depth_path)
                if args.ply:
                    ply_path = aligned_path.with_suffix('.ply')
                    save_ply(ply_path, xyz)
                    files['cloud_ply'] = ply_path
                stats['depth_pixels'] = int(np.isfinite(depth).sum())
                entry = dict(schema_version=1, id=name, image_timestamp_ns=image_ref.stamp_ns,
                             cloud_timestamp_ns=cloud_ref.stamp_ns, cloud_image_delta_ns=delta,
                             image_source=image_ref.provenance(bag.files), cloud_source=cloud_ref.provenance(bag.files),
                             image_frame_id=image_msg.header.frame_id, cloud_frame_id=cached_msg.header.frame_id,
                             world_frame=tracks[0].frame, T_world_camera=pose.tolist(),
                             motion_compensation=args.motion_compensation, pose_source=pose_provenance,
                             files={k: str(v.relative_to(output)) for k, v in files.items()}, stats=stats)
                frames.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
                frames.flush()
                selected += 1
                used[cloud_ref.key] += 1
                pose_sources[pose_provenance['topic']] += 1
                deltas.append(delta / 1e6)
                report.update(selected_samples=selected, skipped=dict(skipped))
                if selected == 1 or selected % 25 == 0:
                    print(f'Exported {selected} samples; candidate {candidate_index + 1}/{len(candidates)}; '
                          f'delta={delta / 1e6:+.3f} ms; depth pixels={stats["depth_pixels"]}', flush=True)
                    dump_json(output / 'report.json', report)
                if args.max_frames and selected >= args.max_frames:
                    break
        if selected == 0:
            report['skipped'] = dict(skipped)
            raise ValueError('No synchronized samples exported; inspect skipped.jsonl and pose/time thresholds')
        report.update(selected_samples=selected, skipped=dict(skipped),
                      cloud_image_delta_ms=distribution(deltas), unique_source_clouds=len(used),
                      reused_cloud_assignments=selected - len(used), pose_source_counts=dict(pose_sources))
        report['extras'] = export_extras(args, bag, refs, output)
        print('Generating projection previews from successful samples...', flush=True)
        report['previews'] = generate_previews(output, camera, args.preview_count)
        print('Checking saved sample files and geometry...', flush=True)
        report['validation'] = validate_outputs(output, camera)
        report.update(status='complete', elapsed_seconds=time.monotonic() - start_time)
        dump_json(output / 'report.json', report)
        print(f'Done: {selected} samples, {sum(skipped.values())} skipped; {output}', flush=True)
        return report
    except BaseException as exc:
        if report is not None:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                          elapsed_seconds=time.monotonic() - start_time)
            dump_json(output / 'report.json', report)
        raise
    finally:
        bag.close()


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
