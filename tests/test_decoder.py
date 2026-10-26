"""Numerical geometry and an end-to-end synthetic split ROS 2 bag."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace as NS
import unittest

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation
import yaml

from bag_decoder import (FishPoly, PoseTrack, PoseUnavailable, SqliteBag, choose_track,
                         decode_image, point_array, rasterize, transform_points)
from decode_bag import parser, point_times, process_geometry, run


def calibration():
    return dict(Tcl_0=np.eye(4).reshape(-1).tolist(), cam_0=dict(
        cam_model='FishPoly', image_width=16, image_height=12, A11=8., A12=-.2,
        A22=8., u0=7.5, v0=5.5, p1=0, p2=0,
        **{f'k{n}': .001 * (-1) ** n for n in range(2, 8)}))


def fake_odom(t, x=0., yaw=0.):
    q = Rotation.from_euler('z', yaw, degrees=True).as_quat()
    return NS(header=NS(stamp=NS(sec=t // 10**9, nanosec=t % 10**9), frame_id='world'),
              child_frame_id='lidar', pose=NS(pose=NS(position=NS(x=x, y=0., z=0.),
                                                    orientation=NS(**dict(zip(('x', 'y', 'z', 'w'), q))))))


class FakeBag:
    def __init__(self, msgs):
        self.msgs = msgs

    def read(self, ref):
        return self.msgs[ref.stamp_ns]


def track(times, xs, yaws=None, topic='/pose'):
    yaws = [0] * len(times) if yaws is None else yaws
    refs = [NS(stamp_ns=t, topic=topic) for t in times]
    return PoseTrack(refs, FakeBag({t: fake_odom(t, x, yaw) for t, x, yaw in zip(times, xs, yaws)}))


class GeometryTests(unittest.TestCase):
    def test_pointcloud_big_endian_padding_and_field_counts(self):
        dtype = np.dtype(dict(names=['x', 'y', 'z', 'vector'],
                              formats=['>f4', '>f4', '>f4', ('>u2', (2,))],
                              offsets=[0, 4, 8, 13], itemsize=19))
        data = bytearray(2 * 44)
        source = np.ndarray((2, 2), dtype=dtype, buffer=data, strides=(44, 19))
        source['x'] = [[1, 2], [3, 4]]
        source['y'] = 5
        source['z'] = 6
        source['vector'] = [35, 72]
        fields = [NS(name=n, offset=o, datatype=d, count=c) for n, o, d, c in
                  [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1), ('vector', 13, 4, 2)]]
        msg = NS(fields=fields, is_bigendian=True, point_step=19, row_step=44,
                 width=2, height=2, data=data)
        decoded = point_array(msg)
        np.testing.assert_array_equal(decoded['x'], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(decoded['vector'][1, 1], [35, 72])
        msg.row_step = 37
        with self.assertRaises(ValueError):
            point_array(msg)

    def test_image_padding_endian_and_rgb_order(self):
        msg = NS(__msgtype__='sensor_msgs/msg/Image', encoding='rgb8', height=1, width=2,
                 is_bigendian=False, step=8, data=np.array([255, 0, 0, 0, 255, 0, 99, 99], dtype='u1'))
        np.testing.assert_array_equal(decode_image(msg), [[[0, 0, 255], [0, 255, 0]]])
        msg.encoding, msg.is_bigendian, msg.step = '16UC1', True, 6
        msg.data = bytes([0, 1, 1, 0, 99, 99])
        np.testing.assert_array_equal(decode_image(msg), [[1, 256]])

    def test_pose_slerp_large_integer_clock_and_coverage(self):
        t = 1_792_997_028_000_000_000
        tr = track([t, t + 100_000_000], [0, 2], [0, 90])
        matrices, brackets = tr.evaluate([t + 50_000_000], 200_000_000)
        np.testing.assert_allclose(matrices[0, :3, 3], [1, 0, 0])
        np.testing.assert_allclose(matrices[0, :3, :3] @ [1, 0, 0], [2**-.5, 2**-.5, 0], atol=1e-10)
        self.assertEqual(brackets.tolist(), [[t, t + 100_000_000]])
        for query, gap in [([t - 1], 200_000_000), ([t + 50_000_000], 10_000_000)]:
            with self.assertRaises(PoseUnavailable):
                tr.evaluate(query, gap)
        # Exact poses do not need an adjacent interval, even next to a long gap.
        tr.evaluate([t], 1)
        fallback = track([t, t + 25_000_000, t + 50_000_000], [0, 0.5, 1], topic='/fallback')
        used, _, _ = choose_track([tr, fallback], [t + 25_000_000], 30_000_000)
        self.assertEqual(used.topic, '/fallback')

    def test_fishpoly_independent_projection_and_rectification(self):
        cam = FishPoly(calibration(), 16, 12, 90)
        rays = np.array([[0., 0., 1.], [.2, -.4, 1.], [-.8, .3, 1.]])
        actual = cam.project_original(rays)
        expected = []
        for x, y, z in rays:
            r = np.hypot(x, y)
            theta = np.arctan2(r, z)
            radius = sum(([theta] + [cam.cam[f'k{n}'] * theta**n for n in range(2, 8)]))
            xd, yd = (x * radius / r, y * radius / r) if r else (0, 0)
            expected.append([8 * xd - .2 * yd + 7.5, 8 * yd + 5.5])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        uv = cam.project_rectified(rays)
        back = np.column_stack(((uv[:, 0] - 7.5) / 8, (uv[:, 1] - 5.5) / 8, np.ones(3)))
        np.testing.assert_allclose(back, rays, atol=1e-12)
        # Bilinear lookup of independently projected rays through the rectification map.
        y, x = np.mgrid[:12, :16]
        dense = np.column_stack(((x.ravel() - 7.5) / 8, (y.ravel() - 5.5) / 8, np.ones(192)))
        projected = cam.project_original(dense).reshape(12, 16, 2)
        np.testing.assert_allclose(cam.map_x, projected[..., 0], atol=1e-6)
        np.testing.assert_allclose(cam.map_y, projected[..., 1], atol=1e-6)

    def test_raster_z_buffer_and_mask(self):
        xyz = np.array([[0, 0, 5], [0, 0, 2], [1, 1, -1], [0, 0, np.nan], [0, 0, 1]])
        uv = np.array([[1.1, 1], [1.2, 1], [0, 0], [0, 0], [3, 0]])
        result = rasterize(uv, xyz, 3, 3)
        self.assertEqual(result[1, 1], 2)
        self.assertEqual(np.isfinite(result).sum(), 1)
        mask = np.ones((3, 3), bool)
        mask[1, 1] = False
        self.assertFalse(np.isfinite(rasterize(uv, xyz, 3, 3, mask)).any())

    def test_known_motion_extrinsic_and_point_timing(self):
        cam = FishPoly(calibration(), 16, 12, 90)
        cam.t_camera_lidar[:3, 3] = [.2, 0, .1]
        tr = track([0, 1_000_000_000, 2_000_000_000], [0., 1., 2.])
        msg = NS(header=NS(stamp=NS(sec=0, nanosec=0), frame_id='lidar'))
        points = np.zeros((1, 2), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('confidence', 'u2'), ('offset_time', 'f4')])
        points['z'] = 5
        points['confidence'] = 100
        points['offset_time'] = [0, 1]
        args = parser().parse_args(['--input', '/tmp/no', '--output', '/tmp/no', '--max-pose-gap-ms', '1500'])
        aligned, _, _, pose, _, _ = process_geometry(args, msg, points, 1_000_000_000, [tr], cam, np.eye(4))
        # World point at scan start x=0, camera at image x=0.8 -> camera point x=-0.8.
        np.testing.assert_allclose(aligned, [[-.8, 0, 5.1], [-.8, 0, 5.1]])
        np.testing.assert_allclose(pose[:3, 3], [.8, 0, -.1])
        np.testing.assert_allclose(transform_points(pose, aligned), [[0, 0, 5], [0, 0, 5]])
        args.motion_compensation, args.point_time_reference, args.point_time_unit = 'point', 'start', 's'
        aligned, _, _, _, _, _ = process_geometry(args, msg, points, 1_000_000_000, [tr], cam, np.eye(4))
        np.testing.assert_allclose(aligned, [[-.8, 0, 5.1], [.2, 0, 5.1]])
        args.point_time_reference = 'end'
        np.testing.assert_array_equal(point_times(msg, points, args), [-1_000_000_000, 0])
        msg.header.frame_id = 'unverified'
        args.motion_compensation = 'frame'
        with self.assertRaisesRegex(ValueError, 'Cloud frame'):
            process_geometry(args, msg, points, 1_000_000_000, [tr], cam, np.eye(4))


class IntegrationTests(unittest.TestCase):
    def make_bag(self, root):
        ts = get_typestore(Stores.ROS2_HUMBLE)
        t = ts.types
        def header(ns, frame='lidar'):
            return t['std_msgs/msg/Header'](t['builtin_interfaces/msg/Time'](ns // 10**9, ns % 10**9), frame)
        def odom(ns):
            pos = t['geometry_msgs/msg/Point'](float(ns - 10**9) * 1e-9, 0., 0.)
            quat = t['geometry_msgs/msg/Quaternion'](0., 0., 0., 1.)
            pose = t['geometry_msgs/msg/PoseWithCovariance'](t['geometry_msgs/msg/Pose'](pos, quat), np.zeros(36))
            vector = t['geometry_msgs/msg/Vector3'](0., 0., 0.)
            twist = t['geometry_msgs/msg/TwistWithCovariance'](t['geometry_msgs/msg/Twist'](vector, vector), np.zeros(36))
            return t['nav_msgs/msg/Odometry'](header(ns, 'world'), 'lidar', pose, twist)
        image = np.zeros((12, 16, 3), np.uint8)
        image[:, :8, 2] = 255
        jpeg = cv2.imencode('.jpg', image)[1]
        xyz = np.array([[0., 0., 5.], [.2, 0., 3.]], dtype='<f4')
        fields = [t['sensor_msgs/msg/PointField'](n, i * 4, 7, 1) for i, n in enumerate(('x', 'y', 'z'))]
        topics = [(1, '/odin1/image/compressed', 'sensor_msgs/msg/CompressedImage', 'cdr'),
                  (2, '/odin1/cloud_raw', 'sensor_msgs/msg/PointCloud2', 'cdr'),
                  (3, '/odin1/odometry', 'nav_msgs/msg/Odometry', 'cdr'),
                  (4, '/imu', 'sensor_msgs/msg/Imu', 'cdr'),
                  (5, '/tf', 'tf2_msgs/msg/TFMessage', 'cdr'),
                  (6, '/extra_cloud', 'sensor_msgs/msg/PointCloud2', 'cdr'),
                  (7, '/depth', 'sensor_msgs/msg/Image', 'cdr')]
        bag = root / 'bag'
        bag.mkdir()
        for split in range(2):
            db = sqlite3.connect(bag / f'bag_{split}.db3')
            db.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT)')
            db.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB)')
            db.executemany('INSERT INTO topics VALUES (?,?,?,?)', topics)
            rows = []
            for i in range(2):
                ns = 10**9 + (2 * split + i) * 100_000_000
                rows.append((3, ns, odom(ns)))
                cloud = t['sensor_msgs/msg/PointCloud2'](header(ns), 1, 2, fields, False, 12, 24,
                                                        np.frombuffer(xyz.tobytes(), dtype='u1'), True)
                rows.append((2, ns, cloud))
                rows.append((1, ns, t['sensor_msgs/msg/CompressedImage'](header(ns, ''), 'jpeg', jpeg)))
                rows.append((6, ns, cloud))
                depth = np.array([[1., np.nan], [3., 4.]], dtype='<f4')
                rows.append((7, ns, t['sensor_msgs/msg/Image'](header(ns), 2, 2, '32FC1', 0, 8,
                                                              np.frombuffer(depth.tobytes(), dtype='u1'))))
                q = t['geometry_msgs/msg/Quaternion'](0., 0., 0., 1.)
                v = t['geometry_msgs/msg/Vector3'](0., 0., 0.)
                rows.append((4, ns, t['sensor_msgs/msg/Imu'](header(ns), q, np.zeros(9), v, np.zeros(9), v, np.zeros(9))))
                transform = t['geometry_msgs/msg/TransformStamped'](header(ns, 'world'), 'lidar',
                                                                    t['geometry_msgs/msg/Transform'](v, q))
                rows.append((5, ns, t['tf2_msgs/msg/TFMessage']([transform])))
            for tid, ns, msg in rows:
                # Deliberately unrelated recorder clock: must never use this time for pairing.
                db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (?,?,?)',
                           (tid, ns + 1_700_000_000_000_000_000 + tid * 1234, bytes(ts.serialize_cdr(msg, msg.__msgtype__))))
            db.commit()
            db.close()
        (root / 'calib.yaml').write_text(yaml.safe_dump(calibration()))

    def test_end_to_end_split_bag_and_optional_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_bag(root)
            out = root / 'out'
            args = parser().parse_args(['--input', str(root), '--output', str(out), '--width', '16', '--height', '12',
                                       '--preview-count', '2', '--ply', '--export-imu', '--export-tf',
                                       '--export-trajectories', '--export-extra-clouds', '--export-depth-products'])
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['selected_samples'], 4)
            self.assertEqual(report['validation']['checked_samples'], 4)
            self.assertEqual(len(report['extras']), 5)
            frames = [json.loads(line) for line in (out / 'frames.jsonl').read_text().splitlines()]
            self.assertTrue(all(f['cloud_image_delta_ns'] == 0 for f in frames))
            self.assertEqual(len(list((out / 'previews').glob('*comparison.jpg'))), 2)
            self.assertTrue(all(f['pose_source']['topic'] == '/odin1/odometry' for f in frames))
            with np.load(out / frames[0]['files']['cloud_raw'], allow_pickle=False) as raw:
                self.assertEqual(raw['points'].shape, (1, 2))
                self.assertEqual(raw['data'].size, 24)
            # Already existing output must be refused even when the bag is otherwise valid.
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(args)
            self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'complete')
            self.assertFalse((out / 'frames.jsonl').read_text().endswith('NaN'))

    def test_default_excludes_optional_exports_and_rejects_unconfirmed_point_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_bag(root)
            args = parser().parse_args(['--input', str(root), '--output', str(root / 'out'),
                                       '--max-frames', '1', '--preview-count', '0'])
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report['extras'], {})
            self.assertFalse((root / 'out' / 'extras').exists())
            args.output = root / 'point_out'
            args.motion_compensation = 'point'
            with self.assertRaisesRegex(ValueError, 'explicit'):
                run(args)
            self.assertFalse(args.output.exists())

    def test_dropped_clouds_and_pose_gaps_record_skips_and_keep_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_bag(root)
            with contextlib.closing(sqlite3.connect(root / 'bag' / 'bag_0.db3')) as db:
                db.execute('DELETE FROM messages WHERE topic_id=2')
                db.commit()
            with contextlib.closing(sqlite3.connect(root / 'bag' / 'bag_1.db3')) as db:
                rows = db.execute('SELECT id FROM messages WHERE topic_id=3 ORDER BY timestamp').fetchall()
                db.execute('DELETE FROM messages WHERE id=?', rows[-1])
                db.commit()
            out = root / 'out'
            args = parser().parse_args(['--input', str(root), '--output', str(out), '--preview-count', '8'])
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report['selected_samples'], 1)
            self.assertEqual(report['skipped'], {'cloud_time_delta_exceeds_limit': 2, 'pose_unavailable': 1})
            self.assertEqual(len(report['previews']), 1)
            self.assertTrue((out / report['previews'][0]).is_file())


if __name__ == '__main__':
    unittest.main()
