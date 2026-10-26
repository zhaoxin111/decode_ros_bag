# Odin1 ROS bag 解析与对齐

将 ROS 2 SQLite3 bag 导出为可直接读取的图片、点云、相机位姿、内外参和稀疏深度。支持录制目录、`bag/` 目录和拆分 `.db3` 文件；不需要启动 ROS，也不依赖相邻的模型仓库。

## 安装与运行

Python 3.10+：

```bash
python -m pip install -r requirements.txt
python decode_bag.py \
  --input /media/xzhao/nas163/data_xzhao/odin1/odin_bags/20261026_144347 \
  --output outputs/20261026_144347
```

输入目录中的 `bag/` 会自动识别；标定从 bag 目录或其父目录的 `calib.yaml` 加载，也可通过 `--calib` 指定。现有输出目录会被拒绝覆盖；发生错误时保留已经导出的文件，并在已建立的 `report.json` 中标记失败。

先抽样检查前、中、后场景：

```bash
python decode_bag.py \
  --input /media/xzhao/nas163/data_xzhao/odin1/odin_bags/20261026_144347 \
  --output outputs/sample_check \
  --stride 89 --max-frames 8 --preview-count 8 --ply
python load_example.py outputs/sample_check
```

默认只导出成功对齐的模型输入及其原始来源。时间索引只读取消息头；选中消息通过 SQLite 主键读取。内存中只保留小型索引、轨迹和当前帧，不整包加载图片和点云。

## 输出与读取

| 文件/目录 | 内容 |
| --- | --- |
| `frames.jsonl` | 一行一个成功样本，包含相对文件路径、整数纳秒时间戳、配对误差、位姿和来源 |
| `images_original/` | 原始鱼眼图片；JPEG/PNG 压缩流直接保存，不重新编码 |
| `images_rectified/` | 针孔图 PNG，默认 800×648、水平 FoV 90° |
| `clouds_raw/` | 原始 PointCloud2 NPZ；同一源帧被复用时只保存一份 |
| `clouds_aligned/` | 图片时刻相机光学坐标下的点云 NPZ，`--ply` 可额外输出 PLY |
| `depth/` | float32 NPY，相机 Z 深度（米），无效值为 NaN |
| `calibration.json` | 原始鱼眼参数、针孔内参、外参、坐标和单位约定 |
| `calib_original.yaml` | 输入标定文件副本 |
| `rectified_valid_mask.png` | 去畸变后有效图像区域，255 有效、0 无效 |
| `previews/` | 原图投影、针孔图投影和补偿前后对比；默认抽样至多 8 帧，包含首尾及运动较明显的样本 |
| `skipped.jsonl` | 跳过图片的来源及原因 |
| `report.json` | 参数、依赖、输入文件信息、时间和轨迹统计、导出数量及保存后校验 |

原始 NPZ 中的 `points` 为 `(height, width)` 结构化数组，保留字段类型及 count；`data` 为原始 PointCloud2 字节。`fields_json`、`point_step`、`row_step`、`is_bigendian` 等字段可恢复完整消息布局。浮点 `rgb` 字段保持原始位编码，不当作普通浮点颜色转换。

对齐 NPZ 中的 `xyz` 为 `(N, 3)` float32，包含所有有限、相机正 Z 的有效点，包括针孔视场外的点；`source_indices` 为原始点云按行展平后的索引。其余属性以 `source_intensity`、`source_confidence`、`source_offset_time` 等名称保留，仍表示**源点属性**。筛选不会改写原始点云。

最小读取示例：

```python
from load_example import iter_samples

for sample in iter_samples("outputs/20261026_144347"):
    rgb = sample["rgb"]                   # H×W×3 uint8 RGB
    xyz = sample["xyz_camera"]            # N×3 float32，米
    sparse_depth = sample["sparse_depth"] # H×W float32，相机 Z，缺失 NaN
    K = sample["K"]                       # 3×3 针孔内参
    T_wc = sample["T_world_camera"]        # 4×4 相机到世界
    xyz_world = xyz @ T_wc[:3, :3].T + T_wc[:3, 3]
    break
```

模型要求缺失深度为零时，在模型输入适配处使用 `np.nan_to_num(sparse_depth, nan=0.0)`。保存的深度不插值、不做深度补全。

## 时间、坐标与运动补偿

- 使用传感器 **header 时间**对齐；bag 写入时间仅用于溯源。本次录制使用设备启动后的时间轴，不能按 Unix 时间解释。
- 以图片为基准选择最近点云，最大时间差默认 50 ms；等距时选择较早点云，允许点云复用。重复时间戳的图片仍有独立样本 ID。
- 默认优先 `/odin1/odometry_highfreq`，在无覆盖或插值间隔超过 250 ms 时尝试 `/odin1/odometry`。每个样本的所有查询时刻使用同一路轨迹，来源和插值区间写入索引；两路均不可用时跳过，禁止外推。轨迹内重复时间戳保留第一条并报告计数。
- 平移线性插值，旋转 SLERP。位姿按 ROS SI 约定解释为米；可用 `--pose-translation-scale` 显式换算其他单位，不根据数值大小自动猜单位。
- `T_A_B` 把 B 中的列向量变换到 A。`Tcl_0` 按 Odin1 标定约定为 `T_camera_lidar`；相机光学坐标为 X 右、Y 下、Z 前。
- 默认点云 frame_id 必须与里程计 child_frame_id 相同。若设备实际具有额外刚体变换，使用 `--base-lidar-transform` 指定 YAML 4×4 矩阵 `T_base_lidar`。必须是已核实的变换，不能仅靠修改标签绕过几何问题。
- 相机位姿为 `T_world_camera(t_image) = T_world_base(t_image) × T_base_lidar × inverse(T_camera_lidar)`。默认去畸变不旋转相机坐标轴。
- 默认帧间补偿为：

  `p_camera = T_camera_lidar × inverse(T_world_lidar(t_image)) × T_world_lidar(t_cloud) × p_lidar`

默认过滤源点云中非有限坐标、距离小于 0.1 m 和置信度低于 35 的点。没有 confidence 字段时只应用坐标与距离筛选，并在统计中注明。投影采用最近 Z 缓冲解决同像素冲突，视场外和无效图像区域不生成深度。

FishPoly 使用 `theta_d = theta + k2*theta² + … + k7*theta⁷` 和 A12 skew，参照[厂家实现](https://github.com/manifoldsdk/odin_ros_driver/blob/main/include/polynomial_camera.hpp)。与 OpenCV 四参数鱼眼模型不同；当前仅支持此模型、单相机和零切向畸变，遇到不支持的标定会报错。

### 逐点补偿

默认 `--motion-compensation frame` 只补偿图片与点云帧头的时间差，**不消除扫描内部运动畸变**。本次样本中 offset_time 覆盖约 94 ms，帧头时间差小并不表示所有点与图片同时采集。

逐点模式已实现，但必须先从实际驱动/SDK 核实 offset_time 的单位、帧头是扫描起点还是终点，以及原始 XYZ 是否已经去运动畸变。脚本不根据字段范围自动启用。确认后可运行，例如：

```bash
# 仅当已确认 offset_time 是秒、帧头代表扫描起点且 XYZ 尚未补偿时使用
python decode_bag.py --input RECORDING --output outputs/point_compensated \
  --motion-compensation point --point-time-reference start --point-time-unit s
```

`start` 使用 `t_point = t_header + offset`；`end` 使用 `t_point = t_header + offset - max(offset)`，即明确约定帧头为当前点集最后一个点的时间。驱动采用其他定义时需先转换时间，不能直接套用。逐点模式要求整个扫描的时间范围均有有效位姿，不静默退回帧间补偿。时间参数是操作者确认的约定，并非工具对该录制驱动语义的认证。

`--motion-compensation none` 可用于对照，仍保存图片时刻的相机位姿。

## 可选数据与常用参数

其他数据默认不导出，按需启用：

```bash
python decode_bag.py --input RECORDING --output outputs/all_products \
  --export-imu --export-tf --export-trajectories \
  --export-extra-clouds --export-depth-products
```

- IMU、TF 和完整里程计保存为包含来源信息的 JSONL；轨迹另附 TUM 格式（位置换算到米，四元数 xyzw）。JSONL 保留原始消息单位、协方差及速度。
- 其他 PointCloud2 保存为原始 NPZ，包括 SLAM 和深度补全点云；保留各自原始 frame_id，不擅自变换到相机。
- 录制的非 RGB Image 产品保存 NPY 和 encoding 元信息，不猜测深度单位。
- 可选导出覆盖**整包**；`--start/--end/--stride/--max-frames` 仅作用于模型样本。topic 在路径中采用百分号编码，避免不同名称碰撞。

| 参数 | 默认/用途 |
| --- | --- |
| `--image-topic` | 优先 `/odin1/image/compressed`，不存在或为空则用 `/odin1/image` |
| `--cloud-topic` | `/odin1/cloud_raw` |
| `--pose-topic` | `/odin1/odometry_highfreq` |
| `--fallback-pose-topic` | `/odin1/odometry`；传空字符串可关闭 |
| `--width --height --fov` | `800 648 90`，fov 为水平角度 |
| `--start --end` | 相对第一张图片 header 时间的秒数，闭区间 |
| `--stride --max-frames` | 图片采样步长、成功样本数上限 |
| `--max-delta-ms` | 最大图片/点云帧头误差，默认 50 |
| `--max-pose-gap-ms` | 最大位姿插值间隔，默认 250 |
| `--confidence --min-range` | 默认 `35 0.1` |
| `--preview-count` | 默认 8；0 关闭投影预览 |
| `--ply` | 额外导出二进制 PLY，默认关闭 |

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖带行填充、非对齐偏移、多元素字段和大端点云解码，图像编码、位姿时间覆盖和 SLERP、已知运动与外参方向、逐点时间约定、FishPoly、最近 Z、拆分 bag 全流程、可选导出及覆盖保护。

每次导出后会重新读取保存的文件，验证路径、尺寸、深度、正 Z 点云和位姿；要求已保存点云能重建相同稀疏深度。`report.json` 中的校验只证明文件和数值流程一致，真实标定、动态物体和扫描内部时序仍需结合 `previews/` 观察。

本次指定数据的实测统计和检查结论见 [VALIDATION.md](VALIDATION.md)。
