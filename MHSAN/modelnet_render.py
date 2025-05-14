import os
import numpy as np
import trimesh
import pyrender
import cv2
import glob
os.environ["PYOPENGL_PLATFORM"] = "egl"

def get_look_at(eye, target=np.array([0,0,0]), up=np.array([0,0,1])):
    # 計算相機座標系的 z 軸（由目標指向眼睛）
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    # x 軸：利用上方向與 z 軸叉乘
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # y 軸：叉乘取得正確的上方向
    y_axis = np.cross(z_axis, x_axis)
    
    mat = np.eye(4)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat

def get_camera_positions(num_views):
    if num_views == 12:
        # **12 視角 (30° 高度，360° 均勻分佈)**
        angles = np.linspace(0, 360, num_views, endpoint=False)
        positions = []
        for angle in angles:
            x = np.cos(np.radians(angle))
            y = np.sin(np.radians(angle))
            positions.append([x, y, np.tan(np.radians(30))])  # 30° 高度
        return np.array(positions)
    elif num_views == 20:
        # **20 視角 (來自十二面體的 20 頂點)**
        dodecahedron_vertices = np.array([
            [0.607, 0.000, 0.795], [0.188, 0.577, 0.795], [-0.491, 0.357, 0.795], 
            [-0.491, -0.357, 0.795], [0.188, -0.577, 0.795], [0.982, 0.000, 0.188], 
            [0.304, 0.934, 0.188], [-0.795, 0.577, 0.188], [-0.795, -0.577, 0.188], 
            [0.304, -0.934, 0.188], [0.795, 0.577, -0.188], [-0.304, 0.934, -0.188], 
            [-0.982, 0.000, -0.188], [-0.304, -0.934, -0.188], [0.795, -0.577, -0.188], 
            [0.491, 0.357, -0.795], [-0.188, 0.577, -0.795], [-0.607, 0.000, -0.795], 
            [-0.188, -0.577, -0.795], [0.491, -0.357, -0.795]
        ])
        return dodecahedron_vertices

def render_views(off_path, output_folder, num_views=12):
    os.makedirs(output_folder, exist_ok=True)

    # 載入 3D 物件並確保重心對齊
    mesh = trimesh.load_mesh(off_path, process=True, maintain_order=True)
    if mesh.vertices.shape[0] == 0:
        print(f"警告: 檔案 {off_path} 無頂點資料，跳過。")
        return
    mesh.fix_normals()

    # 用 bounding box 的中心作為旋轉中心
    centroid = mesh.bounding_box.centroid
    mesh.apply_translation(-centroid)

    # 計算最大尺寸以縮放至標準大小
    try:
        bbox = mesh.bounding_box_oriented.extents
    except Exception as e:
        print(f"警告: 無法計算 {off_path} 的 bounding_box 錯誤: {e}")
        return
    max_dim = np.max(bbox)
    mesh.apply_scale(1.0 / max_dim)  # 縮放至標準大小

    # 建立 Scene，移除背景色設定（採用預設）
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # 加入光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=np.eye(4))
    
    # 設定相機參數
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    renderer = pyrender.OffscreenRenderer(224, 224)  # 設定影像解析度

    # 取得相機位置
    positions = get_camera_positions(num_views)
    for i, pos in enumerate(positions):
        # 調整相機距離倍率從 2.0 改成 1.5，使物件看起來較大
        eye = pos * 1 
        camera_pose = get_look_at(eye, target=np.array([0, 0, 0]), up=np.array([0, 0, 1]))
        camera_node = scene.add(camera, pose=camera_pose)

        # 渲染影像
        color, _ = renderer.render(scene)
        img_path = os.path.join(output_folder, f"view_{i}.png")
        cv2.imwrite(img_path, color)

        # 移除相機節點
        scene.remove_node(camera_node)

    renderer.delete()


if __name__ == "__main__":
    data_root = "modelnet40-princeton-3d-object-dataset/versions/1/ModelNet40"
    categories = os.listdir(data_root)

    for category in categories:
        category_path = os.path.join(data_root, category)
        for split in ["train", "test"]:
            split_path = os.path.join(category_path, split)
            off_files = glob.glob(os.path.join(split_path, "*.off"))
            print(f"類別: {category}, {split} 集共有 {len(off_files)} 個模型")

    # 設定視角張數
    num_views = 20
    # 設定影像儲存路徑
    output_root = f"modelnet40-princeton-3d-object-dataset/rendered_views_{num_views}"
    os.makedirs(output_root, exist_ok=True)

    for category in categories:
        category_path = os.path.join(data_root, category)
        for split in ["train", "test"]:
            split_path = os.path.join(category_path, split)
            off_files = glob.glob(os.path.join(split_path, "*.off"))
            for off_file in off_files:
                obj_name = os.path.basename(off_file).replace(".off", "")
                output_folder = os.path.join(output_root, category, split, obj_name)
                os.makedirs(output_folder, exist_ok=True)
                
                # 渲染多視角影像
                render_views(off_file, output_folder, num_views)
                print('已完成', off_file)
