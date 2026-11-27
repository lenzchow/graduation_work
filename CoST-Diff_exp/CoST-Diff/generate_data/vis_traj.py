import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import numpy as np
import os

# ====== 1. 遍历文件夹，读取所有轨迹点 ======
folder_path = "./operator0_node2vec_diffT500_trajlen28-120_curTime202506241105"
all_points = []
traj_id = -1

for filename in os.listdir(folder_path):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(folder_path, filename)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line.startswith("#"):
                traj_id += 1
                continue
            parts = line.split(',')
            if len(parts) == 5:
                region_code, lon, lat, base_id, hour = parts
                all_points.append((traj_id, float(lat), float(lon)))

df = pd.DataFrame(all_points, columns=["traj_id", "lat", "lon"])
print("📍 总轨迹点数:", len(df))

# ====== 2. 计算地图中心点 ======
center_lat = df['lat'].mean()
center_lon = df['lon'].mean()

# ====== 3. Haversine 距离函数 ======
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径（km）
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ====== 4. 计算每个轨迹平均距离中心点，筛选前 vis_rate% ======
df['dist_to_center_km'] = haversine(center_lat, center_lon, df['lat'], df['lon'])
vis_rate = 0.3  # 保留离中心点最近的前 30% 轨迹
traj_mean_dist = df.groupby("traj_id")['dist_to_center_km'].mean().reset_index()
threshold = traj_mean_dist['dist_to_center_km'].quantile(vis_rate)
valid_traj_ids = traj_mean_dist[traj_mean_dist['dist_to_center_km'] <= threshold]['traj_id']
df_trimmed = df[df['traj_id'].isin(valid_traj_ids)]
print(f"✅ 筛选后轨迹数: {df_trimmed['traj_id'].nunique()}，轨迹点数: {len(df_trimmed)}")

# ====== 5. 设置地图边界并下载道路网络 ======
min_lat, max_lat = df_trimmed['lat'].min(), df_trimmed['lat'].max()
min_lon, max_lon = df_trimmed['lon'].min(), df_trimmed['lon'].max()
buffer = 0.005  # 500m 扩展
north = max_lat + buffer
south = min_lat - buffer
east = max_lon + buffer
west = min_lon - buffer
print(f"🌍 地图边界：N {north:.4f}, S {south:.4f}, E {east:.4f}, W {west:.4f}")

print("⏬ 下载地图中...")
G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='drive')
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
print("🗺️ 地图下载完成。")

# ====== 6. 静态热力图 + 道路 ======
plt.figure(figsize=(10, 10))
try:
    ax = sns.kdeplot(
        x=df_trimmed["lon"],
        y=df_trimmed["lat"],
        cmap="Reds",
        fill=True,
        bw_adjust=0.2,
        levels=100,
        thresh=0.01,
        alpha=0.6,
    )
    gdf_edges.plot(ax=ax, linewidth=0.5, color='gray')
    ax.set_title(f"Generated Signaling Trajectory Density\n(Top {int(vis_rate * 100)}% Close Trajectories)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig("generated_signaling_heatmap_static.png", dpi=300)
    plt.show()
except ValueError as e:
    print("❗ 热力图绘制失败：", e)

# ====== 7. 交互式热力图 ======
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
heat_data = [[row['lat'], row['lon']] for _, row in df_trimmed.iterrows()]
HeatMap(heat_data, radius=10).add_to(m)
m.save("generated_signaling_heatmap_interactive.html")

# ====== 完成提示 ======
print("🎉 可视化完成：")
print("📌 静态图: generated_signaling_heatmap_static.png")
print("🌐 交互图: generated_signaling_heatmap_interactive.html")
