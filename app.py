#DataSet1
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from geopy.distance import geodesic
import folium
import matplotlib.pyplot as plt
import matplotlib

matplotlib.font_manager.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')

# 1. 車站資訊與經緯度
stations = {
    "台北火車站": (25.0478, 121.5170),
    "新竹火車站": (24.8016, 120.9711),
    "台中火車站": (24.1368, 120.6845),
    "斗六火車站": (23.7096, 120.5431),
    "高雄火車站": (22.6394, 120.3020),
    "花蓮玉里": (23.3365, 121.3272),
    "台東知本": (22.7099, 121.0645),
}

station_names = list(stations.keys())
coords = list(stations.values())

# 2. 計算距離矩陣
n = len(coords)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = geodesic(coords[i], coords[j]).kilometers

# 顯示距離矩陣
dist_df = pd.DataFrame(dist_matrix, columns=station_names, index=station_names)
print("火車站之間的距離（單位：km）：\n")
print(dist_df.round(2))

# 3. MDS 降維
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
mds_coords = mds.fit_transform(dist_matrix)

# 4. 繪圖 (MDS)
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(mds_coords):
    plt.scatter(x, y, label=station_names[i])
    plt.text(x + 0.1, y + 0.1, station_names[i], fontsize=9)
plt.title("台灣火車站 MDS")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 5. Folium 地圖標記
m = folium.Map(location=[23.7, 121], zoom_start=7)
for name, (lat, lon) in stations.items():
    folium.Marker(location=[lat, lon], popup=name, tooltip=name).add_to(m)

# 儲存地圖
m.save("stations_map.html")
print("地圖已儲存為 stations_map.html，請在瀏覽器中開啟。")

#DataSet2