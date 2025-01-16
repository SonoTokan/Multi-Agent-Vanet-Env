import sys

sys.path.append("./")

import os

import numpy as np
import traci
from vanet_env.entites import Rsu
from vanet_env import config

file_path = os.path.join(
    os.path.dirname(__file__), "assets", "seattle", "sumo", "osm.sumocfg"
)

gui_settings_path = os.path.join(
    os.path.dirname(__file__), "assets", "seattle", "sumo", "gui_hide_all.xml"
)

icon_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.png")

# 启动SUMO仿真
traci.start(["sumo-gui", "-c", file_path, "--gui-settings-file", gui_settings_path])

# init rsus
rsus = [
    Rsu(
        id=i,
        position=config.RSU_POSITIONS[i],
    )
    for i in range(len(config.RSU_POSITIONS))
]

for rsu in rsus:
    poi_id = f"rsu_icon_{rsu.id}"
    # add icon
    traci.poi.add(
        poi_id,
        rsu.position[0],
        rsu.position[1],
        (255, 0, 0, 255),
        width=20,
        height=20,
        imgFile=icon_path,
        layer=10,
    )

rsu_range = 120

# 绘制RSU范围的虚线圆圈
for rsu in rsus:
    num_segments = 36
    for i in range(num_segments):
        angle1 = 2 * np.pi * i / num_segments
        angle2 = 2 * np.pi * (i + 1) / num_segments
        x1 = rsu.position[0] + rsu_range * np.cos(angle1)
        y1 = rsu.position[1] + rsu_range * np.sin(angle1)
        x2 = rsu.position[0] + rsu_range * np.cos(angle2)
        y2 = rsu.position[1] + rsu_range * np.sin(angle2)
        traci.polygon.add(
            f"circle_segment_rsu{rsu.id}_{i}",
            [(x1, y1), (x2, y2)],
            color=(255, 0, 0, 255),
            fill=False,
            lineWidth=1.0,
            layer=10,
        )

# 运行仿真步骤
for step in range(3600):
    traci.simulationStep()
    # 获取所有车辆ID
    vehicle_ids = traci.vehicle.getIDList()

    # 清除所有连接线
    for polygon_id in traci.polygon.getIDList():
        if polygon_id.startswith("line_rsu"):
            traci.polygon.remove(polygon_id)

    for vehicle_id in vehicle_ids:
        # 获取车辆位置
        vehicle_x, vehicle_y = traci.vehicle.getPosition(vehicle_id)

        for rsu in rsus:
            rsu_x = rsu.position[0]
            rsu_y = rsu.position[1]
            # 计算车辆与RSU的距离
            distance = np.sqrt((vehicle_x - rsu_x) ** 2 + (vehicle_y - rsu_y) ** 2)

            # 如果车辆在RSU范围内，绘制连接线
            if distance <= rsu_range:
                traci.polygon.add(
                    f"line_rsu{rsu.id}_to_{vehicle_id}",
                    [(rsu_x, rsu_y), (vehicle_x, vehicle_y)],
                    color=(0, 255, 0, 255),
                    fill=False,
                    lineWidth=0.5,
                    layer=10,
                )
# 关闭仿真
traci.close()
