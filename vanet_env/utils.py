import sys

sys.path.append("./")
from vanet_env import env_config
import numpy as np
import matplotlib as mpl
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import pandas as pd
from gymnasium import spaces

import os

# Load the images
print("img_load_called")

rsu_img_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.svg")
vehicle_img_path = os.path.join(os.path.dirname(__file__), "assets", "vehicle.svg")

rsu_path, rsu_attributes = svg2paths(rsu_img_path)
vehicle_path, vehicle_attributes = svg2paths(vehicle_img_path)


RSU_MARKER = parse_path(
    """"M 310.62,161.12
           C 299.12,164.12 289.00,172.37 283.75,183.12
             281.25,188.12 280.75,191.37 280.62,199.38
             280.62,212.12 284.62,221.12 293.88,229.12
             297.25,232.00 300.00,234.87 300.00,235.37
             300.00,235.88 268.50,322.00 229.88,426.62
             152.75,635.88 157.12,622.38 164.38,632.00
             170.38,639.88 179.00,641.88 188.12,637.62
             190.88,636.25 221.50,611.25 256.00,581.88
             290.62,552.62 319.62,528.75 320.50,529.00
             321.38,529.25 350.25,553.25 384.75,582.50
             419.38,611.75 449.63,636.62 452.25,637.75
             461.00,641.88 469.75,639.75 475.62,632.00
             483.00,622.38 487.25,635.88 410.12,426.62
             371.50,322.00 340.00,236.00 340.00,235.37
             340.00,234.87 342.75,232.00 346.13,229.12
             349.88,225.88 353.63,221.00 355.88,216.62
             358.88,210.50 359.38,207.87 359.38,199.38
             359.25,190.37 358.88,188.50 355.00,181.50
             348.00,168.50 337.12,161.50 322.62,160.62
             318.12,160.37 312.75,160.50 310.62,161.12 Z
           M 333.12,333.38
           C 339.88,351.88 345.62,367.50 345.88,368.38
             346.38,369.50 329.12,383.38 321.50,388.00
             320.50,388.62 315.38,385.62 307.50,380.00
             300.62,375.00 294.75,370.50 294.38,370.00
             293.62,368.75 318.75,300.00 320.00,300.00
             320.50,300.00 326.38,315.00 333.12,333.38 Z
           M 314.75,393.12
           C 293.25,409.00 275.88,421.25 275.50,420.88
             275.00,420.50 282.75,398.50 291.88,373.88
             292.25,372.87 296.38,375.75 303.75,382.50
             310.00,388.25 314.88,393.00 314.75,393.12 Z
           M 356.75,397.38
           C 361.25,410.00 364.88,420.62 364.50,420.88
             364.25,421.12 355.25,415.00 344.62,407.25
             344.62,407.25 325.12,393.12 325.12,393.12
             325.12,393.12 336.00,382.87 336.00,382.87
             342.00,377.25 347.25,373.12 347.62,373.50
             348.00,374.00 352.12,384.75 356.75,397.38 Z
           M 348.88,425.37
           C 376.62,451.75 376.88,452.00 380.12,461.00
             380.12,461.00 383.50,470.12 383.50,470.12
             383.50,470.12 377.12,474.12 377.12,474.12
             351.62,490.38 321.25,508.75 320.00,508.75
             318.88,508.75 292.50,492.75 261.12,473.00
             261.12,473.00 256.50,470.12 256.50,470.12
             256.50,470.12 259.75,461.12 259.75,461.12
             262.88,452.25 263.00,452.12 291.00,425.50
             306.50,410.75 319.50,398.75 320.00,398.75
             320.62,398.75 333.63,410.75 348.88,425.37 Z
           M 281.62,495.25
           C 295.50,507.00 306.75,517.00 306.75,517.38
             306.38,518.75 220.62,571.25 220.25,570.25
             220.13,569.75 227.88,548.00 237.50,521.88
             247.13,495.88 255.00,474.38 255.00,474.12
             255.00,472.87 258.75,475.88 281.62,495.25 Z
           M 386.13,477.12
           C 391.88,491.62 420.00,569.50 419.75,570.25
             419.38,571.25 334.00,519.00 333.38,517.38
             333.12,516.62 382.50,474.00 383.88,473.88
             384.38,473.75 385.38,475.25 386.13,477.12 Z
           M 404.38,135.25
           C 425.12,179.38 425.12,220.75 404.38,264.62
             401.38,271.00 399.38,276.25 400.00,276.25
             402.25,276.25 414.12,265.75 419.50,259.00
             426.25,250.37 434.00,234.87 437.00,223.75
             440.38,211.12 440.12,187.25 436.50,174.38
             431.25,156.37 420.12,138.87 407.38,128.75
             403.88,126.00 400.50,123.75 400.00,123.75
             399.38,123.75 401.38,129.00 404.38,135.25 Z
           M 231.38,130.00
           C 212.63,145.62 200.00,173.87 200.00,200.00
             200.00,226.12 212.63,254.38 231.38,270.00
             235.50,273.38 239.38,276.25 240.00,276.25
             240.62,276.25 238.62,271.00 235.62,264.62
             214.88,220.50 214.75,179.38 235.62,135.25
             238.62,129.00 240.62,123.75 240.00,123.75
             239.38,123.75 235.50,126.62 231.38,130.00 Z
           M 176.75,72.87
           C 162.62,84.87 146.88,105.00 138.38,121.88
             105.75,187.25 118.00,268.62 168.25,319.25
             172.62,323.62 178.00,328.62 180.25,330.25
             180.25,330.25 184.38,333.12 184.38,333.12
             184.38,333.12 181.50,328.75 181.50,328.75
             153.38,285.88 140.00,244.25 140.00,200.00
             140.00,155.88 153.50,113.87 181.38,71.50
             182.75,69.38 183.75,67.50 183.50,67.50
             183.25,67.50 180.12,69.87 176.75,72.87 Z
           M 462.50,77.62
           C 512.25,154.38 512.25,245.62 462.50,322.38
             462.50,322.38 454.88,334.00 454.88,334.00
             454.88,334.00 459.63,330.25 459.63,330.25
             474.25,318.75 492.88,295.50 501.62,278.12
             533.88,213.38 522.25,133.12 473.25,82.37
             468.75,77.87 462.88,72.25 460.00,70.00
             460.00,70.00 454.88,66.00 454.88,66.00
             454.88,66.00 462.50,77.62 462.50,77.62 Z
           M 517.88,18.37
           C 571.75,89.12 592.12,181.62 572.50,266.25
             563.62,304.12 538.88,356.25 516.62,383.38
             514.25,386.38 512.75,388.75 513.12,388.75
             514.75,388.75 528.12,376.00 537.88,365.25
             576.38,322.75 599.88,260.25 600.00,200.50
             600.12,143.38 579.00,84.00 543.75,41.25
             536.25,32.12 515.12,11.25 513.38,11.25
             512.88,11.25 514.88,14.50 517.88,18.37 Z
           M 118.50,18.12
           C 92.00,41.25 66.88,80.00 53.88,117.87
             22.62,209.00 46.00,311.12 113.12,376.62
             125.88,389.25 129.50,391.12 122.25,381.75
             68.25,310.88 47.88,218.50 67.50,133.75
             76.38,95.62 100.88,44.25 123.38,16.50
             125.75,13.62 127.38,11.25 127.00,11.25
             126.75,11.25 122.88,14.38 118.50,18.12 Z"""
)

VEHICLE_MARKER = parse_path(vehicle_attributes[0]["d"])

RSU_MARKER.vertices -= RSU_MARKER.vertices.mean(axis=0)
VEHICLE_MARKER.vertices -= VEHICLE_MARKER.vertices.mean(axis=0)

RSU_MARKER = RSU_MARKER.transformed(mpl.transforms.Affine2D().rotate_deg(180))
RSU_MARKER = RSU_MARKER.transformed(mpl.transforms.Affine2D().scale(-1, 1))

# VEHICLE_MARKER = VEHICLE_MARKER.transformed(mpl.transforms.Affine2D().rotate_deg(180))
# VEHICLE_MARKER = VEHICLE_MARKER.transformed(mpl.transforms.Affine2D().scale(-1, 1))


# canva distance to real distance
def distance_to_real_distance(distance):
    return distance / (1000 / env_config.COORDINATE_UNIT)


# real distance to canva distance
def real_distance_to_distance(real_distance):
    return real_distance * (1000 / env_config.COORDINATE_UNIT)


# detect sumo env
def sumo_detector():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


def interpolate_color(min_val, max_val, value):
    """
    Linear interpolation between red, yellow and green according to the given values.

    :param min_val: minimum value
    :param max_val: maximum value
    :param value: current value
    :return: (r, g, b) color value
    """
    # Insure in range
    value = max(min_val, min(max_val, value))

    # Compute ratio
    ratio = (value - min_val) / (max_val - min_val)

    if ratio < 0.5:
        # Red to yellow
        r = 255
        g = int(255 * (ratio * 2))
        b = 0
    else:
        # Yellow to green
        r = int(255 * (2 - ratio * 2))
        g = 255
        b = 0

    return (r, g, b)


def RtoCsv():
    import pyreadr
    import pandas as pd

    file = os.path.join(os.path.dirname(__file__), "data", "Timik", "visits.RData")

    # 读取 .RData 文件
    result = pyreadr.read_r(file)

    # 假设数据保存在名为 'x' 的变量中
    df = result["x"]

    # 保存为 CSV 文件
    df.to_csv("timik.csv", index=False)

    print("数据已成功保存为 timik.csv 文件。")


# bug detected do not use
def is_empty(list_in):
    return all(conn is None for conn in list_in) or list_in


def test():
    if [None]:
        print(is_empty([]), is_empty([None] * 2))


def all_none(arr):
    for a in arr:
        if a is not None:
            return False
    return True


def normalize_array_np(arr):
    sum = 0
    norm = []

    for num in arr:
        if num is not None:
            sum += num

    if sum == 0:
        return [0] * len(arr)

    for num in arr:
        norm.append(0 if num is None else num / sum)

    return norm


def normalize_array_np_deprecated(arr):
    """
    标准化处理包含None的数组：
    1. 非None值将被转换为 x * count / total，其中：
       - count 是非None值的数量
       - total 是非None值的总和
    2. None值保持原位置不变

    参数：
    arr (list): 包含数值和None的列表

    返回：
    list: 处理后的列表，非None值标准化，None保持原位置

    异常：
    ValueError: 当非None值的总和为0但存在非零元素时抛出
    """
    # 过滤出非None值
    non_none = [x for x in arr if x is not None]
    count = len(non_none)

    # 处理全None情况
    if count == 0:
        return arr.copy()

    total = sum(non_none)

    # 处理总和为0的情况
    if total == 0:
        if all(x == 0 for x in non_none):
            # 所有非None值均为0时返回0
            return [0 if x is not None else None for x in arr]
        else:
            raise ValueError("非None值的总和为0但存在非零元素，无法标准化")

    # 计算标准化值
    return [x * count / total if x is not None else None for x in arr]


# def normalize_array_np(arr: np.ndarray) -> np.ndarray:
#     """将输入数组归一化为概率分布（总和=1）

#     Args:
#         arr: 输入数组，支持int/float类型

#     Returns:
#         np.ndarray: 归一化后的浮点数组

#     Raises:
#         ValueError: 当数组包含负数或无法归一化时
#     """
#     arr = np.asarray([a if a != None else 0 for a in arr], dtype=np.float64)  # 强制类型转换

#     if np.any(arr < 0):
#         raise ValueError("输入数组包含负值，无法归一化为概率分布")

#     total = arr.sum()

#     if total <= 0:
#         if np.all(arr == 0):
#             # 全零时返回均匀分布
#             return np.full_like(arr, 1.0/arr.size, dtype=np.float64)
#         else:
#             raise ValueError(f"无效的数组总和: {total}，无法归一化")

#     return arr / total


def multi_discrete_space_to_discrete_space(md_space):
    # 计算总的动作数
    action_dims = md_space.nvec
    total_actions = np.prod(action_dims)

    # 创建一个 Discrete 动作空间
    discrete_action_space = spaces.Discrete(total_actions)
    return discrete_action_space


# 将 Discrete 动作转换回 MultiDiscrete 动作
def discrete_to_multi_discrete(md_space, discrete_action):
    action_dims = md_space.nvec

    discrete_action = 0

    action = []
    for i in reversed(range(len(action_dims))):
        action.append(discrete_action % action_dims[i])
        discrete_action = discrete_action // action_dims[i]
    return list(reversed(action))


# 将 MultiDiscrete 动作转换为 Discrete 动作
def multi_discrete_to_discrete(md_space, action):
    action_dims = md_space.nvec

    discrete_action = 0
    for i in range(len(action)):
        discrete_action *= action_dims[i]
        discrete_action += action[i]
    return discrete_action
