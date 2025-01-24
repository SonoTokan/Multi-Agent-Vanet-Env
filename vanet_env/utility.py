import sys

from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("./")
from vanet_env.entites import Rsu, Vehicle
from vanet_env import network, config, utils


def calculate_utility_all_optimized(
    vehs: list,
    rsus: list,
    rsu_network,
    time_step,
    fps,
    weight,
    max_qoe,
    int_utility=True,
):
    """_summary_

    Args:
        vehs (list): _description_
        rsus (list): _description_
        rsu_network (_type_): _description_
        time_step (_type_): _description_
        fps (_type_): _description_
        weight (_type_): _description_
        max_qoe (_type_): _description_
        int_utility (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # ========== 预计算阶段 ==========
    # 预计算所有 RSU 之间的跳数 (O(n^2) 但只需计算一次)
    hop_cache = {}
    for rsu1 in rsus:
        # 顺便清空
        rsu1.qoe_list = []
        for rsu2 in rsus:
            key = (rsu1.id, rsu2.id)
            hop_cache[key] = network.find_hops(rsu1.id, rsu2.id, rsu_network)

    # 预计算公共系数
    max_qoe_half = max_qoe / 2
    cloud_qoe = max_qoe * 0.1
    hop_penalty_rate = 0.04
    caching_penalty = 0.8

    # ========== 车辆处理阶段 ==========
    global_qoe_list = []

    for tv in vehs:
        job = tv.job  # 减少属性访问次数
        process_rsu_id = job.processing_rsu_id
        join_time = tv.join_time

        # 未处理状态的快速判断
        if (process_rsu_id == len(rsus) + 1) or (process_rsu_id is None):
            if (time_step - join_time) // fps >= 2:
                global_qoe_list.append(0)
            continue

        # 获取处理 RSU 的缓存内容（提前获取）
        process_rsu = rsus[process_rsu_id]
        process_cache = process_rsu.caching_contents

        # 计算基础算力参数（移出内层循环）
        cp_base = process_rsu.computation_power * process_rsu.cp_usage // weight

        for c in filter(None, tv.connections):  # 过滤空连接
            if c.is_cloud:
                c.qoe = cloud_qoe
                global_qoe_list.append(cloud_qoe)
                # qoe溯源
                c.rsu.qoe_list.append(cloud_qoe)
                continue

            # 获取连接参数
            conn_rsu = c.rsu
            alloc_idx = process_rsu.handling_jobs.index(c)

            # 计算核心指标（使用预存值）
            cp = cp_base * process_rsu.cp_norm[alloc_idx]
            dr = min(c.data_rate, config.JOB_DR_REQUIRE)  # 简化条件判断

            # QoE 计算（使用预存系数）
            qoe = max_qoe_half * (
                (config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE)
                + (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)
            )

            # 跨节点惩罚
            if process_rsu_id != conn_rsu.id:
                hop = hop_cache.get((process_rsu_id, conn_rsu.id), 0)
                qoe *= 1 - hop * hop_penalty_rate
                qoe = max(qoe, 0)  # 防止负值

            # 缓存未命中惩罚
            if job.job_type not in process_cache:
                qoe *= caching_penalty

            # 结果记录
            c.qoe = int(qoe) if int_utility else qoe
            global_qoe_list.append(qoe)

    # ========== RSU 处理阶段 ==========
    individual_utilities = []
    ee_base = 3  # 能量效率基数

    for rsu in rsus:
        rsu_cache = rsu.caching_contents
        cp_max = rsu.computation_power * rsu.cp_usage * 0.2

        for idx, hconn in enumerate(rsu.handling_jobs):
            if hconn is None:
                continue

            # 云处理快速通道
            # 云是由conn的rsu处理的
            # handling_jobs不可能出现cloud
            # if hconn.is_cloud:
            #     hconn.rsu.qoe_list.append(cloud_qoe)
            #     continue

            # 计算核心指标
            dr = min(hconn.data_rate, config.JOB_DR_REQUIRE)
            cp = cp_max * rsu.cp_norm[idx]

            # QoE 计算
            qoe = max_qoe_half * (
                (config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE)
                + (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)
            )

            # 跨节点惩罚
            if hconn.rsu.id != rsu.id:
                hop = hop_cache.get((rsu.id, hconn.rsu.id), 0)
                qoe *= 1 - hop * hop_penalty_rate

            # 缓存未命中惩罚
            if hconn.veh.job.job_type not in rsu_cache:
                qoe *= caching_penalty

            rsu.qoe_list.append(qoe)
            # 能量效率计算

        ee = ee_base * (1 - rsu.cp_usage / weight)
        rsu.ee = ee
        qoe_list = rsu.qoe_list
        # RSU 效用聚合
        if qoe_list:
            avg_u = sum(qoe_list) + len(qoe_list) * ee / len(qoe_list)
            individual_utilities.append(int(avg_u) if int_utility else avg_u)

    # ========== 最终结果计算 ==========
    avg_global = sum(global_qoe_list) / len(global_qoe_list) if global_qoe_list else 0
    avg_global = int(avg_global) if int_utility else avg_global

    avg_local = (
        sum(individual_utilities) / len(individual_utilities)
        if individual_utilities
        else 0
    )
    avg_local = int(avg_local) if int_utility else avg_local
    
    return avg_global, avg_local


def calculate_utility_all(
    vehs: list,
    rsus: list,
    rsu_network,
    time_step,
    fps,
    weight,
    max_qoe,
    int_utility=True,
):

    tv = Vehicle()
    process_rsu_id = tv.job.processing_rsu_id
    tv.job.job_type
    tv.join_time
    # cloud
    if tv.job.processing_rsu_id == len(rsus):
        ...

    global_qoe_list = []

    # main
    for tv in vehs:
        process_rsu_id = tv.job.processing_rsu_id

        # not handling
        if process_rsu_id == len(rsus) + 1 or process_rsu_id is None:
            # too much time not handling, qoe = 0
            if (time_step - tv.join_time) // fps >= 2:
                global_qoe_list.append(0)
            continue

        # handling
        for c in tv.connections:
            if c is not None:
                # cloud process
                if c.is_cloud:
                    qoe = max_qoe * 0.1
                    global_qoe_list.append(qoe)
                    c.qoe = qoe
                    continue

                conn_rsu_id = c.rsu.id
                process_rsu = rsus[process_rsu_id]

                alloc_index = process_rsu.handling_jobs.index(c)

                cp_max = process_rsu.computation_power * process_rsu.cp_usage // weight

                cp = cp_max * process_rsu.cp_norm[alloc_index]

                dr = c.data_rate
                dr = config.JOB_DR_REQUIRE if dr >= config.JOB_DR_REQUIRE else dr

                qoe = max_qoe / 2 * (
                    config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE
                ) + max_qoe / 2 * (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)

                # not self process
                if process_rsu_id != conn_rsu_id:
                    hop = network.find_hops(process_rsu_id, conn_rsu_id, rsu_network)
                    # 每个hop - 4%的qoe
                    qoe -= hop * 0.04 * (qoe)
                    qoe = 0 if qoe <= 0 else qoe

                caching_contents = rsu.caching_contents

                if c.veh.job.job_type not in caching_contents:
                    qoe *= 0.8
                    global_qoe_list.append(qoe)

            c.qoe = int(qoe) if int_utility else qoe

    avg_global_qoe = sum(global_qoe_list) / len(global_qoe_list)
    avg_global_qoe = int(avg_global_qoe) if int_utility else avg_global_qoe

    individual_utilities = []

    for rsu in rsus:
        max_e = 3
        qoe = max_qoe - max_e
        caching_contents = rsu.caching_contents
        cp_max = rsu.computation_power * rsu.cp_usage * 0.2

        for idx, hconn in enumerate(rsu.handling_jobs):
            if hconn is None:
                continue
            cp = cp_max * rsu.cp_norm[idx]
            dr = hconn.data_rate
            dr = config.JOB_DR_REQUIRE if dr >= config.JOB_DR_REQUIRE else dr
            # dev tag: 云的算力是否设为不同
            # 云固定延迟、码率，且不用计算cp_alloc、bw_alloc
            if hconn.is_cloud:
                qoe *= 0.1
                # 直接为最大utility的10%
                # l_trans = config.CLOUD_TRANS_TIME
                # l_compute = config.CLOUD_COMPUTATIONALLY_TIME
                l_all = ...
                ...
            # 不是本rsu连接的需要计算跳数，不过bw_alloc可以不计
            elif hconn.rsu.id != rsu.id:
                hop = network.find_hops(rsu.id, hconn.rsu.id, rsu_network)
                # 每距离标准码率和标准计算能力差多少就减多少qoe的百分比
                qoe = max_qoe / 2 * (
                    config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE
                ) + max_qoe / 2 * (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)

                # 每个hop - 2%的qoe
                qoe -= hop * 0.04 * (qoe)
            else:

                qoe = max_qoe / 2 * (
                    config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE
                ) + max_qoe / 2 * (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)

            # dev tag: check job_type and caching_content is index
            if not hconn.is_cloud and hconn.veh.job.job_type not in caching_contents:
                qoe *= 0.8

            ee = 3 * (1 - (rsu.cp_usage / weight))
            u = qoe + ee
            individual_utilities.append(u)
            rsu.ee = ee
            rsu.utility = u

    avg_local_u = sum(individual_utilities) / len(individual_utilities)
    avg_local_u = int(avg_local_u) if int_utility else avg_local_u

    return avg_global_qoe, avg_local_u


# 目前不计算抖动
def calculate_utility(rsu: Rsu, rsu_network):
    """
    return: qoe, energy_eff, utility
    """

    caching_contents = rsu.caching_contents

    # computationally power alloc
    cp_max = rsu.computation_power * rsu.cp_usage
    # bw alloc
    # bw = rsu.bw * rsu.bw_ratio

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # reward to μ=0 σ=1
    scaler = StandardScaler()
    w_e = 0.25
    w_qoe = 0.75
    max_energy_efficiency_utility = 0.25
    max_qoe = 1

    qoe = 1
    energy_efficiency_utility = 0.25

    qoe_list = []
    ee_list = []
    u_list = []

    for idx, hconn in enumerate(rsu.handling_jobs):
        if hconn is None:
            continue
        cp = cp_max * rsu.cp_norm[idx]
        dr = hconn.data_rate
        dr = config.JOB_DR_REQUIRE if dr >= config.JOB_DR_REQUIRE else dr
        # dev tag: 云的算力是否设为不同
        # 云固定延迟、码率，且不用计算cp_alloc、bw_alloc
        if hconn.is_cloud:
            qoe *= 0.1
            # 直接为最大utility的10%
            # l_trans = config.CLOUD_TRANS_TIME
            # l_compute = config.CLOUD_COMPUTATIONALLY_TIME
            l_all = ...
            ...
        # 不是本rsu连接的需要计算跳数，不过bw_alloc可以不计
        elif hconn.rsu.id != rsu.id:
            hop = network.find_hops(rsu.id, hconn.rsu.id, rsu_network)
            # 每距离标准码率和标准计算能力差多少就减多少qoe的百分比
            qoe = max_qoe / 2 * (
                config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE
            ) + max_qoe / 2 * (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)

            # 每个hop - 2%的qoe
            qoe -= hop * 0.02 * (qoe)
        else:

            qoe = max_qoe / 2 * (
                config.JOB_DR_REQUIRE - dr / config.JOB_DR_REQUIRE
            ) + max_qoe / 2 * (config.JOB_CP_REQUIRE - cp / config.JOB_CP_REQUIRE)
            ...

        # dev tag: check job_type and caching_content is index
        if not hconn.is_cloud and hconn.veh.job.job_type not in caching_contents:
            qoe *= 0.8

        energy_efficiency_utility = energy_efficiency_utility * (
            1 - (rsu.cp_usage / 100)
        )

        ee = energy_efficiency_utility / max_energy_efficiency_utility

        u = w_qoe * qoe
        +w_e * ee

        hconn.qoe = qoe
        rsu.energy_efficiency = ee

        qoe_list.append(qoe)
        u_list.append(u)
        ee_list.append(ee)

    size = len(u_list)

    # return mean ut
    return 0 if size == 0 else sum(u_list) / size
