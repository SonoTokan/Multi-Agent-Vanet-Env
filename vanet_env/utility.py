from collections import defaultdict
import sys
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("./")
from vanet_env.entites import Rsu, Vehicle, OrderedQueueList
from vanet_env import network, config, utils


def cal_qoe(
    veh: Vehicle,
    rsus,
    max_qoe,
    weight,
    hop_penalty_rate,
    process_rsu_id,
    connect_rsu_id,
    rsu_network,
    time_step,
    proc_veh_qoe_dict=None,
    rsu_qoe_dict=None,
    fps=10,
):

    if proc_veh_qoe_dict is None:
        proc_veh_qoe_dict = {}
    if rsu_qoe_dict is None:
        rsu_qoe_dict = defaultdict(list)

    # 没有处理，connect_rsu_id应该必然有
    if process_rsu_id is None or connect_rsu_id is None:
        # 超过fps个时间步直接算qoe = 0，还是直接算0
        # veh.job.qoe = 0.0
        # # append
        # rsu_qoe_dict[connect_rsu_id] += [veh.job.qoe]

        if (time_step - veh.join_time) >= fps * 2:
            veh.job.qoe = 0.0
            proc_veh_qoe_dict[veh.vehicle_id] = 0.0
            rsu_qoe_dict[connect_rsu_id] += [veh.job.qoe]

    # 若有处理，云，本身，跳三种
    elif process_rsu_id == len(rsus):
        veh.job.qoe = max_qoe * 0.1
        proc_veh_qoe_dict[veh.vehicle_id] = veh.job.qoe
        # 这个qoe要算在connecet的rsu头上，虽然没有真的connect
        rsu_qoe_dict[veh.connected_rsu_id] += [veh.job.qoe]

    elif process_rsu_id == connect_rsu_id:
        rsu = rsus[process_rsu_id]
        index_in_proc_rsu = rsu.handling_jobs.index(veh)
        # may已经断开连接？
        # index_in_trans_rsu = rsu.connections.index(veh)

        cp = (
            rsu.computation_power
            * rsu.cp_norm[index_in_proc_rsu]
            * rsu.cp_usage
            / weight
        )

        process_qoe = veh.data_rate / config.JOB_DR_REQUIRE
        trans_qoe = cp / config.JOB_CP_REQUIRE

        veh.job.qoe = min(process_qoe, trans_qoe)
        # append
        rsu_qoe_dict[process_rsu_id] += [veh.job.qoe]
        proc_veh_qoe_dict[veh.vehicle_id] = veh.job.qoe
    else:
        proc_rsu = rsus[process_rsu_id]
        tran_rsu = rsus[connect_rsu_id]
        index_in_proc_rsu = proc_rsu.handling_jobs.index(veh)
        # pre connected 不一定连接
        # index_in_trans_rsu = tran_rsu.connections.index(veh)

        cp = (
            proc_rsu.computation_power
            * proc_rsu.cp_norm[index_in_proc_rsu]
            * proc_rsu.cp_usage
            / weight
        )

        process_qoe = veh.data_rate / config.JOB_DR_REQUIRE
        trans_qoe = cp / config.JOB_CP_REQUIRE

        qoe = min(process_qoe, trans_qoe)

        # 跨节点通信惩罚
        hop = network.find_hops(process_rsu_id, connect_rsu_id, rsu_network=rsu_network)
        qoe *= 1 - hop * hop_penalty_rate
        qoe = max(qoe, 0)  # 防止负值

        veh.job.qoe = qoe
        # 共同承担qoe
        rsu_qoe_dict[connect_rsu_id] += [veh.job.qoe]
        rsu_qoe_dict[process_rsu_id] += [veh.job.qoe]
        proc_veh_qoe_dict[veh.vehicle_id] = veh.job.qoe

    return proc_veh_qoe_dict, rsu_qoe_dict


# 严谨点的话应该每个rsu的veh qoe只进一次
def calculate_box_utility(
    vehs: Dict[str, Vehicle],
    rsus: List[Rsu],
    rsu_network,
    time_step,
    fps,
    weight,
    max_qoe=1.0,
    int_utility=False,
    max_connections=config.MAX_CONNECTIONS,
    num_cores=config.NUM_CORES,
):
    rsu_utility_dict = defaultdict(list)
    hop_penalty_rate = 0.2

    # 观察reward用
    if time_step >= 1000 and time_step % 1000 == 0:
        time_step

    # 由veh计算，
    for v_id, veh in vehs.items():
        veh: Vehicle

        if veh.is_cloud:
            qoe = max_qoe * 0.1
            rsu_utility_dict[veh.connected_rsu_id].append(qoe)
            veh.job.qoe = qoe
            continue

        trans_qoe = min(veh.data_rate, 40) / config.JOB_DR_REQUIRE

        for p_rsu in veh.job.processing_rsus:

            if p_rsu is not None:
                p_rsu: Rsu
                p_idx = p_rsu.handling_jobs.index((veh, 0))
                process_qoe = (
                    min(p_rsu.real_cp_alloc[p_idx] / p_rsu.handling_jobs[p_idx][1], 40)
                    / config.JOB_CP_REQUIRE
                )

                trans_rsu: Rsu = rsus[veh.connected_rsu_id]

                if p_rsu.id == veh.connected_rsu_id:
                    qoe = min(process_qoe, trans_qoe)

                    # caching debug
                    if veh.job.job_type in trans_rsu.caching_contents:
                        qoe = min(qoe + qoe * 0.2, 1)
                    else:
                        qoe = max(qoe - qoe * 0.05, 0)

                    veh.job.qoe = qoe

                    trans_rsu.ee = 0.3 * (1 - trans_rsu.cp_usage)

                    rsu_utility_dict[veh.connected_rsu_id].append(
                        float(qoe * 0.7 + trans_rsu.ee)
                    )

                else:
                    qoe = min(process_qoe, trans_qoe)
                    qoe = max(qoe - qoe * hop_penalty_rate, 0)

                    if veh.job.job_type in rsus[veh.connected_rsu_id].caching_contents:
                        qoe = min(qoe + qoe * 0.2, 1)
                    else:
                        qoe = max(qoe - qoe * 0.05, 0)

                    veh.job.qoe = qoe

                    p_rsu.ee = 0.3 * (1 - p_rsu.cp_usage)
                    trans_rsu.ee = 0.3 * (1 - trans_rsu.cp_usage)
                    # or give trans and process qoe spretely?
                    rsu_utility_dict[veh.connected_rsu_id].append(
                        float(qoe * 0.7 + trans_rsu.ee)
                    )
                    rsu_utility_dict[p_rsu.id].append(float(qoe * 0.7 + p_rsu.ee))
    return rsu_utility_dict


# python 3.9 + can be dict[str, Vehicle], list[Rsu]
def calculate_frame_utility(
    vehs: Dict[str, Vehicle],
    rsus: List[Rsu],
    proc_veh_set: set[str],
    rsu_network,
    time_step,
    fps,
    weight,
    max_qoe=1.0,
    int_utility=False,
    max_connections=config.MAX_CONNECTIONS,
    num_cores=config.NUM_CORES,
):
    # 观察reward用
    if time_step % 1000 == 0:
        time_step
    frame = time_step % fps
    conn_index = frame % max_connections
    handle_index = frame % num_cores
    alloc_index = frame % num_cores
    proc_qoe_dict = {}
    rsu_qoe_dict = defaultdict(list)
    hop_penalty_rate = 0.04

    # 只计算该frame处理的veh的utility, 这样算应该就是global utility，
    # 但是没有考虑到没有proc之外的车辆的qoe变化
    for veh_id in proc_veh_set:
        veh: Vehicle = vehs[veh_id]
        proc_qoe_dict, rsu_qoe_dict = cal_qoe(
            veh,
            rsus,
            max_qoe,
            weight,
            hop_penalty_rate,
            time_step=time_step,
            fps=fps,
            process_rsu_id=veh.job.processing_rsu_id,
            connect_rsu_id=veh.connected_rsu_id,
            rsu_network=rsu_network,
        )

    # 计算其他veh的，但是由于rsu里handlingjob可能重复的原因，无法从rsu计算。因为veh不可能重复，所以可以计算veh的
    # 这个不一定存在
    # for veh_id, veh in vehs.items():
    #     # 跳过已计算
    #     if veh_id in proc_veh_set:
    #         continue

    #     # 拼接未计算，这里可以搞个比值来权重全局和个人
    #     cal_qoe(
    #         veh,
    #         rsus,
    #         max_qoe,
    #         weight,
    #         hop_penalty_rate,
    #         process_rsu_id=veh.job.processing_rsu_id,
    #         connect_rsu_id=veh.connected_rsu_id,
    #         rsu_network=rsu_network,
    #         proc_veh_qoe_dict=proc_qoe_dict,
    #         rsu_qoe_dict=rsu_qoe_dict,
    #     )

    return proc_qoe_dict, rsu_qoe_dict


# 全局计算法
def calculate_beta_utility(
    vehs: Dict[str, Vehicle],
    rsus: List[Rsu],
    rsu_network,
    time_step,
    fps,
    weight,
    max_qoe=1.0,
    int_utility=False,
):
    frame = time_step % fps
    hop_penalty_rate = 0.04
    # qoe = min(process_qoe, data_trans_qoe)
    # shape (5, 20)
    process_qoes = np.zeros((len(rsus), config.NUM_CORES))
    data_trans_qoes = np.zeros((len(rsus), config.MAX_CONNECTIONS))

    # 方法1遍历Rsu计算所有QoE
    # 方法2遍历Vehicle计算所有QoE
    # 需要测试计算是否有问题
    for v_id, veh in vehs:
        veh: Vehicle
        # 云，本身，跳三种
        if veh.job.processing_rsu_id == len(rsus):
            veh.job.qoe = max_qoe * 0.1
        elif veh.job.processing_rsu_id == veh.connected_rsu_id:
            # 假如并没有connected_rsu_id或processing_rsu_id？

            rsu = rsus[veh.job.processing_rsu_id]
            index_in_proc_rsu = rsu.handling_jobs.index(veh)
            # may已经断开连接？
            index_in_trans_rsu = rsu.connections.index(veh)

            cp = (
                rsu.computation_power
                * rsu.cp_norm[index_in_proc_rsu]
                * rsu.cp_usage
                / weight
            )

            process_qoe = veh.data_rate / config.JOB_DR_REQUIRE
            trans_qoe = cp / config.JOB_CP_REQUIRE

            process_qoes[veh.job.processing_rsu_id][index_in_proc_rsu] = process_qoe
            data_trans_qoes[veh.job.processing_rsu_id][index_in_trans_rsu] = process_qoe

            veh.job.qoe = min(process_qoe, trans_qoe)
        else:
            assert NotImplementedError()
            ...


# 問題特別多
def calculate_utility_all_optimized(
    vehs: list,
    rsus: list,
    rsu_network,
    time_step,
    fps,
    weight,
    max_qoe=1.0,
    int_utility=False,
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

    # process_rsu_id 不同步的问题？
    for tv in vehs:
        job = tv.job  # 减少属性访问次数
        process_rsu_id = job.processing_rsu_id
        join_time = tv.join_time

        # 未处理状态的快速判断
        if (process_rsu_id == len(rsus) + 1) or (process_rsu_id is None):
            if (time_step - join_time) // fps >= 2:
                job.qoe = 0
                global_qoe_list.append(0)
            continue

        # 由于环境的原因，veh只连一个rsu
        for c in filter(None, tv.connections):  # 过滤空连接
            if c.is_cloud:
                # c.qoe = cloud_qoe
                global_qoe_list.append(cloud_qoe)
                # qoe溯源
                c.rsu.qoe_list.append(cloud_qoe)
                job.qoe = cloud_qoe
                c.qoe = job.qoe
                continue

            # why the job not connected?
            if not c.check_connection():
                print(f"timestep{time_step}:veh job not connected")
                if (time_step - join_time) // fps >= 2:
                    job.qoe = 0
                    c.qoe = 0
                    global_qoe_list.append(0)
                    continue

            # 获取处理 RSU 的缓存内容（提前获取）
            process_rsu = rsus[process_rsu_id]
            process_cache = process_rsu.caching_contents
            # 计算基础算力参数（移出内层循环）
            cp_base = process_rsu.computation_power * process_rsu.cp_usage // weight

            # 获取连接参数
            conn_rsu = c.rsu

            alloc_idx = process_rsu.handling_jobs.index(c)

            # 计算核心指标（使用预存值）
            cp = cp_base * process_rsu.cp_norm[alloc_idx]
            dr = min(c.data_rate, config.JOB_DR_REQUIRE)  # 简化条件判断

            # QoE 计算（使用预存系数）
            qoe = min((dr / config.JOB_DR_REQUIRE), cp / config.JOB_CP_REQUIRE)

            # 跨节点惩罚
            if process_rsu_id != conn_rsu.id:
                hop = hop_cache.get((process_rsu_id, conn_rsu.id), 0)
                qoe *= 1 - hop * hop_penalty_rate
                qoe = max(qoe, 0)  # 防止负值

            # 缓存未命中惩罚
            if job.job_type not in process_cache:
                qoe *= caching_penalty

            # 结果记录
            job.qoe = int(qoe) if int_utility else qoe
            c.qoe = job.qoe
            global_qoe_list.append(qoe)

    # ========== RSU 处理阶段 ==========
    individual_utilities = []
    ee_base = 0.3  # 能量效率基数

    for rsu in rsus:
        rsu_cache = rsu.caching_contents
        cp_max = rsu.computation_power * rsu.cp_usage // weight

        for idx, hconn in enumerate(rsu.handling_jobs):

            if hconn is None:
                continue

            # Why?
            if not hconn.connected:
                rsu.qoe_list.append(0)
                print(f"timestep{time_step}:handling disconnected job")
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

            # QoE 计算，符合水桶效应
            qoe = min((dr / config.JOB_DR_REQUIRE), cp / config.JOB_CP_REQUIRE)
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
            avg_u = sum(qoe_list) * 0.7 + len(qoe_list) * ee / len(qoe_list)
            individual_utilities.append(int(avg_u) if int_utility else avg_u)
            rsu.avg_u = avg_u

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


# deprecated 需按照上面的计算方法计算
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
