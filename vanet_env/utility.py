import sys

from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("./")
from vanet_env.entites import Rsu
from vanet_env import network, config, utils


# 目前不计算抖动
def calculate_utility(rsu: Rsu, rsu_network):
    """
    return: qoe, energy_eff, utility
    """

    cahing_contents = rsu.caching_contents

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
        if not hconn.is_cloud and hconn.veh.job.job_type not in cahing_contents:
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
