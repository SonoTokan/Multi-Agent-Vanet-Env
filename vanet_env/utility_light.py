from collections import defaultdict
from itertools import chain
import sys
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("./")
from vanet_env import env_config
from vanet_env.entites import Rsu, Vehicle, OrderedQueueList
from vanet_env import network, utils


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
    max_connections=env_config.MAX_CONNECTIONS,
    num_cores=env_config.NUM_CORES,
):
    rsu_utility_dict = defaultdict(list)
    hop_penalty_rate = 0.1
    max_ee = 0.2
    qoe_factor = 1 - max_ee

    # 观察reward用
    if time_step >= 50:
        time_step

    # 由veh计算，
    # 只需计算in range car的qoe，这里可修改
    for v_id, veh in vehs.items():
        veh: Vehicle

        if veh.is_cloud:
            qoe = max_qoe * 0.1
            rsu_utility_dict[veh.connected_rsu_id].append(qoe)
            veh.job.qoe = qoe
            continue

        trans_qoe = (
            min(veh.data_rate, env_config.JOB_DR_REQUIRE) / env_config.JOB_DR_REQUIRE
        )
        # n者的process qoe除以n
        rsu_trans_qoe_dict = defaultdict(list)
        rsu_proc_qoe_dict = defaultdict(list)
        trans_qoes = defaultdict(list)
        proc_qoes = defaultdict(list)
        # random理论上来说是0.20
        caching_hit_ratios = {}
        caching_hit_states = defaultdict(list)

        if veh.job.processing_rsus.is_empty():
            # 相当于在连接却无处理，一般不会
            if veh.connected_rsu_id != None:
                # 惩罚之
                rsu_utility_dict[veh.connected_rsu_id].append(0.0)

        job_ratio_all = 0.0
        # 一般是邻居
        for p_rsu in veh.job.processing_rsus:
            # 即single ratio

            # qoe calculate after last for
            if p_rsu is not None:
                p_rsu: Rsu
                p_idx = p_rsu.handling_jobs.index((veh, 0))
                job_ratio = p_rsu.handling_jobs[p_idx][1]
                job_ratio_all += job_ratio

                if job_ratio != 0:
                    process_qoe = (
                        min(
                            p_rsu.real_cp_alloc[p_idx] / job_ratio,
                            env_config.JOB_CP_REQUIRE,
                        )
                        / env_config.JOB_CP_REQUIRE
                    )
                else:
                    process_qoe = 0.0

                trans_rsu: Rsu = rsus[veh.connected_rsu_id]

                if p_rsu.id == veh.connected_rsu_id:
                    qoe = min(process_qoe, trans_qoe)

                    # caching debug
                    if veh.job.job_type in trans_rsu.caching_contents:
                        qoe = min(qoe + qoe * 0.15, 1)
                        caching_hit_states[veh.connected_rsu_id].append(1)
                    else:
                        qoe = max(qoe - qoe * 0.1, 0)
                        caching_hit_states[veh.connected_rsu_id].append(0)

                    veh.job.qoe = qoe

                    trans_rsu.ee = max_ee * (1 - trans_rsu.cp_usage)

                    trans_qoes[veh.connected_rsu_id].append(
                        float(qoe * qoe_factor + trans_rsu.ee)
                    )
                    proc_qoes[p_rsu.id].append(float(qoe * qoe_factor + trans_rsu.ee))

                else:
                    qoe = min(process_qoe, trans_qoe)
                    qoe = max(qoe - qoe * hop_penalty_rate, 0)

                    if veh.job.job_type in rsus[veh.connected_rsu_id].caching_contents:

                        qoe = min(qoe + qoe * 0.15, 1)
                        caching_hit_states[veh.connected_rsu_id].append(1)
                    else:
                        qoe = max(qoe - qoe * 0.1, 0)
                        caching_hit_states[veh.connected_rsu_id].append(0)

                    veh.job.qoe = qoe

                    p_rsu.ee = max_ee * (1 - p_rsu.cp_usage)
                    trans_rsu.ee = max_ee * (1 - trans_rsu.cp_usage)

                    trans_qoes[veh.connected_rsu_id].append(
                        float(qoe * qoe_factor + trans_rsu.ee)
                    )
                    proc_qoes[p_rsu.id].append(float(qoe * qoe_factor + trans_rsu.ee))

        # 计算被处理proc rsu的 avg qoe
        num_proc_rus = len(proc_qoes.keys())
        # 没有进入if-else
        if num_proc_rus == 0:
            continue
        else:
            if job_ratio_all > 1.0:
                assert NotImplementedError("Impossible value")

            flattened_trans_qoes = list(chain.from_iterable(trans_qoes.values()))
            avg_trans_qoes = np.mean(flattened_trans_qoes)
            weighted_avg_trans_qoes = float(avg_trans_qoes * job_ratio_all)

            flattened_proc_qoes = list(chain.from_iterable(proc_qoes.values()))
            avg_proc_qoes = np.mean(flattened_proc_qoes)
            weighted_avg_proc_qoes = float(avg_proc_qoes * job_ratio_all)

            # 展平
            # flattened_values = list(chain.from_iterable(rsu_proc_qoe_dict.values()))
            # sum_qoes = np.sum(flattened_values)
            # avg_qoe = sum_qoes / num_proc_rus

            # 如果要把这个策略清理，需要修改proc_qoes
            for rsu_id, qoes in proc_qoes.items():
                # caching 命中次数除以总caching访问次数
                if len(caching_hit_states[rsu_id]) != 0:
                    caching_hit_ratio = sum(caching_hit_states[rsu_id]) / len(
                        caching_hit_states[rsu_id]
                    )
                    caching_hit_ratios[rsu_id].append(caching_hit_ratio)

                if rsu_id == trans_rsu.id:
                    # 这个trans_rsu id理论上必有，且理论上必是这三个proc中的一个，重复会稀释，需要检查吗
                    rsu_utility_dict[trans_rsu.id].append(weighted_avg_trans_qoes)
                    # 不重复导入
                else:
                    rsu_utility_dict[rsu_id].append(weighted_avg_proc_qoes)
            # 将proc rsu的 avg qoe导入trans rsu

    return rsu_utility_dict, caching_hit_ratios
