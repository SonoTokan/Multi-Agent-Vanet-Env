import sys

sys.path.append("./")
from vanet_env.entites import Rsu


def utility(rsu: Rsu):
    """
    $$
    compute: l_{r,n,m}^t = \cfrac{S_{n,m}^t q_{r}}{C_{comp,n,m}^t}
    encode: l_{e,n,m}^t= \cfrac{S_{n,m}^t q_{e}}{C_{comp,n,m}^t}
    transmission: l_{t,n,m}^t = \cfrac{S^t_n}{R^t_{n,m}}
    \mathbf{QoE}_n^t=\mu b_n^t-\delta l_n^t-\omega(b_n^{t+1}-b_n^{t})-\psi(l_n^{t+1}-l_n^{t})
    no caching: + 25% lantency
    $$
    """
    bw_alloc = rsu.bw_alloc
    cp_alloc = rsu.computation_power_alloc
    cahing_contents = rsu.caching_contents
    l_all = 0
    l_trans = 0
    l_compute = 0
    l_encode = 0
    
    for idx, hconn in enumerate(rsu.handling_jobs):

        # 云固定延迟
        if hconn.is_cloud:
            ...
        # 不是本rsu连接的需要计算跳数，不过bw_alloc可以不计
        elif hconn.rsu.id != rsu.id:
            bw_alloc[idx] = 0
        else:
            ...
        cahing_contents
