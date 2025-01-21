import sys

sys.path.append("./")


def utility():
    """
    $$
    compute: l_{r,n,m}^t = \cfrac{S_{n,m}^t q_{r}}{C_{comp,n,m}^t}
    encode: l_{e,n,m}^t= \cfrac{S_{n,m}^t q_{e}}{C_{comp,n,m}^t}
    transmission: l_{t,n,m}^t = \cfrac{S^t_n}{R^t_{n,m}}
    \mathbf{QoE}_n^t=\mu b_n^t-\delta l_n^t-\omega(b_n^{t+1}-b_n^{t})-\psi(l_n^{t+1}-l_n^{t})
    no caching: + 25% lantency
    $$

    """
