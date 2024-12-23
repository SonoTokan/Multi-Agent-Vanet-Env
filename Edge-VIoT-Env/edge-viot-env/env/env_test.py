from edge_viot_env import EdgeVIoTEnv
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = EdgeVIoTEnv(render_mode="ansi")
    parallel_api_test(env, num_cycles=1_000_000)

    env = EdgeVIoTEnv(render_mode="ansi")
    parallel_api_test(env, num_cycles=1_000_000)