from edge_viot_env import EdgeVIoTEnv, raw_env
from pettingzoo.test import parallel_api_test, api_test

if __name__ == "__main__":
    env = EdgeVIoTEnv(render_mode="ansi")
    parallel_api_test(env, num_cycles=1_000_000)
    api_test(raw_env())
