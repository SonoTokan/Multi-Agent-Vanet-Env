import wandb
import pandas as pd

# 初始化 W&B 运行
run = wandb.init()

api = wandb.Api()

# 获取运行数据, url
run_path = "name/xxxx/runs/xxxxxxx"
run = api.run(run_path)

# 获取历史数据
history = run.history(keys=None, samples=10241)

# 将历史数据转换为 DataFrame
df = pd.DataFrame(history)

# 保存为 CSV 文件
df.to_csv("london_eval_rmappo_ts.csv", index=False)

print("数据已成功导出")
