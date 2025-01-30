import wandb
import pandas as pd

# 初始化 W&B 运行
run = wandb.init()

api = wandb.Api()

# 获取运行数据
run_path = "1399858762-fjnu/Vanet/runs/8szfrqb8"
run = api.run(run_path)

# 获取历史数据
history = run.history(keys=None, samples=10241)

# 将历史数据转换为 DataFrame
df = pd.DataFrame(history)

# 保存为 CSV 文件
df.to_csv("wandb_history_data.csv", index=False)

print("数据已成功导出到 wandb_history_data.csv")
