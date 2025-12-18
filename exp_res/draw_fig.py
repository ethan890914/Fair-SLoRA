import json
slora_res = "og_final.jsonl"
fair_res = "fair_final.jsonl"
# disable_res = "disable_final.jsonl"
with open(slora_res, "r") as f:
    jsonl_slora = f.readlines()
with open(fair_res, "r") as f:
    jsonl_fair = f.readlines()
# with open(disable_res, "r") as f:
#     jsonl_disable = f.readlines()

records = []
for line in jsonl_slora:
    line = json.loads(line)
    if line['config']['num_adapters'] == 1: continue
    records.append({
        "model": "slora",
        "num_adapter": line["config"]["num_adapters"],
        "alpha": line["config"]["alpha"],
        "throughput": line["result"]["throughput"],
    })
# for line in jsonl_disable:
#     line = json.loads(line)
#     if line['config']['num_adapters'] == 1: continue
#     records.append({
#         "model": "one time calculation",
#         "num_adapter": line["config"]["num_adapters"],
#         "alpha": line["config"]["alpha"],
#         "throughput": line["result"]["throughput"],
#     })
for line in jsonl_fair:
    line = json.loads(line)
    if line['config']['num_adapters'] == 1: continue
    records.append({
        "model": "fair",
        "num_adapter": line["config"]["num_adapters"],
        "alpha": line["config"]["alpha"],
        "throughput": line["result"]["throughput"],
    })


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(records)

# Quick peek of the result
tp = df.sort_values(["num_adapter", "alpha"], ascending=False, ignore_index=True)
print(tp)

sns.set(style="whitegrid")

g = sns.catplot(
    data=df,
    x="num_adapter",
    y="throughput",
    hue="model",
    col="alpha",
    kind="bar",
    col_wrap=4,
    height=3.5
)

g.set_axis_labels("num_adapter", "Throughput")
g.set_titles("alpha = {col_name}")
plt.savefig("res.png")
