import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Arial"]})

# 三组数据，代表不同状态所占百分比
data = np.array(
    [
        [7, 90, 3],  # 第一组数据
        [6, 88, 6],  # 第二组数据
        [10, 60, 30],  # 第三组数据
    ]
)

# 将百分比转换为实际计数（假设总数为100）
total_count = 100
data_counts = (data / 100) * total_count

# 与第一组比较的两组数据
comparisons = [data_counts[1], data_counts[2]]
group_names = ["Group 2", "Group 3"]
p_values = []

# 进行两两比较
for i, group in enumerate(comparisons):
    chi2, p, dof, expected = stats.chi2_contingency([data_counts[0], group])
    p_values.append(p)
    print(f"Comparison with {group_names[i]} - Chi-squared: {chi2}, p-value: {p}")

# Bonferroni 校正
alpha = 0.05
corrected_alpha = alpha / len(comparisons)
print(f"Bonferroni corrected alpha: {corrected_alpha}")

# 绘制百分比堆积条形图
fig, ax = plt.subplots(figsize=(2, 2.25))
conditions = ["CTRL", "PUS7 KO", "PUKI KO"]
states = ["Low", "Medium", "High"]
colors = ["#F7EBDB", "#E3A995", "#E37366"]

# 绘制堆积条形图
bar_width = 0.4
bottom = np.zeros(len(conditions))
for i, state in enumerate(states):
    p = ax.bar(
        conditions,
        data[:, i],
        bottom=bottom,
        label=state,
        width=bar_width,
        color=colors[i],
        edgecolor="black",
    )

    bottom += data[:, i]
    # ax.bar_label(p, label_type='center')

# 添加显著性标记
for i, p in enumerate(p_values):
    y = 100  # 在百分比堆积图的顶部添加标记
    x = i + 1
    significance = "*"

    if p < 0.0001:
        significance = "****"
    elif p < 0.001:
        significance = "***"
    elif p < 0.01:
        significance = "**"
    elif p < 0.05:
        significance = "*"
    else:
        significance = "ns"

    plt.text(x, y, significance, ha="center", va="bottom")

# reorder legend row-first
reorder = lambda hl, nc: (sum((lis[i::nc] for i in range(nc)), []) for lis in hl)
handles = ax.get_legend_handles_labels()
n_cols = 3
ax.legend(
    # *reorder(handles, nc=n_cols),
    # ncol=n_cols,
    bbox_to_anchor=(1.8, 1),
    loc="upper right",
)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 设置字体大小
# ax.set_title('Gata', fontsize=14)  # 标题字体大小
# ax.set_xlabel('conditions',fontsize=10)        # x轴标签字体大小
ax.set_ylabel("Values", fontsize=10)  # y轴标签字体大小
ax.tick_params(axis="both", which="both", length=0)
ax.tick_params(axis="both", which="major", labelsize=8)  # 刻度标签字体大小

x_pos = np.arange(len(conditions))
ax.set_xticks(x_pos, conditions, rotation=45, rotation_mode="anchor", ha="right")

plt.xlim(-0.5, 2.5)
plt.ylabel("Gata1$^+$ cell")
plt.ylim(0, 100)
# plt.show()
plt.savefig("gata1.png", dpi=600)