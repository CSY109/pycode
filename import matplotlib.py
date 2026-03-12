import matplotlib.pyplot as plt
import numpy as np

# 1. 准备示例数据
x = np.linspace(0, 10, 50)  # 生成0到10之间的50个均匀数据点（x轴数据）
y1 = np.sin(x)               # 正弦曲线（折线图数据）
y2 = np.random.randn(50)     # 随机正态分布数据（散点图数据）
categories = ['A', 'B', 'C', 'D', 'E']  # 分类标签（柱状图x轴）
values = [15, 22, 18, 25, 12]           # 分类对应的值（柱状图y轴）

# 2. 创建画布与子图（1行3列布局，方便同时展示3种图表）
plt.figure(figsize=(15, 5))  # 设置画布大小（宽15，高5）

# 子图1：折线图（展示趋势）
plt.subplot(1, 3, 1)  # 位置：第1行、第3列、第1个
plt.plot(x, y1, color='blue', linewidth=2, label='sin(x)')  # 绘制折线
plt.title('折线图：sin(x)曲线')  # 标题
plt.xlabel('x')  # x轴标签
plt.ylabel('sin(x)')  # y轴标签
plt.legend()  # 显示图例
plt.grid(alpha=0.3)  # 显示网格（透明度0.3，不遮挡数据）

# 子图2：散点图（展示分布）
plt.subplot(1, 3, 2)  # 位置：第1行、第3列、第2个
plt.scatter(x, y2, color='red', s=30, alpha=0.6, label='随机数据')  # 绘制散点
plt.title('散点图：随机正态分布')
plt.xlabel('x')
plt.ylabel('随机值')
plt.legend()
plt.grid(alpha=0.3)

# 子图3：柱状图（展示分类对比）
plt.subplot(1, 3, 3)  # 位置：第1行、第3列、第3个
plt.bar(categories, values, color='green', alpha=0.7)  # 绘制柱状图
plt.title('柱状图：分类数据对比')
plt.xlabel('分类')
plt.ylabel('数值')
# 在柱子顶部添加数值标签
for i, v in enumerate(values):
    plt.text(i, v + 0.5, str(v), ha='center')  # ha='center' 让文字水平居中

# 调整子图间距，避免标签重叠
plt.tight_layout()
# 显示图表
plt.show()