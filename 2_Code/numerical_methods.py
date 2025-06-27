"""
拉格朗日点的求解和可视化：具体包括参数设置，拉格朗日点坐标求解，2D散点图，3D势面图，以及天体和拉格朗日点
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D

# —— 参数设置 —— 
mu = 0.1  # 质量比 m2/(m1+m2)

# —— 求解 L1–L3 —— 
def dPhi_dx(x):
    r1 = abs(x + mu)
    r2 = abs(x - 1 + mu)
    return -(1 - mu) * (x + mu) / r1**3 \
           - mu * (x - 1 + mu) / r2**3 \
           - x

L1 = fsolve(dPhi_dx,  0.5)[0]
L2 = fsolve(dPhi_dx,  1.5)[0]
L3 = fsolve(dPhi_dx, -1.0)[0]
L4 = (0.5 - mu,  np.sqrt(3) / 2)
L5 = (0.5 - mu, -np.sqrt(3) / 2)

# —— 两大天体与拉格朗日点坐标 —— 
primaries = {'m1': (-mu,      0.0),
             'm2': (1 - mu,  0.0)}
lagrange_points = {
    'L1': (L1, 0.0),
    'L2': (L2, 0.0),
    'L3': (L3, 0.0),
    'L4': L4,
    'L5': L5
}

# —— 命令行输出 —— 
print(f"质量比 mu = {mu:.6f}\n")
print("两大天体坐标 (共转系, 归一化单位):")
for name, (x, y) in primaries.items():
    print(f"  {name}: x = {x:.6f}, y = {y:.6f}")

print("\n拉格朗日点坐标 (共转系, 归一化单位):")
for name, (x, y) in lagrange_points.items():
    print(f"  {name}: x = {x:.6f}, y = {y:.6f}")

# —— 第一幅图：2D 散点 —— 
plt.figure(figsize=(6, 4))
# 两大天体
px, py = zip(*primaries.values())
plt.scatter(px, py, s=100, c='gray', marker='o', label='Primaries')
# 拉格朗日点
lx, ly = zip(*lagrange_points.values())
plt.scatter(lx, ly, s=50, c='red', marker='o', label='Lagrange Points')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Lagrange Points and Primaries (mu={mu:.3f})')
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', 'box')
plt.show()

# —— 第二幅图：3D 势面 —— 
def Phi(x, y):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return - (1 - mu) / r1 - mu / r2 - 0.5 * (x**2 + y**2)

# 自动确定 xy 范围，包含所有拉格朗日点及边缘
x_coords = [coord[0] for coord in lagrange_points.values()]
y_coords = [coord[1] for coord in lagrange_points.values()]
x_min, x_max = min(x_coords)-0.5, max(x_coords)+0.5
y_min, y_max = min(y_coords)-0.5, max(y_coords)+0.5

x_vals = np.linspace(x_min, x_max, 200)
y_vals = np.linspace(y_min, y_max, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = Phi(X, Y)

# 只显示 Phi >= -5 的区域
Z_clipped = np.where(Z >= -5, Z, np.nan)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, Z_clipped,
    cmap='viridis',
    edgecolor='none',
    alpha=0.8
)

# 绘制两大天体和拉格朗日点
px, py = zip(*primaries.values())
pz = [Phi(x, y) for x, y in primaries.values()]
ax.scatter(px, py, pz, color='gray', s=50, label='Primaries')

lx, ly = zip(*lagrange_points.values())
lz = [Phi(x, y) for x, y in lagrange_points.values()]
ax.scatter(lx, ly, lz, color='red', s=50, label='Lagrange Points')

# 设置坐标轴与标题
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Effective Potential Φ')
ax.set_title('3D Effective Potential Surface (Φ ≥ -5)')
ax.set_zlim(-5, 0)
ax.legend()

plt.show()
