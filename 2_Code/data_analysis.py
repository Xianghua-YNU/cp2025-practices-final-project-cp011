"""
小天体轨道模拟：包括双星参数设置，初始微扰动参数设置（在惯性系下）， 三种引力扰动源参数设置（在惯性系下），
L点求解，输出并绘制 L点位置，共转系下的原始加速度， 惯性系→共转系转换，小天体初始微扰动参数（转换为共转系），
 数值积分三体源轨迹，扰动产生的额外加速度计算，迭代并绘制轨迹等
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# —— 双星参数设置 —— 
mu = 0.040         # 质量比 m2/(m1+m2)
dt = 0.001        # 时间步长
n_steps = 628320  # 迭代时间次数（dt=1时,迭代n_steps=2π≈6.28次即完成一个周期）
                  # 需要模拟的双星绕转周期数=2π/dt

# —— 初始微扰动参数设置（在惯性系下） —— 
dx_I, dy_I = 0.001, 0.001
dvx_I, dvy_I = 0.001, 0.001

# —— 三种引力扰动源参数设置（在惯性系下） ——
# 静止扰动源
mus = 0.0           # 静止扰动源质量ms与双星总质量之比 ms/(m1+m2)
xI_static, yI_static = 3, 0.0        # 静止扰动源初始位置
# 圆周扰动源
muc = 0.0              # 圆周扰动源质量mc与双星总质量之比mc/(m1+m2)
d_circ = 2             # 圆周半径
k = 1                    # 角速度kΩ的倍数k（正数表示扰动源逆时针运动）
theta0 = 0.0             # 扰动源初始位置与x轴正方向夹角θ0（弧度）
# 三体扰动源
mul = 0.0              # 三体扰动源质量ml(是L不是1)与双星总质量之比ml/(m1+m2)
xl_I, yl_I = 1.832981, 0      # 三体扰动源初始位置、速度参数设置
vxl_I, vyl_I = 0.0, 0.73665     #v=ωr


# —— L点求解 —— 
def dPhi_dx(x):
    r1, r2 = abs(x+mu), abs(x-1+mu)
    return (1-mu)*(x+mu)/r1**3 + mu*(x-1+mu)/r2**3 - x

L1 = fsolve(dPhi_dx,  0.5)[0]
L2 = fsolve(dPhi_dx,  1.5)[0]
L3 = fsolve(dPhi_dx, -1.0)[0]
L4 = (0.5 - mu,  np.sqrt(3)/2)
L5 = (0.5 - mu, -np.sqrt(3)/2)

primaries = {'m1':(-mu,0.0), 'm2':(1-mu,0.0)}
#lagrange = {'L1':(L1,0.0),'L2':(L2,0.0),'L3':(L3,0.0),'L4':L4,'L5':L5}
lagrange = {'L4':L4}

# —— 命令行输出 —— 
print(f"m2与双星总质量M质量比 mu = {mu:.7f}\n")
print("两大天体坐标 (共转系, 归一化单位):")
for name,(x,y) in primaries.items():
    print(f"  {name}: x = {x:.7f}, y = {y:.7f}")
print("\n拉格朗日点坐标 (共转系, 归一化单位):")
for name,(x,y) in lagrange.items():
    print(f"  {name}: x = {x:.7f}, y = {y:.7f}")

# —— 输出并绘制 L点位置 —— 
plt.figure(figsize=(6,6))
px,py = zip(*primaries.values())
lx,ly = zip(*lagrange.values())
plt.scatter(px,py,c='gray',s=100,label='Primaries')
plt.scatter(lx,ly,c='red',s=50,label='Lagrange Points')
plt.title("Lagrange Points")
plt.legend(); plt.grid(); plt.gca().set_aspect('equal','box')
plt.show()


# —— 共转系下的原始加速度 —— 
def accel_core(x, y, vx, vy, u):
    r1 = np.hypot(x+u, y)         # r1=((x+mu)^2 + y^2)^(1/2)
    r2 = np.hypot(x-1+u, y)       # r1=((x-1+mu)^2 + y^2)^(1/2)
    # 由小天体在共转系下的运动方程导出其加速度
    ax = 2*vy - ((1-u)*(x+u)/r1**3 + u*(x-1+u)/r2**3 - x)
    ay = -2*vx - ((1-u)*y/r1**3 + u*y/r2**3 - y)
    return ax, ay

# —— 惯性系→共转系转换（xI, yI, vxI, vyI→xR, yR, vxR, vyR） —— 
def inertial_to_rot_x(xI, yI, t):
    c, s = np.cos(t), np.sin(t)
    xR =  c*xI + s*yI
    yR = -s*xI + c*yI
    return xR, yR
def inertial_to_rot_x_v(xI, yI, vxI, vyI, t):
    c, s = np.cos(t), np.sin(t)
    xR =  c*xI + s*yI
    yR = -s*xI + c*yI
    vxR =  yR + c*vxI + s*vyI
    vyR = -xR - s*vxI + c*vyI
    return xR, yR, vxR, vyR

# 小天体初始微扰动参数（转换为共转系）
dx, dy, dvx, dvy = inertial_to_rot_x_v(dx_I, dy_I, dvx_I, dvy_I, 0.0)

# —— 数值积分三体源轨迹 ——
if mul > 0:
    traj3 = np.zeros((n_steps, 2)) # 三体源位置坐标数组
    xl, yl, vxl, vyl = inertial_to_rot_x_v(xl_I, yl_I, vxl_I, vyl_I, 0.0)
    state3 = np.array([xl, yl, vxl, vyl])
    for i in range(n_steps):
        x, y, vx, vy = state3
        ax, ay = accel_core(x, y, vx, vy, mu)
        def deriv3(s):
            xx, yy, vxx, vyy = s
            a1, a2 = accel_core(xx, yy, vxx, vyy, mu)
            return np.array([vxx, vyy, a1, a2])
        k1 = deriv3(state3)
        k2 = deriv3(state3 + 0.5*dt*k1)
        k3 = deriv3(state3 + 0.5*dt*k2)
        k4 = deriv3(state3 +     dt*k3)
        state3 += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        traj3[i] = state3[:2]        # 前两分量(xl,yl)存入traj3数组的第i行
            

# —— 扰动产生的额外加速度计算 —— 
def accel_perturb(x, y, step_index, t):
    ax_p = ay_p = 0.0      #初始化由三大扰动源产生的额外加速度，以便后续累加
    # 静止源
    if mus > 0:
        # 将扰动源在惯性系下的位置和初始速度转换成共转系
        xs, ys = inertial_to_rot_x(xI_static, yI_static, t)
        dxs, dys = x - xs, y - ys
        rs = np.hypot(dxs, dys)             # 小天体到静止源的距离
        ax_p += mus * (dxs/rs**3)
        ay_p += mus * (dys/rs**3)
    # 圆周源
    if muc > 0:
        # —— 逆时针圆周：初始相位θ0 —— 
        xI_c = d_circ * np.cos(k*t + theta0)
        yI_c = d_circ * np.sin(k*t + theta0)
        # 将扰动源在惯性系下的位置和初始速度转换成共转系
        xc, yc = inertial_to_rot_x(xI_c, yI_c, t)
        dxc, dyc = x - xc, y - yc
        rc = np.hypot(dxc, dyc)             # 小天体到圆周源的距离
        ax_p += muc * (dxc/rc**3)
        ay_p += muc * (dyc/rc**3)
    # 三体源
    if mul > 0:
        xl, yl = traj3[step_index]
        dxl, dyl = x - xl, y - yl
        rl = np.hypot(dxl, dyl)             # 小天体到直线源的距离
        ax_p += mul * (dxl/rl**3)
        ay_p += mul * (dyl/rl**3)
    return ax_p, ay_p

# —— 迭代并绘制轨迹 —— 
for name,(x0,y0) in lagrange.items():
    x, y = x0+dx, y0+dy
    vx, vy = dvx, dvy
    traj = np.zeros((n_steps,2))
    t = 0.0
    def deriv(s, step_index, ti):
        xx,yy,vxx,vyy = s
        ac = accel_core(xx,yy,vxx,vyy,mu)
        ap = accel_perturb(xx,yy,step_index,ti)
        return np.array([vxx, vyy, ac[0]+ap[0], ac[1]+ap[1]])
    for i in range(n_steps):
        ax, ay = accel_core(x,y,vx,vy,mu)   # 共转系下的原始加速度
        apx, apy = accel_perturb(x,y,i,t)  # 共转系下的扰动额外加速度
        
        # 加上扰动项后的运动方程为：
        # ax + apx = 2*vy - ((1-mu)*(x+mu)/r1**3 + mu*(x-1+mu)/r2**3 - x)
        # ay + apy = -2*vx - ((1-mu)*y/r1**3 + mu*y/r2**3 - y)
        # 得：
        # ax = 2*vy - ((1-mu)*(x+mu)/r1**3 + mu*(x-1+mu)/r2**3 - x) - apx
        # ay = -2*vx - ((1-mu)*y/r1**3 + mu*y/r2**3 - y) - apy
        
        ax -= apx; ay -= apy
        # RK4迭代计算
        state = np.array([x,y,vx,vy])
        
        k1 = deriv(state,             i,      t)
        k2 = deriv(state + 0.5*dt*k1, i, t+0.5*dt)
        k3 = deriv(state + 0.5*dt*k2, i, t+0.5*dt)
        k4 = deriv(state +     dt*k3, i, t+dt)
        state += (dt/6)*(k1+2*k2+2*k3+k4)
        x,y,vx,vy = state; t += dt
        traj[i] = (x,y)
    plt.figure(figsize=(6,6))
    px,py = zip(*primaries.values()); lx,ly = zip(*lagrange.values())
    px,py = zip(*primaries.values())
    plt.scatter(px, py,
                c='gray', s=100, marker='o',
                label='Primaries',
                zorder=1)
    lx,ly = zip(*lagrange.values())
    plt.scatter(lx, ly,
                c='red', s=50, marker='o',
                label='Lagrange Points',
                zorder=2)
    #plt.scatter(0.5 - mu + 0.03846*0.5,(np.sqrt(3)/2)*(1.03846),c='blue',s=50,label='Broder Points')
    #plt.scatter(0.5 - mu - 0.02846*0.5,(np.sqrt(3)/2)*(0.97115),c='blue',s=50,label='Broder Points')
    plt.plot(traj[:,0],traj[:,1],lw=1,label=name)
    plt.title(f"Trajectory from {name}")
    plt.legend(); plt.grid(); plt.gca().set_aspect('equal','box')
    plt.show()
