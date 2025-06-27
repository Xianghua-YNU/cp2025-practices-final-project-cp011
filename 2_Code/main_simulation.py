"""
判别式：主要是判断五个拉格朗日点的稳定性
"""
import numpy as np

def calculate_gamma(x_Ln, y_Ln, mu):
    """计算二阶导数γ_xx, γ_xy, γ_yy"""
    r1 = np.sqrt((x_Ln + mu)**2 + y_Ln**2)
    r2 = np.sqrt((x_Ln - 1 + mu)**2 + y_Ln**2)
    
    # 防止除以零
    r1 = max(r1, 1e-15)
    r2 = max(r2, 1e-15)
    
    # γ_xx计算
    term1 = (1 - mu)/r1**3
    term2 = mu/r2**3
    term3 = 3*(1 - mu)*(x_Ln + mu)**2/r1**5
    term4 = 3*mu*(x_Ln - 1 + mu)**2/r2**5
    gamma_xx = term1 + term2 - term3 - term4 - 1
    
    # γ_xy计算
    gamma_xy = -3*(1 - mu)*(x_Ln + mu)*y_Ln/r1**5 - 3*mu*(x_Ln - 1 + mu)*y_Ln/r2**5
    
    # γ_yy计算
    term5 = 3*(1 - mu)*y_Ln**2/r1**5
    term6 = 3*mu*y_Ln**2/r2**5
    gamma_yy = term1 + term2 - term5 - term6 - 1
    
    return gamma_xx, gamma_xy, gamma_yy

def stability_analysis(gamma_xx, gamma_xy, gamma_yy, point_type, mu):
    """稳定性分析"""
    D = gamma_xx * gamma_yy - gamma_xy**2
    trace = gamma_xx + gamma_yy
    
    if point_type == "collinear":
        return "不稳定（鞍点）" if gamma_xx > 0 and gamma_yy < 0 else "需要进一步分析"
    
    elif point_type == "triangular":
        if D > 0.25 * (4 - trace)**2:
            return "稳定（满足Routh-Hurwitz条件）"
        elif mu < 0.0385:
            return "稳定（μ < 临界值）"
        else:
            return "不稳定（μ > 临界值）"
    return "未知类型"

# 完整的数据集
lagrange_points = {
    0.035: {
        "L1": (0.7548055, 0.0000000), "L2": (1.2093600, 0.0000000), "L3": (-1.0145810, 0.0000000),
        "L4": (0.4650000, 0.8660254), "L5": (0.4650000, -0.8660254)
    },
    0.036: {
        "L1": (0.7519582, 0.0000000), "L2": (1.2108538, 0.0000000), "L3": (-1.0149974, 0.0000000),
        "L4": (0.4640000, 0.8660254), "L5": (0.4640000, -0.8660254)
    },
    0.037: {
        "L1": (0.7491466, 0.0000000), "L2": (1.2123059, 0.0000000), "L3": (-1.0154139, 0.0000000),
        "L4": (0.4630000, 0.8660254), "L5": (0.4630000, -0.8660254)
    },
    0.038: {
        "L1": (0.7463690, 0.0000000), "L2": (1.2137183, 0.0000000), "L3": (-1.0158303, 0.0000000),
        "L4": (0.4620000, 0.8660254), "L5": (0.4620000, -0.8660254)
    },
    0.039: {
        "L1": (0.7436239, 0.0000000), "L2": (1.2150927, 0.0000000), "L3": (-1.0162467, 0.0000000),
        "L4": (0.4610000, 0.8660254), "L5": (0.4610000, -0.8660254)
    },
    0.040: {
        "L1": (0.7409098, 0.0000000), "L2": (1.2164306, 0.0000000), "L3": (-1.0166631, 0.0000000),
        "L4": (0.4600000, 0.8660254), "L5": (0.4600000, -0.8660254)
    },
    0.041: {
        "L1": (0.7382256, 0.0000000), "L2": (1.2177335, 0.0000000), "L3": (-1.0170795, 0.0000000),
        "L4": (0.4590000, 0.8660254), "L5": (0.4590000, -0.8660254)
    },
    0.042: {
        "L1": (0.7355698, 0.0000000), "L2": (1.2190030, 0.0000000), "L3": (-1.0174959, 0.0000000),
        "L4": (0.4580000, 0.8660254), "L5": (0.4580000, -0.8660254)
    },
    0.044: {
        "L1": (0.7303393, 0.0000000), "L2": (1.2214464, 0.0000000), "L3": (-1.0183286, 0.0000000),
        "L4": (0.4560000, 0.8660254), "L5": (0.4560000, -0.8660254)
    },
    0.045: {
        "L1": (0.7277624, 0.0000000), "L2": (1.2226228, 0.0000000), "L3": (-1.0187449, 0.0000000),
        "L4": (0.4550000, 0.8660254), "L5": (0.4550000, -0.8660254)
    }
}

# 计算结果
for mu in sorted(lagrange_points.keys()):
    print(f"\n{'='*60}")
    print(f"质量比 μ = {mu:.6f}")
    print(f"{'='*60}")
    
    points = lagrange_points[mu]
    for point in ["L1", "L2", "L3", "L4", "L5"]:
        x, y = points[point]
        point_type = "collinear" if point in ["L1", "L2", "L3"] else "triangular"
        
        gamma_xx, gamma_xy, gamma_yy = calculate_gamma(x, y, mu)
        D = gamma_xx * gamma_yy - gamma_xy**2
        stability = stability_analysis(gamma_xx, gamma_xy, gamma_yy, point_type, mu)
        
        print(f"\n{point}点 (x={x:.7f}, y={y:.7f}):")
        print(f"  γ_xx = {gamma_xx:.8f}")
        print(f"  γ_yy = {gamma_yy:.8f}")
        print(f"  γ_xy = {gamma_xy:.8f}")
        print(f"  γ_xx·γ_yy = {gamma_xx*gamma_yy:.8f}")
        print(f"  γ_xy² = {gamma_xy**2:.8f}")
        print(f"  判别式 D = {D:.8f}")
        print(f"  稳定性: {stability}")
    
    print(f"{'-'*60}")

print("\n理论说明:")
print("1. 共线点(L1,L2,L3): 恒为不稳定鞍点")
print("2. 三角点(L4,L5):")
print("   - 当 μ < 0.0385 时稳定")
print("   - 当 μ > 0.0385 时不稳定")
print("   - 临界值 μ₀ ≈ 0.0385")
