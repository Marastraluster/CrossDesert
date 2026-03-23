import numpy as np
from scipy.optimize import linprog

# ==========================================
# 1. 核心参数设定 (第五关：13节点地图，10天)
# ==========================================
DAYS = 10

# 依据题目，第三关已知无沙暴。为保证严格性，采用期望日消耗。
# 概率：晴朗 0.44，高温 0.56
# 晴天消耗(3*5+4*10=55), 高温消耗(9*5+9*10=135)
E_cost = 0.44 * 55 + 0.56 * 135  # 每日期望消耗 ≈ 99.8 元

# ==========================================
# 2. 全自动生成策略池 (Strategy Generation)
# ==========================================
strategies = []

# (1) 直接通关流: 1 -> 4 -> 11 -> 13
for delay in range(4):
    path = [1]*delay + [1, 4, 11, 13]
    if len(path) <= 11: # 状态数组长度必须为 DAYS+1 = 11
        path += [13] * (11 - len(path))
        strategies.append({'name': f'Rush_等待{delay}天', 'path': path})

# (2) 主流挖矿流: 1 -> 4 -> 9 -> (挖矿) -> 11 -> 13
for delay in range(4):
    for m_days in range(1, 7): # 挖矿1到6天
        path = [1]*delay + [1, 4, 9] + [9]*m_days + [11, 13]
        if len(path) <= 11:
            path += [13] * (11 - len(path))
            strategies.append({'name': f'Mine_等{delay}天_挖{m_days}天', 'path': path})

# (3) 绕路防撞挖矿流: 1 -> 5 -> 4 -> 9 -> (挖矿) -> 11 -> 13
# 故意绕行 5 号节点，避开死亡公路 1->4
for delay in range(3):
    for m_days in range(1, 6):
        path = [1]*delay + [1, 5, 4, 9] + [9]*m_days + [11, 13]
        if len(path) <= 11:
            path += [13] * (11 - len(path))
            strategies.append({'name': f'Detour_等{delay}天_挖{m_days}天', 'path': path})

num_strats = len(strategies)
print(f"成功生成涵盖 {num_strats} 种战术的巨型策略池。")

# ==========================================
# 3. 构建高维博弈支付矩阵 (Payoff Matrix)
# ==========================================
payoff = np.zeros((num_strats, num_strats))

for i in range(num_strats):
    for j in range(num_strats):
        p1_path = strategies[i]['path']
        p2_path = strategies[j]['path']
        
        p1_money = 10000.0 # 初始资金
        
        # 逐日清算物理规则
        for t in range(DAYS):
            u1, v1 = p1_path[t], p1_path[t+1]
            u2, v2 = p2_path[t], p2_path[t+1]
            
            if u1 == 13: # 玩家1已到达终点，游戏结束不耗费资源
                continue
                
            if u1 != v1: # 【移动状态】
                if (u1 == u2) and (v1 == v2): # 发生同路踩踏！
                    p1_money -= E_cost * 4    # 惩罚 2k = 4倍消耗
                else:
                    p1_money -= E_cost * 2    # 正常 2倍消耗
            else:        # 【停留状态】
                p1_money -= E_cost * 1        # 基础消耗
                if u1 == 9 and v1 == 9:       # 在矿山挖矿
                    if u2 == 9 and v2 == 9:   # 两人都在挖矿！
                        p1_money += 1000 / 2  # 收益惨遭平分
                    else:
                        p1_money += 1000      # 独吞 1000 收益

        payoff[i, j] = p1_money

# ==========================================
# 4. 线性规划求解全局纳什均衡 (LP for Nash Equilibrium)
# ==========================================
c = np.zeros(num_strats + 1)
c[-1] = -1 

A_ub = np.zeros((num_strats, num_strats + 1))
A_ub[:, :-1] = -payoff.T
A_ub[:, -1] = 1
b_ub = np.zeros(num_strats)

A_eq = np.zeros((1, num_strats + 1))
A_eq[0, :-1] = 1
b_eq = np.array([1])

bounds = [(0, 1) for _ in range(num_strats)] + [(None, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res.success:
    probs = res.x[:-1]
    game_value = res.x[-1]
    print("\n" + "="*50)
    print("第五关 最优混合策略纳什均衡已找到！")
    print("="*50)
    for idx in range(num_strats):
        if probs[idx] > 0.001: # 过滤掉概率为 0 的劣策略
            print(f" -> 执行策略 [{strategies[idx]['name']}] 的概率: {probs[idx]:.2%}")
    print(f"\n在此纳什均衡下，玩家面临最强针对时的保底期望资金为: {game_value:.2f} 元")
else:
    print("纳什均衡求解失败！")