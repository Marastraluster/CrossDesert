import numpy as np
from scipy.optimize import linprog
import copy

# ==========================================
# 1. 环境与参数
# ==========================================
weather_list = [0, 1, 0, 0, 0, 0, 1, 1, 1, 1] 
base_c_water = [3, 9, 10]
base_c_food = [4, 9, 10]

PRICE_W = 5
PRICE_F = 10
INIT_FUNDS = 10000
MINE_REWARD = 200

# ==========================================
# 2. 策略优化器: 精确物资计算 (核心优化点)
# ==========================================
def optimize_resources(daily_nodes, daily_actions, collision_tolerance=0):
    """
    根据动作序列，计算最低存活所需物资。
    collision_tolerance: 容忍几次被踩踏(2k倍消耗)。0表示赌绝对不重合，1表示预防1次重合。
    """
    req_w, req_f = 0, 0
    tolerance_left = collision_tolerance
    
    for t in range(10):
        w_type = weather_list[t]
        act = daily_actions[t]
        
        mult = 1
        if act == 'move':
            # 如果还有容错额度，按碰撞(4倍)计算，否则按单人(2倍)计算
            if tolerance_left > 0:
                mult = 4
                tolerance_left -= 1
            else:
                mult = 2
        elif act == 'mine':
            mult = 3
            
        req_w += mult * base_c_water[w_type]
        req_f += mult * base_c_food[w_type]
        
    return req_w, req_f

def create_strategy(name, nodes, actions, tolerance):
    w, f = optimize_resources(nodes, actions, tolerance)
    # 检查负重约束
    if 3 * w + 2 * f > 1200:
        return None # 超重，策略无效
        
    return {
        'name': f"{name}_Tol{tolerance}",
        'buy_water': w,
        'buy_food': f,
        'daily_nodes': nodes,
        'daily_actions': actions
    }

# ==========================================
# 3. 策略变体生成器 (时间错峰扩增)
# ==========================================
def generate_time_shifted_variants(base_nodes, base_actions, base_name):
    """通过在前面插入stay，自动生成错峰出行的策略"""
    variants = []
    # 原版策略 (容忍0次和容忍1次碰撞)
    for tol in [0, 1]:
        s = create_strategy(base_name, base_nodes, base_actions, tol)
        if s: variants.append(s)
    
    # 错峰变体: 第1天stay，把后面的动作顺延，去掉最后一天
    if base_actions[-1] == 'stay' or base_actions[-1] == 'mine': # 确保最后一天不是必须的move
        shifted_nodes = [base_nodes[0]] + base_nodes[:-1]
        shifted_actions = ['stay'] + base_actions[:-1]
        for tol in [0, 1]:
            s = create_strategy(f"{base_name}_Delay1", shifted_nodes, shifted_actions, tol)
            if s: variants.append(s)
            
    return variants

# ==========================================
# 4. 拥挤博弈模拟器 (同上一版)
# ==========================================
def simulate_game(path_A, path_B):
    funds = [INIT_FUNDS, INIT_FUNDS]
    water = [path_A['buy_water'], path_B['buy_water']]
    food = [path_A['buy_food'], path_B['buy_food']]
    funds[0] -= (water[0]*PRICE_W + food[0]*PRICE_F)
    funds[1] -= (water[1]*PRICE_W + food[1]*PRICE_F)
    alive = [True, True]
    
    for t in range(10):
        w_type = weather_list[t]
        c_w = base_c_water[w_type]
        c_f = base_c_food[w_type]
        n_A, n_B = path_A['daily_nodes'][t], path_B['daily_nodes'][t]
        a_A, a_B = path_A['daily_actions'][t], path_B['daily_actions'][t]
        
        col_move = (n_A == n_B) and (a_A == 'move') and (a_B == 'move')
        col_mine = (n_A == 9) and (n_B == 9) and (a_A == 'mine') and (a_B == 'mine')
        
        for p, (act, col) in enumerate(zip([a_A, a_B], [col_move, col_move])):
            if not alive[p]: continue
            mult = 1
            if act == 'move': mult = 4 if col else 2
            elif act == 'mine': mult = 3
                
            water[p] -= mult * c_w
            food[p] -= mult * c_f
            if water[p] < 0 or food[p] < 0:
                alive[p], funds[p] = False, 0
                
        if col_mine and alive[0] and alive[1]:
            funds[0] += MINE_REWARD / 2; funds[1] += MINE_REWARD / 2
        else:
            if a_A == 'mine' and alive[0]: funds[0] += MINE_REWARD
            if a_B == 'mine' and alive[1]: funds[1] += MINE_REWARD

    for p, path in enumerate([path_A, path_B]):
        if alive[p] and path['daily_nodes'][-1] == 13:
            funds[p] += 0.5 * PRICE_W * water[p] + 0.5 * PRICE_F * food[p]
        else:
            funds[p] = 0
    return funds[0], funds[1]

# ==========================================
# 5. 纳什均衡求解与策略池组装
# ==========================================
if __name__ == "__main__":
    # 基础路线库 (仅定义核心骨架)
    routes = [
        # 激进中路
        ("MidRush", 
         [4, 4, 9, 9, 9, 9, 11, 13, 13, 13], 
         ['move', 'stay', 'move', 'mine', 'mine', 'mine', 'move', 'move', 'stay', 'stay']),
        # 稳妥上路
        ("TopRush", 
         [2, 2, 8, 9, 9, 9, 10, 13, 13, 13], 
         ['move', 'stay', 'move', 'move', 'mine', 'mine', 'move', 'move', 'stay', 'stay']),
        # 下路避让
        ("BotDodge", 
         [5, 5, 6, 13, 13, 13, 13, 13, 13, 13], 
         ['move', 'stay', 'move', 'move', 'stay', 'stay', 'stay', 'stay', 'stay', 'stay'])
    ]
    
    # 自动扩增策略池
    pool = []
    for name, nodes, actions in routes:
        pool.extend(generate_time_shifted_variants(nodes, actions, name))
        
    print(f"✅ 策略池构建完成，共生成 {len(pool)} 条有效博弈策略。")
    print("-" * 50)
    
    n = len(pool)
    payoff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            reward, _ = simulate_game(pool[i], pool[j])
            payoff_matrix[i, j] = reward
            
    # 线性规划求均衡
    c = np.zeros(n + 1)
    c[-1] = -1 
    A_ub = np.zeros((n, n + 1))
    for j in range(n):
        A_ub[j, :n] = -payoff_matrix[:, j]
        A_ub[j, -1] = 1
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1
    b_eq = np.array([1])
    bounds = [(0, 1) for _ in range(n)] + [(None, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        print(f"🎯 均衡期望资金: {res.x[-1]:.2f} 元")
        print("\n最优混合策略分布:")
        for idx, prob in enumerate(res.x[:n]):
            if prob > 0.001:  # 只打印概率大于0.1%的策略
                s = pool[idx]
                print(f"[{prob*100:5.1f}%] {s['name']:<18} | 买水:{s['buy_water']:3d} 买粮:{s['buy_food']:3d} | 动作: {s['daily_actions'][:4]}...")
    else:
        print("求解失败")