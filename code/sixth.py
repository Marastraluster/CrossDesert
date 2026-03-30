import numpy as np
import itertools

# ==========================================
# 1. 关卡参数 (第六关 n=3)
# ==========================================
NUM_PLAYERS = 3
TOTAL_DAYS = 30
MAX_WEIGHT = 1200
PRICE_W, PRICE_F = 5, 10
MINE_REWARD = 1000

CONS_W = [3, 9, 10] # 晴, 高温, 沙暴
CONS_F = [4, 9, 10]

# 天气概率 (假设: 晴 50%, 高温 40%, 沙暴 10%)
PROB_WEATHER = [0.5, 0.4, 0.1] 

# 地图节点
START_NODE, VILLAGE_NODE, MINE_NODE, END_NODE = 1, 14, 18, 25

def is_adj(n1, n2):
    if n1 == n2: return True
    r1, c1 = (n1 - 1) // 5, (n1 - 1) % 5
    r2, c2 = (n2 - 1) // 5, (n2 - 1) % 5
    return abs(r1 - r2) + abs(c1 - c2) == 1

def get_valid_actions(node, weather):
    """获取当前节点合法的物理动作"""
    actions = [{'act': 'stay', 'target': node}]
    if weather != 2: # 非沙暴天气可移动
        for target in range(1, 26):
            if is_adj(node, target) and node != target:
                actions.append({'act': 'move', 'target': target})
    if node == MINE_NODE:
        actions.append({'act': 'mine', 'target': node})
    return actions

# ==========================================
# 2. 底层指引: 启发式单人价值函数 V(s)
# ==========================================
def get_expected_V(node, water, food):
    """
    【注】在实际论文中，这里是通过逆向 DP 查表获取的精确 V 值。
    由于穷举 30 天状态空间太大，这里用“曼哈顿距离启发式评估”代替 DP 表，
    逻辑：离终点越近、物资越充沛，期望价值 V 越大。
    """
    if water < 0 or food < 0: return -99999 # 死路一条
    if node == END_NODE: return 0.5 * PRICE_W * water + 0.5 * PRICE_F * food
    
    # 距离终点的曼哈顿距离
    r1, c1 = (node - 1) // 5, (node - 1) % 5
    r2, c2 = (END_NODE - 1) // 5, (END_NODE - 1) % 5
    dist = abs(r1 - r2) + abs(c1 - c2)
    
    # 启发式评估：物资的折现价值 - 走到终点的预估消耗 + 矿山潜在收益
    estimated_safe_funds = (water * PRICE_W + food * PRICE_F) * 0.5
    penalty = dist * 200 # 距离越远，潜在风险越大
    bonus = 1500 if node == MINE_NODE else (500 if dist < 3 else 0)
    
    return estimated_safe_funds - penalty + bonus

# ==========================================
# 3. 核心：逐日联合收益评估 (计算 3 人真实支付)
# ==========================================
def evaluate_joint_action(states, actions, weather):
    """计算三人在某一天采取特定联合动作后的期望收益 U"""
    next_states = []
    payoffs = [0, 0, 0]
    
    # 解析意图 (简化的物资计算，不涉及村庄买卖穷举)
    cw, cf = CONS_W[weather], CONS_F[weather]
    
    for i in range(3):
        st = states[i].copy()
        act = actions[i]
        
        mult = 1
        # 碰撞惩罚计算
        if act['act'] == 'move':
            # 同日同目标的 move 算作碰撞
            k_move = sum(1 for a in actions if a['act'] == 'move' and a['target'] == act['target'])
            mult = 2 * k_move if k_move > 0 else 2
        elif act['act'] == 'mine':
            mult = 3
            
        st['water'] -= mult * cw
        st['food'] -= mult * cf
        st['node'] = act['target']
        
        # 即时收益 (矿山平摊)
        if act['act'] == 'mine':
            k_mine = sum(1 for a in actions if a['act'] == 'mine')
            payoffs[i] += MINE_REWARD / k_mine
            
        next_states.append(st)
        
    # 结合底层 V 值计算总支付 U = R_today + V_next
    for i in range(3):
        if next_states[i]['water'] < 0 or next_states[i]['food'] < 0:
            payoffs[i] = -99999 # 饿死，极大惩罚
        else:
            payoffs[i] += get_expected_V(next_states[i]['node'], next_states[i]['water'], next_states[i]['food'])
            
    return payoffs

# ==========================================
# 4. 博弈求解器：迭代最佳应对 (Iterated Best Response)
# ==========================================
def solve_daily_game(states, weather):
    """
    基于当天的局势，寻找 3 人博弈的近似纳什均衡。
    通过“假设别人不变，我选对自己最有利的”不断迭代，直到策略收敛。
    """
    actions_pool = [get_valid_actions(states[i]['node'], weather) for i in range(3)]
    
    # 初始假设：大家都原地不动 (stay)
    current_strategy = [0, 0, 0] 
    
    max_iter = 50
    for _ in range(max_iter):
        strategy_changed = False
        
        for i in range(3): # 轮流优化每个人
            best_payoff = -float('inf')
            best_a_idx = current_strategy[i]
            
            # 遍历玩家 i 的所有可行物理动作
            for a_idx, act in enumerate(actions_pool[i]):
                test_actions = [actions_pool[p][current_strategy[p]] for p in range(3)]
                test_actions[i] = act # 换成玩家 i 正在评估的动作
                
                payoffs = evaluate_joint_action(states, test_actions, weather)
                if payoffs[i] > best_payoff:
                    best_payoff = payoffs[i]
                    best_a_idx = a_idx
                    
            if best_a_idx != current_strategy[i]:
                current_strategy[i] = best_a_idx
                strategy_changed = True
                
        if not strategy_changed:
            break # 达到纯策略纳什均衡或局部最优
            
    # 返回当天的最优联合动作
    return [actions_pool[i][current_strategy[i]] for i in range(3)]

# ==========================================
# 5. 第六关完整滚动时域仿真 (Rolling Horizon)
# ==========================================
if __name__ == "__main__":
    print("🚀 启动基于滚动时域马尔可夫博弈的第六关求解器...\n")
    
    # 初始状态 (假设第 0 天已在起点购买了充足物资)
    # 在优秀论文中，初始购买量也是通过外层网格搜索确定的
    states = [
        {'id': 0, 'node': 1, 'water': 120, 'food': 120, 'funds': 10000 - (120*5+120*10)},
        {'id': 1, 'node': 1, 'water': 120, 'food': 120, 'funds': 10000 - (120*5+120*10)},
        {'id': 2, 'node': 1, 'water': 120, 'food': 120, 'funds': 10000 - (120*5+120*10)}
    ]
    
    # 模拟前 5 天的动态博弈进程
    for day in range(1, 6):
        # 1. 每天早晨揭晓局部天气
        weather = np.random.choice([0, 1, 2], p=PROB_WEATHER)
        w_name = ["晴朗", "高温", "沙暴"][weather]
        print(f"[{'='*15} 第 {day} 天 | 天气: {w_name} {'='*15}]")
        
        # 2. 构建今日博弈矩阵并求解均衡动作
        best_joint_actions = solve_daily_game(states, weather)
        
        # 3. 打印意图并结算物理真实状态
        for i in range(3):
            act = best_joint_actions[i]
            print(f"  ▶ 玩家 {i} 预测均衡策略: 采取 [{act['act'].upper()}] -> 目标节点 {act['target']}")
            
            # (简化的物理结算，仅扣除基础消耗示意，真实结算需按 evaluate_joint_action 扣除 2k 倍)
            mult = 2 if act['act'] == 'move' else 1
            states[i]['water'] -= mult * CONS_W[weather]
            states[i]['food'] -= mult * CONS_F[weather]
            states[i]['node'] = act['target']
            
        print("  [夜晚局势] 剩余水量:", [s['water'] for s in states], " 剩余资金:", [s['funds'] for s in states], "\n")