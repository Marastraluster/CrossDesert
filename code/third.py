import numpy as np

# ==========================================
# 1. 基础参数设定 (第三关)
# ==========================================
DAYS = 10
START_NODE = 1
END_NODE = 13
MINE_NODE = 9
# 状态节点扩展：1~13为常规位置，14专门代表"已在矿山停留(可挖矿状态)"
# 这是为了满足规则：到达矿山当天不能挖矿
NUM_STATES = 14 

# 资源消耗参数 [晴朗(0), 高温(1)]
base_w = np.array([3, 9])
base_f = np.array([4, 9])

# 根据前两关历史数据推断的先验天气概率 (排除沙暴)
# 晴朗: 0.44, 高温: 0.56
P_weather = np.array([0.44, 0.56]) 

# 经济与负重参数
pW, pF = 5, 10
wW, wF = 3, 2
WEIGHT_LIMIT = 1200
MAX_W = WEIGHT_LIMIT // wW  # 最大水容量 400箱
MAX_F = WEIGHT_LIMIT // wF  # 最大食物容量 600箱
BASE_INCOME = 200

# ---------------------------------------------------------
# 【重要】请填写第三关真实的地图连线
# ---------------------------------------------------------
adj_list = {
    1: [2, 4, 5],
    2: [1, 3, 4],
    3: [2, 4, 8, 9],
    4: [1, 2, 3, 5, 7, 9, 11],
    5: [1, 4, 6],
    6: [5, 7, 12, 13],
    7: [4, 6, 11, 12],
    8: [3, 9],
    9: [3, 4, 8, 10, 11],
    10: [9, 11, 13],
    11: [4, 7, 9, 10, 12],
    12: [6, 7, 11, 13],
    13: [6, 10, 12]
}

# ==========================================
# 2. 动态规划核心 (逆向推导)
# ==========================================
print("正在构建并求解 MDP 价值张量，请稍候...")

# 初始化 DP 价值表 V[pos, w, f, weather]
# 记录在某一天、某个位置、携带特定物资、看到特定天气时的【未来最大期望收益】
V = np.full((NUM_STATES + 1, MAX_W + 1, MAX_F + 1, 2), -np.inf)

# 预计算终点折现矩阵 (所有有效负重状态的清算价值)
w_grid, f_grid = np.ogrid[0:MAX_W + 1, 0:MAX_F + 1]
liquidation_value = 0.5 * (w_grid * pW + f_grid * pF)
valid_mask = (w_grid * wW + f_grid * wF <= WEIGHT_LIMIT)

# 逆向时间循环：从最后一天(Day 10)推导回第1天(Day 1)
for t in range(DAYS, 0, -1):
    V_next = V.copy()
    V = np.full((NUM_STATES + 1, MAX_W + 1, MAX_F + 1, 2), -np.inf)

    # 【终点吸收态】一旦到达终点，游戏结束，价值恒定为折现值
    for w_th in range(2):
        V[END_NODE, :, :, w_th] = np.where(valid_mask, liquidation_value, -np.inf)

    # 计算明天的天气期望价值 E(V_next)
    E_V_next = P_weather[0] * V_next[:, :, :, 0] + P_weather[1] * V_next[:, :, :, 1]

    for pos in range(1, NUM_STATES + 1):
        if pos == END_NODE:
            continue
            
        real_node = MINE_NODE if pos == 14 else pos

        for weather in range(2):
            bw, bf = base_w[weather], base_f[weather]

            # 动作 1: 停留 (消耗 1 倍)
            cost_w, cost_f = bw * 1, bf * 1
            next_pos_stay = 14 if real_node == MINE_NODE else pos
            V_stay = np.full((MAX_W + 1, MAX_F + 1), -np.inf)
            if cost_w <= MAX_W and cost_f <= MAX_F:
                V_stay[cost_w:, cost_f:] = E_V_next[next_pos_stay, :-cost_w, :-cost_f]

            # 动作 2: 行走 (消耗 2 倍)
            cost_w_m, cost_f_m = bw * 2, bf * 2
            V_move = np.full((MAX_W + 1, MAX_F + 1), -np.inf)
            for neighbor in adj_list[real_node]:
                V_m = np.full((MAX_W + 1, MAX_F + 1), -np.inf)
                if cost_w_m <= MAX_W and cost_f_m <= MAX_F:
                    V_m[cost_w_m:, cost_f_m:] = E_V_next[neighbor, :-cost_w_m, :-cost_f_m]
                V_move = np.maximum(V_move, V_m)

            # 动作 3: 挖矿 (仅限已在矿山且就绪的状态 14，消耗 3 倍，获得 200 收益)
            V_mine = np.full((MAX_W + 1, MAX_F + 1), -np.inf)
            if pos == 14:
                cost_w_mi, cost_f_mi = bw * 3, bf * 3
                if cost_w_mi <= MAX_W and cost_f_mi <= MAX_F:
                    V_mine[cost_w_mi:, cost_f_mi:] = E_V_next[14, :-cost_w_mi, :-cost_f_mi] + BASE_INCOME

            # 当前状态价值 = 最优动作带来的期望价值
            V[pos, :, :, weather] = np.maximum.reduce([V_stay, V_move, V_mine])
            # 无效负重状态设为极小值
            V[pos, ~valid_mask, weather] = -np.inf

# ==========================================
# 3. 第0天：决定初始物资购买量
# ==========================================
# 第0天购买时，尚不知道第1天的天气，需取期望
E_V_day1 = P_weather[0] * V[START_NODE, :, :, 0] + P_weather[1] * V[START_NODE, :, :, 1]

# 优化目标：初始 10000元 - 买水开销 - 买食物开销 + 未来期望最大总收益
cost_grid = w_grid * pW + f_grid * pF
total_expected_funds = 10000 - cost_grid + E_V_day1
total_expected_funds = np.where(valid_mask, total_expected_funds, -np.inf)

best_val = np.max(total_expected_funds)
# 提取使得期望资金最大的买水和买食物的数量
best_w, best_f = np.unravel_index(np.argmax(total_expected_funds), total_expected_funds.shape)

print("="*50)
print("MDP 策略计算完成！")
if best_val > 0:
    print(f"最大期望最终保留资金: {best_val:.2f} 元")
    print(f"第0天最优购买策略 -> 水: {best_w} 箱, 食物: {best_f} 箱")
    print("\n【游戏指南】")
    print("在实际游戏(Day 1~10)中，您只需要根据当前所处位置和今日天气，")
    print("去我们计算出的 V[pos, w, f, weather] 张量中查询上下左右相邻状态，")
    print("哪个动作引向的期望值最大，就执行该动作，这就是本题要求解的【最优策略函数】。")
else:
    print("当前地图下无存活可能（可能是占位地图无法在10天内走到终点，请替换真实地图后重试）。")