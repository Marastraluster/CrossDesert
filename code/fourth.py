import numpy as np

# ==========================================
# 1. 基础参数设定 (第四关)
# ==========================================
DAYS = 30
START_NODE = 1
END_NODE = 25
VILLAGE_NODE = 14
MINE_NODE = 18

NUM_STATES = 26 # 26代表"已在矿山停留(就绪态)"

base_w = np.array([3, 9, 10])
base_f = np.array([4, 9, 10])

# 天气先验概率 [晴朗: 45%, 高温: 45%, 沙暴: 10%]
P_weather = np.array([0.45, 0.45, 0.10]) 

pW, pF = 5, 10
wW, wF = 3, 2
WEIGHT_LIMIT = 1200
MAX_W = WEIGHT_LIMIT // wW  
MAX_F = WEIGHT_LIMIT // wF  
BASE_INCOME = 1000

# 自动生成第四关 5x5 网格地图连线
adj_list = {i: [] for i in range(1, 26)}
for i in range(1, 26):
    r = (i - 1) // 5 + 1
    c = (i - 1) % 5 + 1
    if c > 1: adj_list[i].append(i - 1)
    if c < 5: adj_list[i].append(i + 1)
    if r > 1: adj_list[i].append(i - 5)
    if r < 5: adj_list[i].append(i + 5)

# ==========================================
# 2. 动态规划核心 (逆向推导)
# ==========================================
print("正在构建并求解 第四关 MDP 价值张量...")
print("包含 O(W+F) 村庄购买降维优化，计算约需十余秒，请稍候...")

# 【核心修复】将 -inf 替换为有限的巨额惩罚，防止期望值被黑洞吞噬
PENALTY = -100000.0 

V = np.full((NUM_STATES + 1, MAX_W + 1, MAX_F + 1, 3), PENALTY, dtype=np.float64)

w_grid, f_grid = np.ogrid[0:MAX_W + 1, 0:MAX_F + 1]
liquidation_value = 0.5 * (w_grid * pW + f_grid * pF)
valid_mask = (w_grid * wW + f_grid * wF <= WEIGHT_LIMIT)

for t in range(DAYS, 0, -1):
    V_next = V.copy()
    V = np.full((NUM_STATES + 1, MAX_W + 1, MAX_F + 1, 3), PENALTY, dtype=np.float64)

    for w_th in range(3):
        V[END_NODE, :, :, w_th] = np.where(valid_mask, liquidation_value, PENALTY)

    E_V_next = P_weather[0] * V_next[:, :, :, 0] + \
               P_weather[1] * V_next[:, :, :, 1] + \
               P_weather[2] * V_next[:, :, :, 2]

    for pos in range(1, NUM_STATES + 1):
        if pos == END_NODE:
            continue
            
        real_node = MINE_NODE if pos == 26 else pos

        for weather in range(3):
            bw, bf = base_w[weather], base_f[weather]

            # 动作 1: 停留
            cost_w, cost_f = bw * 1, bf * 1
            next_pos_stay = 26 if real_node == MINE_NODE else pos
            V_stay = np.full((MAX_W + 1, MAX_F + 1), PENALTY)
            if cost_w <= MAX_W and cost_f <= MAX_F:
                V_stay[cost_w:, cost_f:] = E_V_next[next_pos_stay, :-cost_w, :-cost_f]
            V[pos, :, :, weather] = np.maximum(V[pos, :, :, weather], V_stay)

            # 动作 2: 行走 (沙暴天禁止)
            if weather != 2:
                cost_w_m, cost_f_m = bw * 2, bf * 2
                if cost_w_m <= MAX_W and cost_f_m <= MAX_F:
                    V_move = np.full((MAX_W + 1, MAX_F + 1), PENALTY)
                    for neighbor in adj_list[real_node]:
                        V_m = np.full((MAX_W + 1, MAX_F + 1), PENALTY)
                        V_m[cost_w_m:, cost_f_m:] = E_V_next[neighbor, :-cost_w_m, :-cost_f_m]
                        V_move = np.maximum(V_move, V_m)
                    V[pos, :, :, weather] = np.maximum(V[pos, :, :, weather], V_move)

            # 动作 3: 挖矿
            if pos == 26:
                cost_w_mi, cost_f_mi = bw * 3, bf * 3
                if cost_w_mi <= MAX_W and cost_f_mi <= MAX_F:
                    V_mine = np.full((MAX_W + 1, MAX_F + 1), PENALTY)
                    V_mine[cost_w_mi:, cost_f_mi:] = E_V_next[26, :-cost_w_mi, :-cost_f_mi] + BASE_INCOME
                    V[pos, :, :, weather] = np.maximum(V[pos, :, :, weather], V_mine)

            V[pos, ~valid_mask, weather] = PENALTY

    # 村庄购买机制的价值逆向传播优化
    for weather in range(3):
        for w in range(MAX_W - 1, -1, -1):
            V[VILLAGE_NODE, w, :, weather] = np.maximum(
                V[VILLAGE_NODE, w, :, weather],
                V[VILLAGE_NODE, w+1, :, weather] - 2 * pW
            )
        for f in range(MAX_F - 1, -1, -1):
            V[VILLAGE_NODE, :, f, weather] = np.maximum(
                V[VILLAGE_NODE, :, f, weather],
                V[VILLAGE_NODE, :, f+1, weather] - 2 * pF
            )
        V[VILLAGE_NODE, ~valid_mask, weather] = PENALTY

# ==========================================
# 3. 第0天：决定初始物资购买量
# ==========================================
E_V_day1 = P_weather[0] * V[START_NODE, :, :, 0] + \
           P_weather[1] * V[START_NODE, :, :, 1] + \
           P_weather[2] * V[START_NODE, :, :, 2]

cost_grid = w_grid * pW + f_grid * pF
total_expected_funds = 10000 - cost_grid + E_V_day1
total_expected_funds = np.where(valid_mask, total_expected_funds, PENALTY)

best_val = np.max(total_expected_funds)
best_w, best_f = np.unravel_index(np.argmax(total_expected_funds), total_expected_funds.shape)

print("\n" + "="*50)
print("第四关 MDP 策略求解完成！")
if best_val > -50000:
    print(f"最大期望最终保留资金: {best_val:.2f} 元")
    print(f"第0天最优初始购买策略 -> 水: {best_w} 箱, 食物: {best_f} 箱")
else:
    print("当前参数下生存概率极低或无解，请检查地图数据。")