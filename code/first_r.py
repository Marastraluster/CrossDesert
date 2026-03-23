import pulp

# ==========================================
# 1. 基础参数与数据输入 (第一关)
# ==========================================
DAYS = 30
START_NODE = 1
END_NODE = 27
MINES = [12]      # 第一关矿山节点
VILLAGES = [15]   # 第一关村庄节点
NODES = list(range(1, 28))

# 天气状况映射 (1:晴朗, 2:高温, 3:沙暴)
weather_seq = [
    2, 2, 1, 3, 1, 2, 3, 1, 2, 2,
    3, 2, 1, 2, 2, 2, 3, 3, 2, 2,
    1, 1, 2, 1, 3, 2, 1, 1, 2, 2
]

# 资源基础消耗 [晴朗, 高温, 沙暴]
WATER_CONSUME = {1: 5, 2: 8, 3: 10}
FOOD_CONSUME  = {1: 7, 2: 6, 3: 10}

WEIGHT_LIMIT = 1200
WATER_WEIGHT = 3
FOOD_WEIGHT = 2
INIT_FUNDS = 10000
BASE_INCOME = 1000
WATER_PRICE = 5
FOOD_PRICE = 10

# 第一关地图邻接表
adj_list = {
            1:  [2, 25],
            2:  [1, 3],
            3:  [2, 4, 25],
            4:  [3, 5, 24, 25],
            5:  [4, 6, 24],
            6:  [5, 7, 23, 24],
            7:  [6, 8, 22],
            8:  [7, 9, 22],
            9:  [8, 10, 15, 16, 17, 21, 22],
            10: [9, 11, 13, 15],
            11: [10, 12, 13],
            12: [11, 13, 14],
            13: [10, 11, 12, 14, 15],
            14: [12, 13, 15, 16],
            15: [9, 10, 13, 14, 16],
            16: [9, 14, 15, 17, 18],
            17: [9, 16, 18, 21],
            18: [16, 17, 19, 20],
            19: [18, 20],
            20: [18, 19, 21],
            21: [9, 17, 20, 22, 23, 27],
            22: [7, 8, 9, 21, 23],
            23: [6, 21, 22, 24, 26],
            24: [4, 5, 6, 23, 25, 26],
            25: [1, 3, 4, 24, 26],
            26: [23, 24, 25, 27],
            27: [21, 23, 26]
}

# ==========================================
# 2. 建立线性规划模型
# ==========================================
prob = pulp.LpProblem("Desert_Crossing_Level1_Stop_Immediately", pulp.LpMaximize)

# --- 决策变量 ---
# 位置变量
x = pulp.LpVariable.dicts("Position", (range(DAYS + 1), NODES), cat="Binary")

# 动作变量
stay = pulp.LpVariable.dicts("Stay", range(1, DAYS + 1), cat="Binary")
move = pulp.LpVariable.dicts("Move", range(1, DAYS + 1), cat="Binary")
mine = pulp.LpVariable.dicts("Mine", range(1, DAYS + 1), cat="Binary")

# 结束状态变量：finished[t] = 1 表示到第t天结束时已到达终点
finished = pulp.LpVariable.dicts("Finished", range(DAYS + 1), cat="Binary")

# 资源变量
water = pulp.LpVariable.dicts("Water", range(DAYS + 1), lowBound=0)
food = pulp.LpVariable.dicts("Food", range(DAYS + 1), lowBound=0)
money = pulp.LpVariable.dicts("Money", range(DAYS + 1), lowBound=0)

# 购买变量
buy_w_start = pulp.LpVariable("BuyWaterStart", lowBound=0, cat="Integer")
buy_f_start = pulp.LpVariable("BuyFoodStart", lowBound=0, cat="Integer")
buy_w_vill = pulp.LpVariable.dicts("BuyWaterVill", range(1, DAYS + 1), lowBound=0, cat="Integer")
buy_f_vill = pulp.LpVariable.dicts("BuyFoodVill", range(1, DAYS + 1), lowBound=0, cat="Integer")

# --- 初始状态约束 ---
prob += x[0][START_NODE] == 1
for i in NODES:
    if i != START_NODE:
        prob += x[0][i] == 0

prob += finished[0] == 0

# 起点购买与初始化
prob += money[0] == INIT_FUNDS - buy_w_start * WATER_PRICE - buy_f_start * FOOD_PRICE
prob += water[0] == buy_w_start
prob += food[0] == buy_f_start
prob += WATER_WEIGHT * water[0] + FOOD_WEIGHT * food[0] <= WEIGHT_LIMIT

# ==========================================
# 3. 每日状态转移
# ==========================================
for t in range(1, DAYS + 1):
    w_today = weather_seq[t - 1]

    # 每天只在一个节点
    prob += pulp.lpSum(x[t][i] for i in NODES) == 1

    # finished[t] 递推：到第t天结束时是否已经到达终点
    prob += finished[t] >= finished[t - 1]
    prob += finished[t] >= x[t][END_NODE]
    prob += finished[t] <= finished[t - 1] + x[t][END_NODE]

    # 如果前一天已经结束，则今天不再允许任何动作
    # 如果前一天未结束，则今天必须且只能选择一个动作
    prob += stay[t] + move[t] + mine[t] == 1 - finished[t - 1]

    # 如果前一天已经结束，则今天必须固定在终点
    prob += x[t][END_NODE] >= finished[t - 1]

    # 图论拓扑约束
    # 说明：
    # 1. 不结束时，正常沿邻接点移动或原地停留
    # 2. 已结束时，因为 move/stay/mine 全为0，且位置被锁在终点，所以不会继续乱走
    for i in NODES:
        neighbors = adj_list[i]
        prob += x[t][i] <= x[t - 1][i] + pulp.lpSum(x[t - 1][j] for j in neighbors)
        prob += x[t][i] - x[t - 1][i] <= move[t]
        prob += x[t - 1][i] - x[t][i] <= move[t]
        prob += x[t - 1][i] + x[t][i] <= 2 - move[t]

    # 沙暴天禁止移动
    if w_today == 3:
        prob += move[t] == 0

    # 矿山约束：只有前一天已经在矿山，今天才能挖矿
    is_at_mine_yesterday = pulp.lpSum(x[t - 1][m] for m in MINES)
    prob += mine[t] <= is_at_mine_yesterday

    # 村庄购买约束：只有当天在村庄，才能购买
    is_in_village = pulp.lpSum(x[t][v] for v in VILLAGES)
    BIG_M = 1000
    prob += buy_w_vill[t] + buy_f_vill[t] <= BIG_M * is_in_village

    # 已结束判定
    is_game_over = finished[t - 1]

    # 当天基础消耗
    base_w = WATER_CONSUME[w_today]
    base_f = FOOD_CONSUME[w_today]
    consume_multiplier = stay[t] * 1 + move[t] * 2 + mine[t] * 3

    calc_c_w = base_w * consume_multiplier
    calc_c_f = base_f * consume_multiplier

    # 若已结束，则实际消耗强制为0
    actual_c_w = pulp.LpVariable(f"ActualCw_{t}", lowBound=0)
    actual_c_f = pulp.LpVariable(f"ActualCf_{t}", lowBound=0)
    MAX_C = 50

    prob += actual_c_w <= calc_c_w
    prob += actual_c_w >= calc_c_w - MAX_C * is_game_over
    prob += actual_c_w <= MAX_C * (1 - is_game_over)

    prob += actual_c_f <= calc_c_f
    prob += actual_c_f >= calc_c_f - MAX_C * is_game_over
    prob += actual_c_f <= MAX_C * (1 - is_game_over)

    # 生存约束
    prob += water[t - 1] >= actual_c_w
    prob += food[t - 1] >= actual_c_f

    # 资源更新
    prob += water[t] == water[t - 1] - actual_c_w + buy_w_vill[t]
    prob += food[t] == food[t - 1] - actual_c_f + buy_f_vill[t]

    # 负重约束
    prob += WATER_WEIGHT * water[t] + FOOD_WEIGHT * food[t] <= WEIGHT_LIMIT

    # 到达终点后不能再挖矿赚钱
    prob += mine[t] <= (1 - is_game_over)

    # 资金更新
    prob += money[t] == (
        money[t - 1]
        + mine[t] * BASE_INCOME
        - (buy_w_vill[t] * WATER_PRICE * 2 + buy_f_vill[t] * FOOD_PRICE * 2)
    )

# 截止第30天必须已经到达终点
prob += x[DAYS][END_NODE] == 1

# ==========================================
# 4. 目标函数：最大化最终保留资金
# ==========================================
final_money = money[DAYS] + water[DAYS] * WATER_PRICE * 0.5 + food[DAYS] * FOOD_PRICE * 0.5
prob += final_money

# ==========================================
# 5. 求解
# ==========================================
solver = pulp.PULP_CBC_CMD(msg=1)
prob.solve(solver)

# ==========================================
# 6. 输出结果
# ==========================================
print("\n" + "=" * 60)
print("求解状态:", pulp.LpStatus[prob.status])

if prob.status == pulp.LpStatusOptimal:
    print(f"最大保留资金: {pulp.value(prob.objective):.2f} 元")
    print(f"起点购买 -> 水: {pulp.value(buy_w_start):.0f} 箱, 食物: {pulp.value(buy_f_start):.0f} 箱")
    print("-" * 60)

    first_finish_day = None
    for t in range(1, DAYS + 1):
        loc = [i for i in NODES if pulp.value(x[t][i]) > 0.5][0]

        if pulp.value(finished[t - 1]) > 0.5:
            act = "已完成"
        elif pulp.value(stay[t]) > 0.5:
            act = "停留"
        elif pulp.value(move[t]) > 0.5:
            act = "行走"
        elif pulp.value(mine[t]) > 0.5:
            act = "挖矿"
        else:
            act = "无动作"

        if first_finish_day is None and pulp.value(x[t][END_NODE]) > 0.5:
            first_finish_day = t

        buy_str = ""
        w_vill = pulp.value(buy_w_vill[t])
        f_vill = pulp.value(buy_f_vill[t])
        if w_vill > 0 or f_vill > 0:
            buy_str = f" | [村庄购买] 水: {w_vill:.0f}, 食物: {f_vill:.0f}"

        print(
            f"第{t:02d}天 | 位置: {loc:02d} | 动作: {act} | "
            f"水量: {pulp.value(water[t]):.0f} | 食物: {pulp.value(food[t]):.0f} | "
            f"资金: {pulp.value(money[t]):.0f}{buy_str}"
        )

    print("-" * 60)
    if first_finish_day is not None:
        print(f"首次到达终点的日期: 第 {first_finish_day} 天")
else:
    print("[警告] 求解失败，模型无解！")