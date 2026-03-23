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

# 真实的第一关地图节点连线 (已双向验证连通性)
adj_list = {
    1: [2, 25],
    2: [1, 3, 25],
    3: [2, 4],
    4: [3, 5, 24, 25],
    5: [4, 6, 24],
    6: [5, 7, 22, 23, 24],
    7: [6, 8, 22],
    8: [7, 9, 10, 22],
    9: [8, 10, 15, 16, 17, 22],
    10: [8, 9, 11, 15],
    11: [10, 12, 13],
    12: [11, 13, 14],
    13: [11, 12, 14, 15],
    14: [12, 13, 15, 16],
    15: [9, 10, 13, 14, 16],
    16: [9, 14, 15, 17, 18],
    17: [9, 16, 18, 21, 22],
    18: [16, 17, 19, 20, 21],
    19: [18, 20],
    20: [18, 19, 21],
    21: [17, 18, 20, 22, 23, 26, 27],
    22: [6, 7, 8, 9, 17, 21, 23],
    23: [6, 21, 22, 24, 26],
    24: [4, 5, 6, 23, 25, 26],
    25: [1, 2, 4, 24, 26],
    26: [21, 23, 24, 25, 27],
    27: [21, 26]
}

# ==========================================
# 2. 建立线性规划模型
# ==========================================
prob = pulp.LpProblem("Desert_Crossing_Level1", pulp.LpMaximize)

# --- 决策变量 ---
x = pulp.LpVariable.dicts("Position", (range(DAYS + 1), NODES), cat='Binary')

stay = pulp.LpVariable.dicts("Stay", range(1, DAYS + 1), cat='Binary')
move = pulp.LpVariable.dicts("Move", range(1, DAYS + 1), cat='Binary')
mine = pulp.LpVariable.dicts("Mine", range(1, DAYS + 1), cat='Binary')

water = pulp.LpVariable.dicts("Water", range(DAYS + 1), lowBound=0)
food = pulp.LpVariable.dicts("Food", range(DAYS + 1), lowBound=0)
money = pulp.LpVariable.dicts("Money", range(DAYS + 1), lowBound=0)

buy_w_start = pulp.LpVariable("BuyWaterStart", lowBound=0, cat='Integer')
buy_f_start = pulp.LpVariable("BuyFoodStart", lowBound=0, cat='Integer')
buy_w_vill = pulp.LpVariable.dicts("BuyWaterVill", range(1, DAYS + 1), lowBound=0, cat='Integer')
buy_f_vill = pulp.LpVariable.dicts("BuyFoodVill", range(1, DAYS + 1), lowBound=0, cat='Integer')

# --- 约束条件 ---
# 1. 初始状态
prob += x[0][START_NODE] == 1
for i in NODES:
    if i != START_NODE:
        prob += x[0][i] == 0

# 起点购买与资源初始化
prob += money[0] == INIT_FUNDS - buy_w_start * WATER_PRICE - buy_f_start * FOOD_PRICE
prob += water[0] == buy_w_start
prob += food[0] == buy_f_start
prob += WATER_WEIGHT * water[0] + FOOD_WEIGHT * food[0] <= WEIGHT_LIMIT

# 2. 每日状态转移
for t in range(1, DAYS + 1):
    w_today = weather_seq[t-1]
    
    # 动作互斥：每天只能停留、行走或挖矿
    prob += stay[t] + move[t] + mine[t] == 1
    # 每天只在一个节点
    prob += pulp.lpSum(x[t][i] for i in NODES) == 1
    
    # 图论拓扑约束：行走必须沿已知路线
    for i in NODES:
        neighbors = adj_list[i]
        prob += x[t][i] <= x[t-1][i] + pulp.lpSum(x[t-1][j] for j in neighbors)
        prob += x[t][i] - x[t-1][i] <= move[t]
        prob += x[t-1][i] - x[t][i] <= move[t]
        prob += x[t-1][i] + x[t][i] <= 2 - move[t]
    
    # 天气特殊规则：沙暴禁止移动
    if w_today == 3:
        prob += move[t] == 0
        
    # 矿山特殊规则：到达当天不能挖矿
    is_at_mine_yesterday = pulp.lpSum(x[t-1][m] for m in MINES)
    prob += mine[t] <= is_at_mine_yesterday
    
    # 村庄购买规则 (大M法限制)
    is_in_village = pulp.lpSum(x[t][v] for v in VILLAGES)
    BIG_M = 1000
    prob += buy_w_vill[t] + buy_f_vill[t] <= BIG_M * is_in_village
    
    # 消耗结算计算 (若提前到达终点则游戏结束，停止消耗)
    is_game_over = x[t-1][END_NODE]
    
    base_w = WATER_CONSUME[w_today]
    base_f = FOOD_CONSUME[w_today]
    consume_multiplier = stay[t]*1 + move[t]*2 + mine[t]*3
    
    calc_c_w = base_w * consume_multiplier
    calc_c_f = base_f * consume_multiplier
    
    # 利用大M法将已到达终点后的实际消耗强制降为0
    actual_c_w = pulp.LpVariable(f"ActualCw_{t}", lowBound=0)
    actual_c_f = pulp.LpVariable(f"ActualCf_{t}", lowBound=0)
    MAX_C = 50
    
    prob += actual_c_w <= calc_c_w
    prob += actual_c_w >= calc_c_w - MAX_C * is_game_over
    prob += actual_c_w <= MAX_C * (1 - is_game_over)
    
    prob += actual_c_f <= calc_c_f
    prob += actual_c_f >= calc_c_f - MAX_C * is_game_over
    prob += actual_c_f <= MAX_C * (1 - is_game_over)

    # 强制生存判定：昨日剩余的物资必须 >= 今天的消耗！
    prob += water[t-1] >= actual_c_w
    prob += food[t-1] >= actual_c_f

    # 资源与资金更新
    prob += water[t] == water[t-1] - actual_c_w + buy_w_vill[t]
    prob += food[t] == food[t-1] - actual_c_f + buy_f_vill[t]
    
    # 每日负重约束不能超标
    prob += WATER_WEIGHT * water[t] + FOOD_WEIGHT * food[t] <= WEIGHT_LIMIT
    
    # 资金更新逻辑 (到达终点后不能再挖矿刷钱)
    prob += mine[t] <= (1 - is_game_over) 
    prob += money[t] == money[t-1] + mine[t]*BASE_INCOME - (buy_w_vill[t]*WATER_PRICE*2 + buy_f_vill[t]*FOOD_PRICE*2)

# 3. 终点约束：截止日期必须到达终点
prob += x[DAYS][END_NODE] == 1

# ==========================================
# 3. 目标函数：最大化最终保留资金
# ==========================================
# 终点剩余物资半价折现
final_money = money[DAYS] + water[DAYS] * WATER_PRICE * 0.5 + food[DAYS] * FOOD_PRICE * 0.5
prob += final_money

# ==========================================
# 4. 求解与格式化输出
# ==========================================
prob.solve(pulp.PULP_CBC_CMD(msg=1))

print("\n" + "="*50)
print("求解状态:", pulp.LpStatus[prob.status])

if prob.status == pulp.LpStatusOptimal:
    print(f"最大保留资金: {pulp.value(prob.objective)} 元\n")
    print(f"起点购买 -> 水: {pulp.value(buy_w_start)} 箱, 食物: {pulp.value(buy_f_start)} 箱")
    print("-" * 50)
    
    for t in range(1, DAYS + 1):
        loc = [i for i in NODES if pulp.value(x[t][i]) > 0.5][0]
        
        if pulp.value(stay[t]) > 0.5: act = "停留"
        elif pulp.value(move[t]) > 0.5: act = "行走"
        else: act = "挖矿"
            
        # 仅在有购买行为时打印购买量
        buy_str = ""
        w_vill = pulp.value(buy_w_vill[t])
        f_vill = pulp.value(buy_f_vill[t])
        if w_vill > 0 or f_vill > 0:
            buy_str = f" | [村庄购买] 水: {w_vill:.0f}, 食物: {f_vill:.0f}"
            
        print(f"第{t:02d}天 | 位置: {loc:02d} | 动作: {act} | "
              f"水量: {pulp.value(water[t]):.0f} | 食物: {pulp.value(food[t]):.0f} | "
              f"资金: {pulp.value(money[t]):.0f}{buy_str}")
else:
    print("\n[警告] 求解失败，模型无解！")