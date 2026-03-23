import pulp

# ==========================================
# 1. 基础参数与数据输入 (第二关)
# ==========================================
DAYS = 30
START_NODE = 1
END_NODE = 64     # 终点为 64
MINES = [30, 55]  # 第二关矿山节点
VILLAGES = [39, 62] # 第二关村庄节点
NODES = list(range(1, 65))

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

# ---------------------------------------------------------
# 自动生成六边形网格连线 (Adj List)
# 根据原图：8x8 矩阵，奇数行左对齐，偶数行右移半格
# ---------------------------------------------------------
adj_list = {i: [] for i in NODES}
for i in NODES:
    r = (i - 1) // 8 + 1  # 行号 1-8
    c = (i - 1) % 8 + 1   # 列号 1-8
    
    # 左右相邻
    if c > 1: adj_list[i].append(i - 1)
    if c < 8: adj_list[i].append(i + 1)
    
    # 上下斜向相邻
    if r % 2 != 0: # 奇数行
        if r > 1:
            adj_list[i].append(i - 8) # 右上
            if c > 1: adj_list[i].append(i - 9) # 左上
        if r < 8:
            adj_list[i].append(i + 8) # 右下
            if c > 1: adj_list[i].append(i + 7) # 左下
    else: # 偶数行
        if r > 1:
            adj_list[i].append(i - 8) # 左上
            if c < 8: adj_list[i].append(i - 7) # 右上
        if r < 8:
            adj_list[i].append(i + 8) # 左下
            if c < 8: adj_list[i].append(i + 9) # 右下

# 排序去重确保干净
for i in adj_list:
    adj_list[i] = sorted(list(set(adj_list[i])))

# ==========================================
# 2. 建立线性规划模型
# ==========================================
prob = pulp.LpProblem("Desert_Crossing_Level2", pulp.LpMaximize)

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
prob += x[0][START_NODE] == 1
for i in NODES:
    if i != START_NODE:
        prob += x[0][i] == 0

prob += money[0] == INIT_FUNDS - buy_w_start * WATER_PRICE - buy_f_start * FOOD_PRICE
prob += water[0] == buy_w_start
prob += food[0] == buy_f_start
prob += WATER_WEIGHT * water[0] + FOOD_WEIGHT * food[0] <= WEIGHT_LIMIT

for t in range(1, DAYS + 1):
    w_today = weather_seq[t-1]
    
    prob += stay[t] + move[t] + mine[t] == 1
    prob += pulp.lpSum(x[t][i] for i in NODES) == 1
    
    for i in NODES:
        neighbors = adj_list[i]
        prob += x[t][i] <= x[t-1][i] + pulp.lpSum(x[t-1][j] for j in neighbors)
        prob += x[t][i] - x[t-1][i] <= move[t]
        prob += x[t-1][i] - x[t][i] <= move[t]
        prob += x[t-1][i] + x[t][i] <= 2 - move[t]
    
    if w_today == 3:
        prob += move[t] == 0
        
    is_at_mine_yesterday = pulp.lpSum(x[t-1][m] for m in MINES)
    prob += mine[t] <= is_at_mine_yesterday
    
    is_in_village = pulp.lpSum(x[t][v] for v in VILLAGES)
    BIG_M = 1000
    prob += buy_w_vill[t] + buy_f_vill[t] <= BIG_M * is_in_village
    
    is_game_over = x[t-1][END_NODE]
    
    base_w = WATER_CONSUME[w_today]
    base_f = FOOD_CONSUME[w_today]
    consume_multiplier = stay[t]*1 + move[t]*2 + mine[t]*3
    
    calc_c_w = base_w * consume_multiplier
    calc_c_f = base_f * consume_multiplier
    
    actual_c_w = pulp.LpVariable(f"ActualCw_{t}", lowBound=0)
    actual_c_f = pulp.LpVariable(f"ActualCf_{t}", lowBound=0)
    MAX_C = 50
    
    prob += actual_c_w <= calc_c_w
    prob += actual_c_w >= calc_c_w - MAX_C * is_game_over
    prob += actual_c_w <= MAX_C * (1 - is_game_over)
    
    prob += actual_c_f <= calc_c_f
    prob += actual_c_f >= calc_c_f - MAX_C * is_game_over
    prob += actual_c_f <= MAX_C * (1 - is_game_over)

    # 强制生存判定：出发前物资必须足够
    prob += water[t-1] >= actual_c_w
    prob += food[t-1] >= actual_c_f

    prob += water[t] == water[t-1] - actual_c_w + buy_w_vill[t]
    prob += food[t] == food[t-1] - actual_c_f + buy_f_vill[t]
    prob += WATER_WEIGHT * water[t] + FOOD_WEIGHT * food[t] <= WEIGHT_LIMIT
    
    prob += mine[t] <= (1 - is_game_over) 
    prob += money[t] == money[t-1] + mine[t]*BASE_INCOME - (buy_w_vill[t]*WATER_PRICE*2 + buy_f_vill[t]*FOOD_PRICE*2)

prob += x[DAYS][END_NODE] == 1

# ==========================================
# 3. 目标函数
# ==========================================
final_money = money[DAYS] + water[DAYS] * WATER_PRICE * 0.5 + food[DAYS] * FOOD_PRICE * 0.5
prob += final_money

# ==========================================
# 4. 求解与输出
# ==========================================
# ⚠️ 注意：第二关节点数达64个，搜索空间呈指数级放大，求解器可能需要 1~5 分钟。请耐心等待！
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