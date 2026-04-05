import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from math import inf

# =========================================================
# 0. 基础参数
# =========================================================
NUM_PLAYERS = 3
TOTAL_DAYS = 30
MAX_WEIGHT = 1200

PRICE_W, PRICE_F = 5, 10
MINE_REWARD = 1000

# 天气: 0=晴朗, 1=高温, 2=沙暴
CONS_W = [3, 9, 10]
CONS_F = [4, 9, 10]
PROB_WEATHER = [0.5, 0.4, 0.1]

START_NODE, VILLAGE_NODE, MINE_NODE, END_NODE = 1, 14, 18, 25

# 蒙特卡洛次数
N_SIM_EVAL = 300         # 网格搜索时每组参数的模拟次数
N_SIM_FINAL = 1000       # 最终方案统计次数

# 存活率下限
SURVIVAL_THRESHOLD = 0.50

# 是否打印搜索过程
VERBOSE_SEARCH = True

# 图片保存目录
SAVE_DIR = "mc_figures_safe"

# 字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 1. 地图函数（5x5 网格）
# =========================================================
def node_rc(node):
    r = (node - 1) // 5
    c = (node - 1) % 5
    return r, c

def manhattan(n1, n2):
    r1, c1 = node_rc(n1)
    r2, c2 = node_rc(n2)
    return abs(r1 - r2) + abs(c1 - c2)

def is_adj(n1, n2):
    if n1 == n2:
        return True
    return manhattan(n1, n2) == 1

def neighbors(node):
    return [j for j in range(1, 26) if is_adj(node, j) and j != node]


# =========================================================
# 2. 资源与安全评估
# =========================================================
def weight_of(water, food):
    return 3 * water + 2 * food

def liquidation_value(funds, water, food):
    return funds + 0.5 * PRICE_W * max(water, 0) + 0.5 * PRICE_F * max(food, 0)

def worst_daily_need(action_type):
    """
    保守估计：用沙暴/高温较大消耗做安全储备
    action_type:
        'stay' -> 1倍
        'move' -> 2倍
        'mine' -> 3倍
    """
    mult = {"stay": 1, "move": 2, "mine": 3}[action_type]
    # 用最坏情况沙暴的日消耗做保守安全估计
    return mult * CONS_W[2], mult * CONS_F[2]

def expected_move_need(days):
    """
    用天气期望估计若干天移动的资源需求（每次移动按2倍基础消耗）
    """
    ew = PROB_WEATHER[0] * CONS_W[0] + PROB_WEATHER[1] * CONS_W[1] + PROB_WEATHER[2] * CONS_W[2]
    ef = PROB_WEATHER[0] * CONS_F[0] + PROB_WEATHER[1] * CONS_F[1] + PROB_WEATHER[2] * CONS_F[2]
    return days * 2 * ew, days * 2 * ef

def safety_need_to_target(cur_node, target_node, buffer_days=2):
    """
    到目标点的保命需求估计：
    距离 * 期望移动消耗 + 若干天缓冲
    """
    d = manhattan(cur_node, target_node)
    mw, mf = expected_move_need(d)
    bw, bf = expected_move_need(buffer_days)
    return mw + bw, mf + bf


# =========================================================
# 3. 村庄补给策略
# =========================================================
def buy_at_village(state, target_node):
    """
    到达村庄后自动补给：
    目标是保证能从村庄安全走到 target_node，并留出缓冲
    """
    if state["node"] != VILLAGE_NODE or (not state["alive"]) or state["arrived"]:
        return

    need_w, need_f = safety_need_to_target(VILLAGE_NODE, target_node, buffer_days=3)

    # 向上取整，至少留一点整数余量
    target_w = int(np.ceil(need_w))
    target_f = int(np.ceil(need_f))

    # 当前已有资源
    cur_w = state["water"]
    cur_f = state["food"]

    add_w = max(0, target_w - cur_w)
    add_f = max(0, target_f - cur_f)

    # 先按资金限制
    max_afford_w = state["funds"] // (2 * PRICE_W)
    add_w = min(add_w, max_afford_w)

    state["funds"] -= add_w * 2 * PRICE_W
    cur_w += add_w

    max_afford_f = state["funds"] // (2 * PRICE_F)
    add_f = min(add_f, max_afford_f)

    # 再按负重限制
    while weight_of(cur_w, cur_f + add_f) > MAX_WEIGHT and add_f > 0:
        add_f -= 1

    state["funds"] -= add_f * 2 * PRICE_F
    cur_f += add_f

    # 最后再检查水的负重
    while weight_of(cur_w, cur_f) > MAX_WEIGHT and cur_w > state["water"]:
        cur_w -= 1
        state["funds"] += 2 * PRICE_W

    state["water"] = cur_w
    state["food"] = cur_f


# =========================================================
# 4. 玩家策略：保命优先 + 收益次优
# =========================================================
def next_step_towards(cur, target, forbidden_targets=None):
    """
    返回朝 target 前进的一步，若有多个最优邻居，尽量避开 forbidden_targets
    """
    if forbidden_targets is None:
        forbidden_targets = set()

    cands = neighbors(cur)
    cands.sort(key=lambda x: manhattan(x, target))

    # 先选不冲突的
    for nb in cands:
        if nb not in forbidden_targets:
            return nb

    # 实在避不开就选最近的
    return cands[0] if cands else cur


def choose_goal(state, day):
    """
    给定玩家当前状态，先决定战略目标：
    终点 / 村庄 / 矿山
    """
    node = state["node"]
    water = state["water"]
    food = state["food"]

    # 已到终点或已死
    if (not state["alive"]) or state["arrived"]:
        return END_NODE

    # 终点安全需求
    need_end_w, need_end_f = safety_need_to_target(node, END_NODE, buffer_days=1)

    # 村庄安全需求
    need_v_w, need_v_f = safety_need_to_target(node, VILLAGE_NODE, buffer_days=1)

    # 若资源不足以安全到终点，优先去村庄
    if water < need_end_w or food < need_end_f:
        return VILLAGE_NODE

    # 若已接近截止日期，直接冲终点
    if day >= TOTAL_DAYS - 6:
        return END_NODE

    # 若离矿山近，且去矿山再回终点仍有安全余量，则考虑矿山
    need_m_w, need_m_f = safety_need_to_target(node, MINE_NODE, buffer_days=2)
    need_back_w, need_back_f = safety_need_to_target(MINE_NODE, END_NODE, buffer_days=2)

    if water >= (need_m_w + need_back_w) and food >= (need_m_f + need_back_f):
        return MINE_NODE

    return END_NODE


def choose_action_for_player(state, day, weather, used_move_targets):
    """
    单个玩家决策：
    1) 先选战略目标
    2) 再决定今天 stay / move / mine
    """
    if (not state["alive"]) or state["arrived"]:
        return {"act": "stay", "target": state["node"]}, END_NODE

    node = state["node"]
    goal = choose_goal(state, day)

    # 在村庄先补给
    if node == VILLAGE_NODE:
        if goal == VILLAGE_NODE:
            # 若本来就是来补给，则补给后改冲终点
            buy_at_village(state, END_NODE)
            goal = END_NODE
        else:
            buy_at_village(state, goal)

    # 若在矿山，判断是否挖矿
    if node == MINE_NODE and day <= TOTAL_DAYS - 5:
        # 挖矿前检查安全余量
        mine_need_w, mine_need_f = worst_daily_need("mine")
        back_need_w, back_need_f = safety_need_to_target(MINE_NODE, END_NODE, buffer_days=2)

        if state["water"] >= mine_need_w + back_need_w and state["food"] >= mine_need_f + back_need_f:
            return {"act": "mine", "target": node}, goal

    # 沙暴天不能走
    if weather == 2:
        return {"act": "stay", "target": node}, goal

    # 否则向目标走一步，同时尽量避碰撞
    if node != goal:
        next_node = next_step_towards(node, goal, forbidden_targets=used_move_targets)
        used_move_targets.add(next_node)
        return {"act": "move", "target": next_node}, goal

    return {"act": "stay", "target": node}, goal


# =========================================================
# 5. 单日结算
# =========================================================
def settle_day(states, actions, weather):
    """
    单日真实结算：
    - 同目标移动的人越多，消耗越大
    - 同时挖矿收益平分
    """
    new_states = deepcopy(states)
    cw, cf = CONS_W[weather], CONS_F[weather]

    # 统计 move 目标
    move_target_count = {}
    for act in actions:
        if act["act"] == "move":
            move_target_count[act["target"]] = move_target_count.get(act["target"], 0) + 1

    mine_count = sum(1 for act in actions if act["act"] == "mine")

    for i in range(NUM_PLAYERS):
        st = new_states[i]
        act = actions[i]

        if (not st["alive"]) or st["arrived"]:
            continue

        mult = 1
        if act["act"] == "move":
            k = move_target_count.get(act["target"], 1)
            mult = 2 * k
        elif act["act"] == "mine":
            mult = 3
        else:
            mult = 1

        st["water"] -= mult * cw
        st["food"] -= mult * cf
        st["node"] = act["target"]

        if act["act"] == "mine" and mine_count > 0:
            st["funds"] += MINE_REWARD / mine_count

        if st["node"] == END_NODE:
            st["arrived"] = True

        if st["water"] < 0 or st["food"] < 0:
            st["alive"] = False

    return new_states


# =========================================================
# 6. 单次仿真
# =========================================================
def run_one_sim(init_water, init_food):
    init_funds = 10000 - init_water * PRICE_W - init_food * PRICE_F
    if init_funds < 0:
        return None

    if weight_of(init_water, init_food) > MAX_WEIGHT:
        return None

    states = [
        {"id": 0, "node": START_NODE, "water": init_water, "food": init_food, "funds": init_funds, "alive": True, "arrived": False},
        {"id": 1, "node": START_NODE, "water": init_water, "food": init_food, "funds": init_funds, "alive": True, "arrived": False},
        {"id": 2, "node": START_NODE, "water": init_water, "food": init_food, "funds": init_funds, "alive": True, "arrived": False},
    ]

    weather_seq = []
    trajectory = [[] for _ in range(NUM_PLAYERS)]

    for day in range(1, TOTAL_DAYS + 1):
        weather = np.random.choice([0, 1, 2], p=PROB_WEATHER)
        weather_seq.append(weather)

        used_move_targets = set()
        actions = []
        goals = []

        # 按玩家编号顺序做错峰决策，减少碰撞
        for i in range(NUM_PLAYERS):
            act, goal = choose_action_for_player(states[i], day, weather, used_move_targets)
            actions.append(act)
            goals.append(goal)

        states = settle_day(states, actions, weather)

        # 到村庄后再补给一次（晚间补给）
        for i in range(NUM_PLAYERS):
            if states[i]["alive"] and (not states[i]["arrived"]) and states[i]["node"] == VILLAGE_NODE:
                target_goal = choose_goal(states[i], day)
                if target_goal == VILLAGE_NODE:
                    target_goal = END_NODE
                buy_at_village(states[i], target_goal)

        for i in range(NUM_PLAYERS):
            trajectory[i].append(states[i]["node"])

        if all((not s["alive"]) or s["arrived"] for s in states):
            break

    final_funds = []
    for s in states:
        if s["alive"]:
            value = liquidation_value(s["funds"], s["water"], s["food"])
        else:
            value = 0.0
        final_funds.append(value)

    alive_num = sum(1 for s in states if s["alive"])
    arrived_num = sum(1 for s in states if s["alive"] and s["arrived"])
    total_final_funds_alive_only = sum(final_funds[i] for i in range(NUM_PLAYERS) if states[i]["alive"])

    return {
        "states": states,
        "weather_seq": weather_seq,
        "trajectory": trajectory,
        "final_funds": final_funds,
        "alive_num": alive_num,
        "arrived_num": arrived_num,
        "total_final_funds_alive_only": total_final_funds_alive_only,
        "all_alive": int(alive_num == NUM_PLAYERS),
        "all_arrived": int(arrived_num == NUM_PLAYERS),
        "at_least_one_arrived": int(arrived_num >= 1),
    }


# =========================================================
# 7. 蒙特卡洛评估
# =========================================================
def monte_carlo_simulation(init_water, init_food, n_sim=300):
    total_funds_list = []
    alive_num_list = []
    arrived_num_list = []
    all_alive_list = []
    all_arrived_list = []
    one_arrived_list = []
    player_final_funds = [[] for _ in range(NUM_PLAYERS)]
    weather_count_all = np.zeros(3)

    valid_runs = 0

    for _ in range(n_sim):
        result = run_one_sim(init_water, init_food)
        if result is None:
            continue

        valid_runs += 1
        total_funds_list.append(result["total_final_funds_alive_only"])
        alive_num_list.append(result["alive_num"])
        arrived_num_list.append(result["arrived_num"])
        all_alive_list.append(result["all_alive"])
        all_arrived_list.append(result["all_arrived"])
        one_arrived_list.append(result["at_least_one_arrived"])

        for i in range(NUM_PLAYERS):
            player_final_funds[i].append(result["final_funds"][i])

        for w in result["weather_seq"]:
            weather_count_all[w] += 1

    if valid_runs == 0:
        return None

    return {
        "init_water": init_water,
        "init_food": init_food,
        "avg_total_final_funds": np.mean(total_funds_list),
        "std_total_final_funds": np.std(total_funds_list),
        "avg_alive_num": np.mean(alive_num_list),
        "avg_arrived_num": np.mean(arrived_num_list),
        "prob_all_alive": np.mean(all_alive_list),
        "prob_all_arrived": np.mean(all_arrived_list),
        "prob_at_least_one_arrived": np.mean(one_arrived_list),
        "avg_player_final_funds": [np.mean(player_final_funds[i]) for i in range(NUM_PLAYERS)],
        "std_player_final_funds": [np.std(player_final_funds[i]) for i in range(NUM_PLAYERS)],
        "weather_freq": weather_count_all / weather_count_all.sum() if weather_count_all.sum() > 0 else np.zeros(3),

        "total_funds_list": total_funds_list,
        "alive_num_list": alive_num_list,
        "arrived_num_list": arrived_num_list,
        "player_final_funds": player_final_funds,
    }


# =========================================================
# 8. 搜索满足存活率>50%的较优初始资源
# =========================================================
def search_best_initial_plan():
    best = None

    # 你可以自行改搜索范围
    water_grid = range(120, 241, 20)
    food_grid = range(100, 221, 20)

    candidates = []

    for iw in water_grid:
        for ifd in food_grid:
            if weight_of(iw, ifd) > MAX_WEIGHT:
                continue
            if 10000 - iw * PRICE_W - ifd * PRICE_F < 0:
                continue

            summary = monte_carlo_simulation(iw, ifd, n_sim=N_SIM_EVAL)
            if summary is None:
                continue

            survival_rate = summary["avg_alive_num"] / NUM_PLAYERS

            if VERBOSE_SEARCH:
                print(
                    f"测试 (水={iw}, 食物={ifd}) -> "
                    f"平均存活率={survival_rate:.2%}, "
                    f"平均总资金={summary['avg_total_final_funds']:.2f}"
                )

            if survival_rate >= SURVIVAL_THRESHOLD:
                candidates.append(summary)

    if len(candidates) == 0:
        print("\n未找到满足“平均存活率≥50%”的方案，改为选择存活率最高方案。")
        for iw in water_grid:
            for ifd in food_grid:
                if weight_of(iw, ifd) > MAX_WEIGHT:
                    continue
                if 10000 - iw * PRICE_W - ifd * PRICE_F < 0:
                    continue
                summary = monte_carlo_simulation(iw, ifd, n_sim=N_SIM_EVAL)
                if summary is None:
                    continue
                if (best is None or
                    summary["avg_alive_num"] > best["avg_alive_num"] or
                    (summary["avg_alive_num"] == best["avg_alive_num"] and
                     summary["avg_total_final_funds"] > best["avg_total_final_funds"])):
                    best = summary
        return best

    # 在满足存活率约束的候选中，选平均总资金最高的
    best = max(candidates, key=lambda x: x["avg_total_final_funds"])
    return best


# =========================================================
# 9. 画图
# =========================================================
def save_monte_carlo_figures(summary, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # 图1：最终总资金分布
    plt.figure(figsize=(8, 5))
    plt.hist(summary["total_funds_list"], bins=25, edgecolor="black")
    plt.xlabel("最终总资金")
    plt.ylabel("频数")
    plt.title("问题三蒙特卡洛模拟：最终总资金分布")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_total_funds_hist.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 图2：存活人数分布
    alive_counts = np.bincount(summary["alive_num_list"], minlength=NUM_PLAYERS + 1)
    plt.figure(figsize=(8, 5))
    plt.bar(range(NUM_PLAYERS + 1), alive_counts, edgecolor="black")
    plt.xlabel("存活人数")
    plt.ylabel("出现次数")
    plt.title("问题三蒙特卡洛模拟：存活人数分布")
    plt.xticks(range(NUM_PLAYERS + 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_alive_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 图3：到达终点人数分布
    arrived_counts = np.bincount(summary["arrived_num_list"], minlength=NUM_PLAYERS + 1)
    plt.figure(figsize=(8, 5))
    plt.bar(range(NUM_PLAYERS + 1), arrived_counts, edgecolor="black")
    plt.xlabel("到达终点人数")
    plt.ylabel("出现次数")
    plt.title("问题三蒙特卡洛模拟：到达终点人数分布")
    plt.xticks(range(NUM_PLAYERS + 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_arrived_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 图4：关键概率指标
    prob_labels = ["全员存活概率", "全员到达概率", "至少1人到达概率"]
    prob_values = [
        summary["prob_all_alive"],
        summary["prob_all_arrived"],
        summary["prob_at_least_one_arrived"],
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(prob_labels, prob_values, edgecolor="black")
    plt.ylabel("概率")
    plt.ylim(0, 1)
    plt.title("问题三蒙特卡洛模拟：关键概率指标")
    plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(prob_values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig4_probability_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 图5：各玩家终端资金均值及波动
    x = np.arange(NUM_PLAYERS)
    means = summary["avg_player_final_funds"]
    stds = summary["std_player_final_funds"]
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=6, edgecolor="black")
    plt.xlabel("玩家编号")
    plt.ylabel("终端资金")
    plt.title("问题三蒙特卡洛模拟：各玩家终端资金均值及波动")
    plt.xticks(x, [f"玩家{i}" for i in range(NUM_PLAYERS)])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig5_player_funds_errorbar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 图6：天气频率统计
    weather_labels = ["晴朗", "高温", "沙暴"]
    weather_values = summary["weather_freq"]
    plt.figure(figsize=(8, 5))
    plt.bar(weather_labels, weather_values, edgecolor="black")
    plt.ylabel("频率")
    plt.ylim(0, 1)
    plt.title("问题三蒙特卡洛模拟：天气频率统计")
    plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(weather_values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig6_weather_frequency.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n所有图片已保存到文件夹: {save_dir}")


# =========================================================
# 10. 主程序
# =========================================================
if __name__ == "__main__":
    print("开始搜索满足“平均存活率≥50%且资金较高”的初始资源方案...\n")

    best_plan = search_best_initial_plan()

    print("\n" + "=" * 65)
    print("搜索得到的较优初始方案")
    print("=" * 65)
    print(f"初始水量: {best_plan['init_water']}")
    print(f"初始食物: {best_plan['init_food']}")
    print(f"平均总资金: {best_plan['avg_total_final_funds']:.2f}")
    print(f"平均存活人数: {best_plan['avg_alive_num']:.4f}")
    print(f"平均到达终点人数: {best_plan['avg_arrived_num']:.4f}")
    print(f"平均存活率: {best_plan['avg_alive_num']/NUM_PLAYERS:.2%}")

    print("\n使用最优初始资源进行高精度蒙特卡洛模拟...\n")
    final_summary = monte_carlo_simulation(
        best_plan["init_water"],
        best_plan["init_food"],
        n_sim=N_SIM_FINAL
    )

    print("\n" + "=" * 65)
    print("问题三 / 第六关 最终蒙特卡洛统计结果")
    print("=" * 65)
    print(f"最优初始水量: {final_summary['init_water']}")
    print(f"最优初始食物: {final_summary['init_food']}")
    print(f"平均最终总资金: {final_summary['avg_total_final_funds']:.2f}")
    print(f"最终总资金标准差: {final_summary['std_total_final_funds']:.2f}")
    print(f"平均存活人数: {final_summary['avg_alive_num']:.4f}")
    print(f"平均到达终点人数: {final_summary['avg_arrived_num']:.4f}")
    print(f"全员存活概率: {final_summary['prob_all_alive']:.4%}")
    print(f"全员到达终点概率: {final_summary['prob_all_arrived']:.4%}")
    print(f"至少1人到达终点概率: {final_summary['prob_at_least_one_arrived']:.4%}")

    for i in range(NUM_PLAYERS):
        print(
            f"玩家{i} 平均终端资金: {final_summary['avg_player_final_funds'][i]:.2f} "
            f"(标准差 {final_summary['std_player_final_funds'][i]:.2f})"
        )

    print(
        "天气频率估计: "
        f"晴朗={final_summary['weather_freq'][0]:.4%}, "
        f"高温={final_summary['weather_freq'][1]:.4%}, "
        f"沙暴={final_summary['weather_freq'][2]:.4%}"
    )

    save_monte_carlo_figures(final_summary, save_dir=SAVE_DIR)