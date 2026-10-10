import numpy as np
import pandas as pd
import re
from itertools import combinations
from scipy import optimize

# =====================================================================
# 1. 基础数据结构
# =====================================================================

class DriftWire:
    def __init__(self, wire_id, x, y, r, true_xy=(0, 0), original_item=None):
        self.id = wire_id
        self.x = x
        self.y = y
        self.r = r
        self.true_xy = true_xy
        self.original_item = original_item  # 方便回溯原始属性

class TangentCell:
    def __init__(self, w1, w2, p1, p2, label):
        self.w1 = w1
        self.w2 = w2
        self.p1 = p1
        self.p2 = p2
        self.label = label
        self.state = 1
        self.neighbors =[]

    @property
    def direction(self):
        vec = self.p2 - self.p1
        dist = np.linalg.norm(vec)
        return vec / dist if dist > 0 else np.zeros(2)


# =====================================================================
# 2. CA 几何支持函数 (100% 忠于 test9)
# =====================================================================

def get_tangents_all(w1, w2, epsilon=1e-8):
    c1, r1 = np.array([w1.x, w1.y]), w1.r
    c2, r2 = np.array([w2.x, w2.y]), w2.r
    vec = c2 - c1
    dist = np.linalg.norm(vec)
    if dist < abs(r1 - r2) + epsilon: return []
    u = vec / dist
    v = np.array([-u[1], u[0]])
    tangents =[]

    cos_a = np.clip((r1 - r2) / dist, -1, 1)
    sin_a = np.sqrt(1 - cos_a ** 2)
    for s, lab in zip([1, -1], ['LL', 'RR']):
        n = cos_a * u + s * sin_a * v
        tangents.append((c1 + r1 * n, c2 + r2 * n, lab))

    if dist > (r1 + r2) + epsilon:
        cos_p = np.clip((r1 + r2) / dist, -1, 1)
        sin_p = np.sqrt(1 - cos_p ** 2)
        for s, lab in zip([1, -1], ['LR', 'RL']):
            n1 = cos_p * u + s * sin_p * v
            tangents.append((c1 + r1 * n1, c2 + r2 * (-n1), lab))
    return tangents

def is_obstructed(p1, p2, w1_id, w2_id, all_wires):
    v = p2 - p1
    v_mag_sq = np.dot(v, v)
    if v_mag_sq < 1e-6: return False
    for w in all_wires:
        if w.id == w1_id or w.id == w2_id: continue
        w_c = np.array([w.x, w.y])
        t = np.dot(w_c - p1, v) / v_mag_sq
        if 0 < t < 1:
            dist = np.linalg.norm(p1 + t * v - w_c)
            if dist < (w.r - 0.2): return True
    return False


# =====================================================================
# 3. 拟合与迭代纠偏模块 (100% 忠于 test9)
# =====================================================================

def calc_R(xc, yc, x, y):
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

def cost_function(params, x, y):
    xc, yc, R = params
    return calc_R(xc, yc, x, y) - R

def fit_track_circle(track):
    pts = [track[0].p1, track[0].p2]
    for cell in track[1:]: pts.append(cell.p2)
    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    x_m, y_m = np.mean(x), np.mean(y)
    R_m = np.mean(calc_R(x_m, y_m, x, y))
    res = optimize.least_squares(cost_function, (x_m, y_m, R_m), args=(x, y))
    return res.x[0], res.x[1], res.x[2], x, y

def get_best_tangent_point(wire, xc, yc, R):
    vx, vy = wire.x - xc, wire.y - yc
    dist = np.hypot(vx, vy)
    if dist == 0: return np.array([wire.x, wire.y])
    ux, uy = vx / dist, vy / dist
    p_in = np.array([wire.x - wire.r * ux, wire.y - wire.r * uy])
    p_out = np.array([wire.x + wire.r * ux, wire.y + wire.r * uy])
    d_in = abs(np.hypot(p_in[0] - xc, p_in[1] - yc) - R)
    d_out = abs(np.hypot(p_out[0] - xc, p_out[1] - yc) - R)
    return p_in if d_in < d_out else p_out

def iterative_refine_fit(track_cells, max_iter=2):
    """提取自 test9 的迭代拟合函数，返回: 圆心x, 圆心y, 半径R, 切点x列表, 切点y列表"""
    xc, yc, R, _, _ = fit_track_circle(track_cells)
    curr_xc, curr_yc, curr_R = xc, yc, R
    final_x, final_y = [],[]
    for _ in range(max_iter):
        all_wires = [track_cells[0].w1] +[c.w2 for c in track_cells]
        new_x, new_y = [],[]
        for w in all_wires:
            p = get_best_tangent_point(w, curr_xc, curr_yc, curr_R)
            new_x.append(p[0])
            new_y.append(p[1])
        new_x, new_y = np.array(new_x), np.array(new_y)
        res = optimize.least_squares(cost_function, (curr_xc, curr_yc, curr_R), args=(new_x, new_y))
        curr_xc, curr_yc, curr_R = res.x
        final_x, final_y = new_x, new_y
    return curr_xc, curr_yc, curr_R, final_x, final_y


# =====================================================================
# 4. 元胞自动机核心类 (可实例化配置严格/宽松)
# =====================================================================

class CATracker:
    def __init__(self, search_dist=100.0, gap_tol=1.5, angle_cut=0.98):
        """
        初始化寻迹器，支持调参以适配第一轮/第二轮。
        - 严格模式建议：search_dist=100.0, gap_tol=1.0, angle_cut=0.98
        - 宽松模式建议：search_dist=150.0, gap_tol=2.0, angle_cut=0.85
        """
        self.search_dist = search_dist
        self.gap_tol_sq = gap_tol ** 2
        self.angle_cut = angle_cut
        self.wires = []
        self.cells =[]

    def set_wires(self, wires):
        self.wires = wires

    def build_cells(self):
        self.cells =[]
        for w1, w2 in combinations(self.wires, 2):
            r1_layer = np.hypot(w1.x, w1.y)
            r2_layer = np.hypot(w2.x, w2.y)
            inner_w, outer_w = (w1, w2) if r1_layer < r2_layer else (w2, w1)

            if np.hypot(w1.x - w2.x, w1.y - w2.y) < self.search_dist:
                for p1, p2, lab in get_tangents_all(inner_w, outer_w):
                    if not is_obstructed(p1, p2, inner_w.id, outer_w.id, self.wires):
                        self.cells.append(TangentCell(inner_w, outer_w, p1, p2, lab))

    def connect_neighbors(self):
        for c1 in self.cells:
            for c2 in self.cells:
                if c1.w2.id == c2.w1.id:
                    if np.sum((c1.p2 - c2.p1) ** 2) < self.gap_tol_sq:
                        if np.dot(c1.direction, c2.direction) > self.angle_cut:
                            c1.neighbors.append(c2)

    def evolve(self):
        for _ in range(15):
            changed = False
            for c in self.cells:
                if c.neighbors:
                    s_new = 1 + max(n.state for n in c.neighbors)
                    if s_new > c.state:
                        c.state = s_new
                        changed = True
            if not changed: break

    def get_tracks(self, min_len=5):
        tracks =[]
        used = set()
        # 长度越长、状态越好的优先提取
        for c in sorted(self.cells, key=lambda x: x.state, reverse=True):
            if c in used or c.state < min_len: continue
            track, curr =[c], c
            while curr.neighbors:
                next_c = max(curr.neighbors, key=lambda x: x.state)
                if next_c in used or next_c.state >= curr.state: break
                track.append(next_c)
                curr = next_c
            if len(track) >= min_len:
                tracks.append(track)
                for t in track: used.add(t) # 霸气锁死
        return tracks
    def run_pipeline(self, min_len=7):
        """一键封装执行构建、连接、演化和提取"""
        self.build_cells()
        self.connect_neighbors()
        self.evolve()
        return self.get_tracks(min_len=min_len)

# =====================================================================
# 5. 数据读取辅助工具 (提供给主函数调用)
# =====================================================================

def parse_vector_string(vector_str):
    if pd.isna(vector_str): return 0.0, 0.0, 0.0
    pattern = r'[-+]?\d*\.\d+|\d+'
    matches = re.findall(pattern, str(vector_str))
    if len(matches) >= 3: return float(matches[0]), float(matches[1]), float(matches[2])
    return 0.0, 0.0, 0.0

def get_evt_data(df0, df1, evt_id):
    cell_coords = {row['CellID']: (row['x'], row['y'], row['u'], row['v'], row['w']) for _, row in df0.iterrows()}
    try:
        group = df1.groupby('EvtID').get_group(evt_id)
    except KeyError:
        return {'zs': [], 'xs': []}

    data = {'zs':[], 'xs': []}
    for _, row in group.iterrows():
        cid = row['CellID']
        if cid not in cell_coords: continue
        x, y, u, v, w = cell_coords[cid]
        d = row['d']

        # 忠于 test9 从第 16 列提取真值
        try:
            true_pos_str = row.iloc[16]
        except IndexError:
            true_pos_str = ""
        tx, ty, _ = parse_vector_string(true_pos_str)

        item = {'a': x, 'b': y, 'd': d, 'dir': (u, v, w), 'cell_id': cid, 'true_xy': (tx, ty), 'evt_id': evt_id}
        if u == 0 and v == 0: data['zs'].append(item)
        else: data['xs'].append(item)
    return data

def get_mixed_evt_data(df0, df1, evt_id_list):
    mixed_data = {'zs': [], 'xs':[]}
    for eid in evt_id_list:
        single_evt = get_evt_data(df0, df1, eid)
        mixed_data['zs'].extend(single_evt['zs'])
        mixed_data['xs'].extend(single_evt['xs'])
    return mixed_data