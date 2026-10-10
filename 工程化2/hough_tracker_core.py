import numpy as np
from scipy.optimize import least_squares


# =====================================================================
# 1. 组长终极拟合器 (供 Hough 提取内点后精修使用)
# =====================================================================

def fit_2d_circle(zs_data):
    """基于 PCA 法向网格搜索的高鲁棒性 2D 直丝圆拟合"""
    if len(zs_data) < 3: return None
    a = np.array([i['a'] for i in zs_data])
    b = np.array([i['b'] for i in zs_data])
    d = np.array([i['d'] for i in zs_data])

    a_mean, b_mean = np.mean(a), np.mean(b)
    cov_matrix = np.cov(a - a_mean, b - b_mean)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_idx = np.argmax(eigenvalues)
    v_long = eigenvectors[:, main_idx]
    v_norm = np.array([-v_long[1], v_long[0]])

    def geometric_residuals(params, a, b, d):
        A, B, R = params
        dist_to_center = np.sqrt((a - A) ** 2 + (b - B) ** 2)
        return np.abs(dist_to_center - R) - d

    best_res_sum = float('inf')
    best_guess = [a_mean, b_mean, 1000.0]
    curvatures = [-0.02, -0.005, -0.001, -0.0001, 0.0001, 0.001, 0.005, 0.02]

    for k in curvatures:
        R_guess = abs(1.0 / k)
        A_guess = a_mean + (v_norm[0] / k)
        B_guess = b_mean + (v_norm[1] / k)
        res = least_squares(geometric_residuals, [A_guess, B_guess, R_guess], args=(a, b, d), ftol=1e-2, xtol=1e-2)
        res_sum = np.sum(res.fun ** 2)
        if res_sum < best_res_sum:
            best_res_sum = res_sum
            best_guess = res.x

    final_res = least_squares(geometric_residuals, best_guess, args=(a, b, d),
                              loss='soft_l1', bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf]),
                              ftol=1e-6, xtol=1e-6)
    return final_res.x


# =====================================================================
# 2. Hough 变换底层算法核心
# =====================================================================

class HoughTracker2D:
    def __init__(self, theta_bins=1000, rho_bins=1000, rho_range=(0.0001, 0.02)):
        self.theta_bins = theta_bins
        self.rho_bins = rho_bins
        self.rho_min, self.rho_max = rho_range
        self.thetas = np.linspace(0, 2 * np.pi, self.theta_bins, endpoint=False)
        self.cos_t = np.cos(self.thetas)
        self.sin_t = np.sin(self.thetas)

    def _conformal_transform_exact(self, x, y, d):
        r2 = x ** 2 + y ** 2
        D = r2 - d ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            u = 2 * x / D
            v = 2 * y / D
            r_conf = 2 * d / D
        valid = np.isfinite(u) & np.isfinite(v)
        if not np.all(valid):
            u[~valid] = 0
            v[~valid] = 0
            r_conf[~valid] = 0
        return u, v, r_conf

    def build_accumulator(self, x_arr, y_arr, d_arr):
        u, v, r_conf = self._conformal_transform_exact(x_arr, y_arr, d_arr)

        term1 = np.outer(u, self.cos_t)
        term2 = np.outer(v, self.sin_t)
        rho_base = term1 + term2

        r_conf_col = r_conf[:, np.newaxis]
        all_rhos = np.concatenate([(rho_base + r_conf_col).flatten(), (rho_base - r_conf_col).flatten()])
        all_thetas = np.tile(self.thetas, 2 * len(u))

        mask = (all_rhos >= self.rho_min) & (all_rhos <= self.rho_max)
        valid_rhos = all_rhos[mask]
        valid_thetas = all_thetas[mask]

        if len(valid_rhos) == 0:
            return None, None, None

        H, xedges, yedges = np.histogram2d(
            valid_thetas, valid_rhos,
            bins=[self.theta_bins, self.rho_bins],
            range=[[0, 2 * np.pi], [self.rho_min, self.rho_max]]
        )
        return H, xedges, yedges

    def find_peaks_nms(self, H, xedges, yedges, top_k=20, nms_window=20, score_min=4):
        peaks = []
        flat_indices = np.argsort(H.ravel())[::-1]
        suppressed = np.zeros_like(H, dtype=bool)
        rows, cols = H.shape

        for idx in flat_indices:
            if len(peaks) >= top_k: break
            i_theta, i_rho = np.unravel_index(idx, (rows, cols))
            if suppressed[i_theta, i_rho]: continue

            score = H[i_theta, i_rho]
            if score < score_min: break

            theta_best = xedges[i_theta] + (xedges[1] - xedges[0]) / 2.0
            rho_best = yedges[i_rho] + (yedges[1] - yedges[0]) / 2.0
            R_guess = 1.0 / rho_best
            xc_guess = R_guess * np.cos(theta_best)
            yc_guess = R_guess * np.sin(theta_best)

            peaks.append((R_guess, xc_guess, yc_guess, score))

            # NMS 抑制
            t_min = i_theta - nms_window
            t_max = i_theta + nms_window + 1
            r_min = max(0, i_rho - nms_window)
            r_max = min(cols, i_rho + nms_window + 1)

            if t_min < 0:
                suppressed[t_min:, r_min:r_max] = True
                suppressed[:t_max, r_min:r_max] = True
            elif t_max > rows:
                suppressed[t_min:, r_min:r_max] = True
                suppressed[:t_max - rows, r_min:r_max] = True
            else:
                suppressed[t_min:t_max, r_min:r_max] = True

        return peaks


# =====================================================================
# 3. 面向对象封装的黑盒流水线 (方便主程序实例化调用)
# =====================================================================

class HoughPipeline:
    def __init__(self,
                 theta_bins=1000,
                 rho_bins=1000,
                 rho_range=(0.0001, 0.02),
                 top_k=30,
                 nms_window=15,
                 score_min=4,
                 residual_cut_mm=2.5,
                 rms_cut=1.0,
                 dca_cut=25.0):
        """
        初始化 Hough 黑盒寻迹器。
        可以通过修改这些参数，来配置是跑“严格版Hough”还是“宽松版Hough”。
        """
        self.tracker = HoughTracker2D(theta_bins, rho_bins, rho_range)
        self.top_k = top_k
        self.nms_window = nms_window
        self.score_min = score_min
        self.residual_cut_mm = residual_cut_mm
        self.rms_cut = rms_cut
        self.dca_cut = dca_cut

    def run(self, zs_raw_data):
        """
        输入: zs_raw_data (击中点字典列表)
        输出: 经过拟合与仲裁后的物理轨迹列表 final_tracks
        """
        if len(zs_raw_data) < 5:
            return []

        # 1. 提取数组准备喂给 Hough
        x_arr = np.array([d['a'] for d in zs_raw_data])
        y_arr = np.array([d['b'] for d in zs_raw_data])
        d_arr = np.array([d['d'] for d in zs_raw_data])

        # 2. 构建累加器
        H, xedges, yedges = self.tracker.build_accumulator(x_arr, y_arr, d_arr)
        if H is None:
            return []

        # 3. 提取峰值
        seeds = self.tracker.find_peaks_nms(
            H, xedges, yedges,
            top_k=self.self.top_k if hasattr(self, 'top_k') else 30,  # 兼容老版本属性
            nms_window=self.nms_window,
            score_min=self.score_min
        )

        # 4. 从峰值中捞取 Inliers，并进行组长重度拟合
        scored_tracks = []
        for (R_g, xc_g, yc_g, score) in seeds:
            u_ids = set()
            t_data = []
            for item in zs_raw_data:
                dist = np.sqrt((item['a'] - xc_g) ** 2 + (item['b'] - yc_g) ** 2)
                res = abs(dist - R_g) - item['d']
                if abs(res) < self.residual_cut_mm:
                    if item['cell_id'] not in u_ids:
                        u_ids.add(item['cell_id'])
                        t_data.append(item)

            if len(t_data) < 5:
                continue

            fit_p = fit_2d_circle(t_data)
            if fit_p is None: continue

            A, B, R_fit = fit_p
            res_arr = [abs(np.sqrt((d['a'] - A) ** 2 + (d['b'] - B) ** 2) - R_fit) - d['d'] for d in t_data]
            rms = np.sqrt(np.mean(np.array(res_arr) ** 2))
            dca = abs(np.sqrt(A ** 2 + B ** 2) - R_fit)

            scored_tracks.append({
                'fit_p': fit_p,
                'rms': rms,
                'dca': dca,
                'wire_ids': u_ids,
                'wire_data': t_data,
                'score': score
            })

        # 5. 极速智能仲裁 (按长度降序、RMS升序排序)
        scored_tracks.sort(key=lambda x: (-len(x['wire_ids']), x['rms']))
        final_tracks = []

        for candidate in scored_tracks:
            if candidate['rms'] >= self.rms_cut or candidate['dca'] >= self.dca_cut:
                continue

            is_clone = False
            for accepted in final_tracks:
                shared_hits = candidate['wire_ids'].intersection(accepted['wire_ids'])
                if len(shared_hits) / len(candidate['wire_ids']) > 0.6:
                    is_clone = True
                    break

            if is_clone: continue

            unique_hits = [wid for wid in candidate['wire_ids'] if not any(wid in t['wire_ids'] for t in final_tracks)]
            if len(unique_hits) >= 3:
                final_tracks.append(candidate)

        return final_tracks

    def get_candidates(self, zs_raw_data):
        """只负责找峰值、捞取内点并进行重度拟合，不进行克隆去重仲裁"""
        if len(zs_raw_data) < 5: return []

        x_arr = np.array([d['a'] for d in zs_raw_data])
        y_arr = np.array([d['b'] for d in zs_raw_data])
        d_arr = np.array([d['d'] for d in zs_raw_data])

        H, xedges, yedges = self.tracker.build_accumulator(x_arr, y_arr, d_arr)
        if H is None: return []

        seeds = self.tracker.find_peaks_nms(
            H, xedges, yedges, top_k=self.top_k, nms_window=self.nms_window, score_min=self.score_min
        )

        scored_tracks = []
        for (R_g, xc_g, yc_g, score) in seeds:
            u_ids = set()
            t_data = []
            for item in zs_raw_data:
                dist = np.sqrt((item['a'] - xc_g) ** 2 + (item['b'] - yc_g) ** 2)
                res = abs(dist - R_g) - item['d']
                if abs(res) < self.residual_cut_mm:
                    if item['cell_id'] not in u_ids:
                        u_ids.add(item['cell_id'])
                        t_data.append(item)

            if len(t_data) < 5: continue

            fit_p = fit_2d_circle(t_data)  # 调用文件顶部的拟合器
            if fit_p is None: continue

            A, B, R_fit = fit_p
            res_arr = [abs(np.sqrt((d['a'] - A) ** 2 + (d['b'] - B) ** 2) - R_fit) - d['d'] for d in t_data]
            rms = np.sqrt(np.mean(np.array(res_arr) ** 2))
            dca = abs(np.sqrt(A ** 2 + B ** 2) - R_fit)

            scored_tracks.append({
                'fit_p': fit_p, 'rms': rms, 'dca': dca,
                'wire_ids': u_ids, 'wire_data': t_data, 'source': 'Hough'
            })
        return scored_tracks