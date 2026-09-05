import numpy as np
import gurobipy as gp
import math
from utils.tool import get_dimension, read_img

def cal_feature(image):
    """优化后的频域特征计算"""
    if len(image.shape) == 3:
        f = np.fft.fftn(image)
    else:
        gray = np.mean(image, axis=-1)
        f = np.fft.fftn(gray)
    f_abs = np.abs(f)
    return float(f_abs.max() / (f_abs.sum() + 1e-8))


class FastOctTreeOptimizer:
    def __init__(self, data, max_level, var_thr, e_thr):
        self.data = data
        self.d, self.h, self.w = data.shape
        self.max_level = max_level
        self.var_thr = var_thr
        self.e_thr = e_thr
        
        self.nodes = []  # 存储所有有效节点的字典/列表
        self._build_flatten_tree()

    def _build_flatten_tree(self):
        """非递归、基于层级循环的扁平化树构建"""
        print("Building octree nodes...")
        
        # 预先计算各层的基础步长
        for level in range(self.max_level + 1):
            num_blocks = 2 ** level
            sd = self.d // num_blocks
            sh = self.h // num_blocks
            sw = self.w // num_blocks
            
            for z in range(num_blocks):
                for y in range(num_blocks):
                    for x in range(num_blocks):
                        # 计算当前块的数据范围
                        dz_start, dz_end = z * sd, (z + 1) * sd
                        dy_start, dy_end = y * sh, (y + 1) * sh
                        dx_start, dx_end = x * sw, (x + 1) * sw
                        
                        patch_data = self.data[dz_start:dz_end, dy_start:dy_end, dx_start:dx_end]
                        
                        # 计算方差和均值用于剪枝
                        mean_val = patch_data.mean()
                        var_val = np.mean((patch_data - mean_val) ** 2)
                        
                        # 剪枝判断：如果平坦，则直接标记 pruned
                        is_pruned = (var_val <= self.var_thr) and (abs(mean_val) <= self.e_thr)
                        
                        node_info = {
                            "level": level,
                            "index": (z, y, x),
                            "box": (dz_start, dz_end, dy_start, dy_end, dx_start, dx_end),
                            "data": patch_data,
                            "var": var_val,
                            "mean": mean_val,
                            "pruned": is_pruned,
                            "feature": 0.0,
                            "id": len(self.nodes)
                        }
                        self.nodes.append(node_info)
        
        # 递归剪枝传播：如果父节点被剪枝，其所有后代也必须被剪枝
        self._propagate_pruning()

    def _propagate_pruning(self):
        """利用空间坐标数学关系替代慢速的 get_descendants 递归查找"""
        # 在扁平化结构中，可以通过 level 和 grid index 快速判定父子包含关系
        # level l 的 (z, y, x) 对应 level l+1 的 (2z+dz, 2y+dy, 2x+dx)
        node_map = { (n["level"], n["index"]): n for n in self.nodes }
        
        for node in self.nodes:
            if node["pruned"]:
                # 寻找并禁用所有子孙后代
                curr_level = node["level"]
                curr_z, curr_y, curr_x = node["index"]
                
                # 遍历后续所有更深层级的节点
                for target_node in self.nodes:
                    if target_node["level"] > curr_level:
                        # 检查 target 是否在 curr 的包围盒内
                        factor = 2 ** (target_node["level"] - curr_level)
                        tz, ty, tx = target_node["index"]
                        if (tz // factor == curr_z) and (ty // factor == curr_y) and (tx // factor == curr_x):
                            target_node["pruned"] = True

    def solve_optim(self, Nb, min_level):
        self.optim_model = gp.Model("Octree_Optimization")
        self.optim_model.Params.OutputFlag = 0  # 静默求解
        
        active_vars = []
        obj_expr = []
        
        # 1. 创建决策变量并加入目标函数
        for node in self.nodes:
            if not node["pruned"]:
                # 计算特征
                node["feature"] = cal_feature(node["data"])
                
                # 创建 Gurobi 二元变量
                v_name = f"L{node['level']}_Z{node['index'][0]}_Y{node['index'][1]}_X{node['index'][2]}"
                node["var"] = self.optim_model.addVar(vtype=gp.GRB.BINARY, name=v_name)
                active_vars.append(node["var"])
                
                # 最小层级约束
                if node["level"] < min_level:
                    self.optim_model.addConstr(node["var"] == 0)
                else:
                    obj_expr.append(node["feature"] * node["var"] / (8.0 ** node["level"]))

        self.optim_model.setObjective(gp.quicksum(obj_expr), gp.GRB.MAXIMIZE)
        
        # 2. 约束 1：总激活块数限制
        self.optim_model.addConstr(gp.quicksum(active_vars) <= Nb)
        
        # 3. 约束 2 & 3：血缘互斥约束（同一条空间路径上只能激活一个）
        # 找到最深层的节点作为路径终点
        max_lvl = max(n["level"] for n in self.nodes if not n["pruned"])
        deepest_nodes = [n for n in self.nodes if n["level"] == max_lvl and not n["pruned"]]
        
        for d_node in deepest_nodes:
            # 向上追溯祖先链
            genealogy_vars = []
            curr_level = d_node["level"]
            curr_z, curr_y, curr_x = d_node["index"]
            
            for l in range(curr_level + 1):
                # 对应层级的祖先索引
                factor = 2 ** (curr_level - l)
                z_l, y_l, x_l = curr_z // factor, curr_y // factor, curr_x // factor
                
                # 在节点中查找该祖先
                for n in self.nodes:
                    if n["level"] == l and n["index"] == (z_l, y_l, x_l) and not n["pruned"]:
                        genealogy_vars.append(n["var"])
                        break
            
            # 添加互斥约束：同一血脉中激活数 <= 1（若整条链全有效则 == 1）
            if len(genealogy_vars) > 0:
                self.optim_model.addConstr(gp.quicksum(genealogy_vars) <= 1)

        # 求解
        self.optim_model.optimize()
        print(f"Optimal objective value: {self.optim_model.objVal}")

    def get_active_patches(self):
        active_list = []
        for node in self.nodes:
            if not node["pruned"] and "var" in node and node["var"].X > 0.5:
                active_list.append(node)
        return active_list