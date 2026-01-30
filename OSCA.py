import numpy as np
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class OCSA:
    """
    基于论文的Optimal Connected Subgraph-Based Conflict-free Association实现
    输入：多无人机的成对关联分数矩阵
    输出：无冲突的目标关联结果（每个目标映射到全局唯一ID）
    """
    def __init__(self, sigma: float = 0.5):
        """
        初始化OCSA
        :param sigma: 论文中的手动参数（防止边权过大），默认0.5（与论文一致）
        """
        self.sigma = sigma
        self.global_id = 0  # 全局目标ID计数器
        self.target_to_global_id = {}  # 目标→全局ID映射：key=(drone_id, target_id), value=global_id
    
    def _unify_score_type(self, score_matrix: np.ndarray, is_similarity: bool = True) -> np.ndarray:
        """
        统一关联分数类型为相异度（论文2.2.1节）
        :param score_matrix: 成对关联分数矩阵 (drone_m_target_num, drone_n_target_num)
        :param is_similarity: 输入分数是否为相似度（True=相似度→转相异度；False=已为相异度）
        :return: 统一后的相异度矩阵
        """
        if not is_similarity:
            return score_matrix
        
        # 相似度转相异度：使用最大相似度归一化后取反（确保数值在合理范围）
        max_sim = score_matrix.max() if score_matrix.max() != 0 else 1.0
        dissimilarity = max_sim - score_matrix
        return dissimilarity
    
    def _select_association_pairs(self, 
                                 score_matrix: np.ndarray,
                                 drone_m: int,
                                 drone_n: int,
                                 threshold: float = 0.0) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        筛选关联对（论文2.2.2节）：选择行列均为最小值的元素作为有效关联对
        :param score_matrix: 统一后的相异度矩阵
        :param drone_m: 无人机m的ID
        :param drone_n: 无人机n的ID
        :param threshold: 最小相异度阈值（小于阈值才视为有效关联）
        :return: 关联对列表：[( (m, t1), (n, t2), 相异度分数 ), ...]
        """
        association_pairs = []
        rows, cols = score_matrix.shape
        
        # 找到每行最小值的位置和数值
        row_mins = np.min(score_matrix, axis=1)
        row_min_cols = np.argmin(score_matrix, axis=1)
        
        # 找到每列最小值的位置和数值
        col_mins = np.min(score_matrix, axis=0)
        col_min_rows = np.argmin(score_matrix, axis=0)
        
        # 筛选行列均为最小值的元素
        for i in range(rows):
            for j in range(cols):
                if (score_matrix[i, j] == row_mins[i] and 
                    score_matrix[i, j] == col_mins[j] and 
                    score_matrix[i, j] <= threshold):
                    # 目标标识：(无人机ID, 目标在该无人机的本地ID)
                    target_m = (drone_m, i + 1)  # 目标ID从1开始（符合论文表述）
                    target_n = (drone_n, j + 1)
                    association_pairs.append((target_m, target_n, score_matrix[i, j]))
        
        return association_pairs
    
    def _build_connected_graph(self, all_association_pairs: List[Tuple[Tuple[int, int], Tuple[int, int], float]]) -> List[Dict]:
        """
        构建连通图（论文2.3.1节）：目标为节点，关联对为边，边权=1/(相异度+sigma)
        :param all_association_pairs: 所有无人机对的关联对列表
        :return: 连通图列表，每个图格式：
                {
                    'nodes': 节点集合 {(drone_id, target_id), ...},
                    'edges': 边列表 [(node1, node2, weight), ...],
                    'edge_weights': 边权字典 {(node1, node2): weight, ...}
                }
        """
        # 构建全局边字典
        global_edges = {}
        all_nodes = set()
        
        for (t1, t2, dissim) in all_association_pairs:
            # 计算边权：论文公式（4）
            weight = 1.0 / (dissim + self.sigma)
            # 确保边的双向性
            if (t1, t2) not in global_edges and (t2, t1) not in global_edges:
                global_edges[(t1, t2)] = weight
                global_edges[(t2, t1)] = weight
            all_nodes.add(t1)
            all_nodes.add(t2)
        
        # 基于边构建连通图（使用并查集快速查找连通分量）
        parent = {node: node for node in all_nodes}
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        
        # 合并连通节点
        for (t1, t2) in global_edges:
            if t1 in parent and t2 in parent:
                union(t1, t2)
        
        # 分组连通分量
        components = {}
        for node in all_nodes:
            root = find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)
        
        # 构建每个连通图的详细信息
        connected_graphs = []
        for comp_nodes in components.values():
            comp_nodes_set = set(comp_nodes)
            comp_edges = []
            comp_edge_weights = {}
            
            # 筛选该连通分量内的所有边
            for (t1, t2), weight in global_edges.items():
                if t1 in comp_nodes_set and t2 in comp_nodes_set:
                    comp_edges.append((t1, t2, weight))
                    comp_edge_weights[(t1, t2)] = weight
            
            connected_graphs.append({
                'nodes': comp_nodes_set,
                'edges': comp_edges,
                'edge_weights': comp_edge_weights
            })
        
        # 添加孤立节点（无关联的目标）作为单独的连通图
        for node in all_nodes:
            if len([g for g in connected_graphs if node in g['nodes']]) == 0:
                connected_graphs.append({
                    'nodes': {node},
                    'edges': [],
                    'edge_weights': {}
                })
        
        return connected_graphs
    
    def _has_conflict(self, graph: Dict) -> bool:
        """
        冲突判断（论文2.3.2节）：同一无人机的多个目标是否在同一连通图
        :param graph: 连通图
        :return: True=有冲突，False=无冲突
        """
        drone_targets = {}
        for (drone_id, target_id) in graph['nodes']:
            if drone_id not in drone_targets:
                drone_targets[drone_id] = []
            drone_targets[drone_id].append(target_id)
        
        # 若任一无人机有多个目标在该图中，则存在冲突
        for drone_id, targets in drone_targets.items():
            if len(targets) > 1:
                return True
        return False
    
    def _calculate_node_edge_weight_sum(self, node: Tuple[int, int], graph: Dict) -> float:
        """
        计算节点的所有边权和（用于选择初始节点）
        :param node: 目标节点 (drone_id, target_id)
        :param graph: 连通图
        :return: 该节点所有关联边的权重和
        """
        weight_sum = 0.0
        for (t1, t2), weight in graph['edge_weights'].items():
            if t1 == node or t2 == node:
                weight_sum += weight
        return weight_sum
    
    def _select_initial_node(self, graph: Dict) -> Tuple[int, int]:
        """
        选择初始节点（论文2.3.3节）：选择边权和最大的节点
        :param graph: 冲突连通图
        :return: 初始节点 (drone_id, target_id)
        """
        node_weight_sums = {
            node: self._calculate_node_edge_weight_sum(node, graph)
            for node in graph['nodes']
        }
        # 选择边权和最大的节点
        return max(node_weight_sums.items(), key=lambda x: x[1])[0]
    
    def _get_candidate_nodes(self, current_subgraph: Set[Tuple[int, int]], graph: Dict) -> List[Tuple[int, int]]:
        """
        筛选候选节点（论文算法1步骤2）：与当前子图连通且无冲突的节点
        :param current_subgraph: 当前子图的节点集合
        :param graph: 原始冲突连通图
        :return: 候选节点列表
        """
        candidate_nodes = []
        current_drones = {drone_id for (drone_id, _) in current_subgraph}
        
        # 遍历所有不在当前子图的节点
        for node in graph['nodes'] - current_subgraph:
            node_drone = node[0]
            # 候选节点条件：1. 与当前子图有边连接；2. 所属无人机不在当前子图中（无冲突）
            has_connection = any(
                (node, t) in graph['edge_weights'] or (t, node) in graph['edge_weights']
                for t in current_subgraph
            )
            no_conflict = node_drone not in current_drones
            
            if has_connection and no_conflict:
                candidate_nodes.append(node)
        
        return candidate_nodes
    
    def _calculate_node_contribution(self, node: Tuple[int, int], current_subgraph: Set[Tuple[int, int]], graph: Dict) -> float:
        """
        计算候选节点的贡献值（边权和增量）
        :param node: 候选节点
        :param current_subgraph: 当前子图
        :param graph: 原始冲突连通图
        :return: 节点加入子图后的边权和增量
        """
        contribution = 0.0
        for t in current_subgraph:
            if (node, t) in graph['edge_weights']:
                contribution += graph['edge_weights'][(node, t)]
        return contribution
    
    def _search_optimal_subgraph(self, graph: Dict) -> Tuple[Set[Tuple[int, int]], List[Tuple[Tuple[int, int], Tuple[int, int], float]]]:
        """
        最优子图搜索（论文算法1：加法贪心策略）
        :param graph: 冲突连通图
        :return: 最优无冲突子图的节点集和边集
        """
        # 步骤1：选择边权和最大的初始节点
        initial_node = self._select_initial_node(graph)
        optimal_nodes = {initial_node}
        optimal_edges = []
        
        while True:
            # 步骤2：筛选候选节点
            candidate_nodes = self._get_candidate_nodes(optimal_nodes, graph)
            if not candidate_nodes:
                break
            
            # 步骤3：选择贡献值最大的候选节点
            node_contributions = {
                node: self._calculate_node_contribution(node, optimal_nodes, graph)
                for node in candidate_nodes
            }
            best_node = max(node_contributions.items(), key=lambda x: x[1])[0]
            
            # 添加节点和对应的边
            optimal_nodes.add(best_node)
            for t in optimal_nodes - {best_node}:
                if (best_node, t) in graph['edge_weights']:
                    weight = graph['edge_weights'][(best_node, t)]
                    optimal_edges.append((best_node, t, weight))
        
        return optimal_nodes, optimal_edges
    
    def resolve_conflicts(self, connected_graphs: List[Dict]) -> Dict[Tuple[int, int], int]:
        """
        冲突解决（论文2.3节）：处理所有连通图，输出全局ID映射
        :param connected_graphs: 连通图列表
        :return: 目标→全局ID映射
        """
        target_to_global_id = {}
        pending_graphs = connected_graphs.copy()
        
        while pending_graphs:
            # 取出第一个待处理图
            graph = pending_graphs.pop(0)
            
            if not self._has_conflict(graph):
                # 无冲突：所有节点分配同一全局ID
                for node in graph['nodes']:
                    target_to_global_id[node] = self.global_id
                self.global_id += 1
            else:
                # 有冲突：搜索最优子图
                optimal_nodes, _ = self._search_optimal_subgraph(graph)
                # 最优子图节点分配当前全局ID
                for node in optimal_nodes:
                    target_to_global_id[node] = self.global_id
                self.global_id += 1
                
                # 剩余节点组成新图，加入待处理队列
                remaining_nodes = graph['nodes'] - optimal_nodes
                if remaining_nodes:
                    # 构建剩余节点的连通图
                    remaining_edges = []
                    remaining_edge_weights = {}
                    for (t1, t2), weight in graph['edge_weights'].items():
                        if t1 in remaining_nodes and t2 in remaining_nodes:
                            remaining_edges.append((t1, t2, weight))
                            remaining_edge_weights[(t1, t2)] = weight
                    
                    pending_graphs.append({
                        'nodes': remaining_nodes,
                        'edges': remaining_edges,
                        'edge_weights': remaining_edge_weights
                    })
        
        return target_to_global_id
    
    def run(self,
            pairwise_score_matrices: List[Dict],
            is_similarity: bool = True,
            threshold: float = 0.0) -> Dict[Tuple[int, int], int]:
        """
        执行OCSA完整流程
        :param pairwise_score_matrices: 成对关联分数矩阵列表，每个元素格式：
                                       {
                                           'drone_m': 无人机m的ID（int）,
                                           'drone_n': 无人机n的ID（int）,
                                           'score_matrix': 关联分数矩阵（np.ndarray）
                                       }
        :param is_similarity: 输入分数是否为相似度（True=需要转相异度）
        :param threshold: 关联对筛选的相异度阈值
        :return: 最终无冲突关联结果：key=(drone_id, target_id), value=global_id
        """
        # 重置状态
        self.global_id = 0
        self.target_to_global_id = {}
        
        # 步骤1：处理所有成对关联，筛选有效关联对
        all_association_pairs = []
        for pair_data in pairwise_score_matrices:
            drone_m = pair_data['drone_m']
            drone_n = pair_data['drone_n']
            score_matrix = pair_data['score_matrix']
            
            # 统一分数类型为相异度
            dissim_matrix = self._unify_score_type(score_matrix, is_similarity)
            
            # 筛选关联对
            pairs = self._select_association_pairs(dissim_matrix, drone_m, drone_n, threshold)
            all_association_pairs.extend(pairs)
        
        # 步骤2：构建连通图
        connected_graphs = self._build_connected_graph(all_association_pairs)
        
        # 步骤3：冲突解决，生成全局ID映射
        self.target_to_global_id = self.resolve_conflicts(connected_graphs)
        
        return self.target_to_global_id


# ------------------------------ 示例：如何使用OCSA ------------------------------
if __name__ == "__main__":
    # 示例1：模拟3个无人机的成对关联分数矩阵（论文场景）
    # 无人机ID：1, 2, 3；每个无人机检测到的目标数：2, 2, 2
    pairwise_score_matrices = [
        # 无人机1 ↔ 无人机2：相似度矩阵（值越大越相似）
        {
            'drone_m': 1,
            'drone_n': 2,
            'score_matrix': np.array([
                [0.92, 0.31],  # 无人机1的目标1与无人机2的目标1/2的相似度
                [0.28, 0.89]   # 无人机1的目标2与无人机2的目标1/2的相似度
            ])
        },
        # 无人机1 ↔ 无人机3：相似度矩阵
        {
            'drone_m': 1,
            'drone_n': 3,
            'score_matrix': np.array([
                [0.87, 0.42],  # 无人机1的目标1与无人机3的目标1/2的相似度
                [0.35, 0.91]   # 无人机1的目标2与无人机3的目标1/2的相似度
            ])
        },
        # 无人机2 ↔ 无人机3：相似度矩阵
        {
            'drone_m': 2,
            'drone_n': 3,
            'score_matrix': np.array([
                [0.93, 0.29],  # 无人机2的目标1与无人机3的目标1/2的相似度
                [0.33, 0.88]   # 无人机2的目标2与无人机3的目标1/2的相似度
            ])
        }
    ]
    
    # 创建OCSA实例
    ocsa = OCSA(sigma=0.5)
    
    # 运行OCSA（输入为相似度矩阵，阈值设为0.1）
    result = ocsa.run(
        pairwise_score_matrices=pairwise_score_matrices,
        is_similarity=True,
        threshold=0.1
    )
    
    # 打印结果
    print("=" * 60)
    print("OCSA无冲突关联结果（目标 → 全局ID）")
    print("=" * 60)
    for (drone_id, target_id), global_id in sorted(result.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"无人机{drone_id}的目标{target_id} → 全局ID: {global_id}")
    
    # 示例2：验证冲突解决效果（模拟有冲突的场景）
    print("\n" + "=" * 60)
    print("冲突场景测试")
    print("=" * 60)
    # 构造有冲突的关联对：无人机1的目标1同时关联无人机2的目标1和目标2
    conflicting_pairs = [
        {
            'drone_m': 1,
            'drone_n': 2,
            'score_matrix': np.array([
                [0.95, 0.94],  # 无人机1的目标1与无人机2的两个目标都高度相似（造成冲突）
                [0.21, 0.18]
            ])
        },
        {
            'drone_m': 2,
            'drone_n': 3,
            'score_matrix': np.array([
                [0.92, 0.33],
                [0.27, 0.89]
            ])
        }
    ]
    
    conflicting_result = ocsa.run(
        pairwise_score_matrices=conflicting_pairs,
        is_similarity=True,
        threshold=0.1
    )
    
    for (drone_id, target_id), global_id in sorted(conflicting_result.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"无人机{drone_id}的目标{target_id} → 全局ID: {global_id}")