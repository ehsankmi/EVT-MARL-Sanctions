import networkx as nx
import numpy as np

def generate_agent_network(n_agents=50, m_edges=2):
    """
    تولید شبکه پیچیده عامل‌ها با مدل Barabási–Albert
    مناسب برای شبیه‌سازی توزیع قدرت/سرمایه در سیستم‌های واقعی
    """
    G = nx.barabasi_albert_graph(n_agents, m_edges)
    
    # تخصیص ظرفیت تحمل ریسک اولیه به هر عامل
    for node in G.nodes():
        G.nodes[node]['capacity'] = np.random.uniform(10, 50)
        G.nodes[node]['risk_state'] = 'Safe'
        
    return G

def apply_evt_shock(G, threshold, shape_param):
    """
    تزریق شوک با توزیع پارتو (Peaks-Over-Threshold)
    shape_param (alpha): ضخامت دم توزیع را کنترل می‌کند
    """
    # تولید شوک‌های فرین برای تمامی عامل‌ها
    shocks = (np.random.pareto(shape_param, len(G.nodes())) + 1) * threshold
    
    failed_nodes = []
    for i, node in enumerate(G.nodes()):
        if shocks[i] > G.nodes[node]['capacity']:
            G.nodes[node]['risk_state'] = 'Failed'
            failed_nodes.append(node)
        else:
            G.nodes[node]['risk_state'] = 'Safe'
            
    return G, failed_nodes, shocks
