import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from core.network_sim import generate_agent_network, apply_evt_shock

# پیکربندی صفحه
st.set_page_config(page_title="EVT-MARL Risk Simulator", layout="wide")
st.title("شبیه‌سازی تعاملی ریسک سیستمی در محیط چندعامله")
st.markdown("ارزیابی مقاومت شبکه در برابر شوک‌های دنباله‌پهن (Heavy-tailed Shocks)")

# پنل تنظیمات پارامترها
st.sidebar.header("پارامترهای محیط و ریسک")
n_agents = st.sidebar.slider("تعداد عامل‌ها (Network Size)", 20, 150, 50)
shape_param = st.sidebar.slider("پارامتر شکل توزیع (Tail Heaviness - α)", 1.1, 4.0, 2.0, step=0.1)
threshold = st.sidebar.slider("آستانه ریسک پایه (POT Threshold)", 1, 30, 5)

# اجرای شبیه‌سازی
G = generate_agent_network(n_agents)
G, failed_nodes, shocks = apply_evt_shock(G, threshold, shape_param)

# بخش‌بندی صفحه برای نمایش داده‌ها
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("توپولوژی شبکه و وضعیت عامل‌ها")
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # رنگ‌بندی بر اساس وضعیت شکست یا ایمنی
    color_map = ['#FF4B4B' if G.nodes[node]['risk_state'] == 'Failed' else '#00CC96' for node in G]
    node_sizes = [G.nodes[node]['capacity'] * 10 for node in G]
    
    nx.draw(G, pos, node_color=color_map, node_size=node_sizes, 
            with_labels=False, ax=ax, edge_color='#E0E0E0', alpha=0.9)
    st.pyplot(fig)

with col2:
    st.subheader("متریک‌های ریسک سیستمی")
    failure_rate = (len(failed_nodes) / n_agents) * 100
    
    st.metric(label="عامل‌های از کار افتاده", value=f"{len(failed_nodes)} / {n_agents}")
    st.metric(label="نرخ شکست سیستمی", value=f"{failure_rate:.1f}%")
    st.metric(label="شدیدترین شوک ثبت شده", value=f"{np.max(shocks):.2f}")
    
    st.markdown("---")
    st.markdown("**راهنمای رنگ‌ها:**")
    st.markdown("🟢 **سبز:** ظرفیت عامل > شدت شوک")
    st.markdown("🔴 **قرمز:** وقوع شکست (سرریز ریسک فرین)")
