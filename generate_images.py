"""
生成WindRAG项目科研绘图
用于README.md和项目文档
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json
from pathlib import Path

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

output_dir = Path("docs/images")
output_dir.mkdir(parents=True, exist_ok=True)


# ==================== 图1：系统架构图 ====================
def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 颜色方案
    colors = {
        'data': '#4A90D9',
        'index': '#50B86C',
        'retrieval': '#F5A623',
        'generation': '#D0021B',
        'arrow': '#666666',
        'text': '#333333',
        'bg_data': '#E8F0FE',
        'bg_index': '#E6F4EA',
        'bg_retrieval': '#FEF3E0',
        'bg_gen': '#FCE8E8',
    }

    # 层级背景
    layers = [
        (0.3, 0.3, 13.4, 1.4, colors['bg_data'], '数据层'),
        (0.3, 2.0, 13.4, 1.4, colors['bg_index'], '索引层'),
        (0.3, 3.7, 13.4, 1.8, colors['bg_retrieval'], '检索层'),
        (0.3, 5.8, 13.4, 1.8, colors['bg_gen'], '生成层'),
    ]

    for x, y, w, h, color, label in layers:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#999999', linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.25, label, fontsize=11, fontweight='bold',
                color=colors['text'], va='top')

    # 数据层组件
    data_items = [
        (1.5, 0.7, '故障案例\n30条', colors['data']),
        (4.0, 0.7, '技术文档\n10篇', colors['data']),
        (6.5, 0.7, '运维手册\n10篇', colors['data']),
        (9.0, 0.7, '问答对\n10组', colors['data']),
        (11.5, 0.7, 'OpenWind\nSCADA', '#7B68EE'),
    ]
    for x, y, text, color in data_items:
        rect = FancyBboxPatch((x-0.7, y-0.15), 1.8, 0.8, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + 0.25, text, fontsize=8, color='white',
                ha='center', va='center', fontweight='bold')

    # 索引层组件
    index_items = [
        (2.5, 2.4, '向量索引\ntext2vec-base-chinese\nL2归一化余弦相似度', colors['index']),
        (7.0, 2.4, 'BM25索引\njieba中文分词\n词频-逆文档频率', colors['index']),
        (11.0, 2.4, '知识图谱\n162节点/160边\n轻量级实现', '#2E8B57'),
    ]
    for x, y, text, color in index_items:
        rect = FancyBboxPatch((x-1.2, y-0.15), 2.8, 0.85, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + 0.28, text, fontsize=7.5, color='white',
                ha='center', va='center', fontweight='bold')

    # 检索层组件
    ret_items = [
        (1.8, 4.3, '向量检索', colors['retrieval']),
        (4.2, 4.3, 'BM25检索', colors['retrieval']),
        (6.6, 4.3, '图谱检索\n(语义匹配)', '#E07020'),
        (9.5, 4.3, '加权融合\nα=自适应', '#CC6600'),
        (12.0, 4.3, '语义\n重排序', '#B8860B'),
    ]
    # Fix the typo
    ret_items[1] = (4.2, 4.3, 'BM25检索', colors['retrieval'])
    for x, y, text, color in ret_items:
        rect = FancyBboxPatch((x-0.7, y-0.2), 1.8, 0.9, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + 0.25, text, fontsize=8, color='white',
                ha='center', va='center', fontweight='bold')

    # 检索层箭头
    arrow_pairs = [(2.7, 4.55), (5.3, 4.55), (7.7, 4.55), (11.1, 4.55)]
    for x1, x2 in [(2.7, 3.5), (5.3, 5.8), (7.7, 8.8), (11.1, 11.3)]:
        ax.annotate('', xy=(x2, 4.55), xytext=(x1, 4.55),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # 生成层组件
    gen_items = [
        (3.0, 6.3, '模型池管理器\n线程安全·自动切换', '#C0392B'),
        (7.5, 6.3, 'LLM生成\nQwen系列', '#E74C3C'),
        (11.5, 6.3, '模板降级\nAPI不可用时', '#9B59B6'),
    ]
    for x, y, text, color in gen_items:
        rect = FancyBboxPatch((x-1.2, y-0.2), 2.8, 0.9, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + 0.25, text, fontsize=8, color='white',
                ha='center', va='center', fontweight='bold')

    # 层间箭头
    for x in [2.5, 7.0, 11.0]:
        ax.annotate('', xy=(x, 2.0), xytext=(x, 1.7),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    for x in [3.0, 9.5]:
        ax.annotate('', xy=(x, 3.7), xytext=(x, 3.4),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(7.5, 5.8), xytext=(7.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # 标题
    ax.text(7, 7.85, 'WindRAG 系统架构', fontsize=16, fontweight='bold',
            ha='center', va='center', color='#2C3E50')

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ 架构图已生成")


# ==================== 图2：混合检索流程图 ====================
def draw_retrieval_flow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 查询输入
    rect = FancyBboxPatch((0.2, 2.0), 1.5, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#3498DB', edgecolor='#2980B9', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.95, 2.5, '用户查询', fontsize=11, color='white', ha='center', va='center', fontweight='bold')

    # 向量检索
    rect = FancyBboxPatch((2.5, 3.2), 2.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#2ECC71', edgecolor='#27AE60', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.5, 3.7, '向量检索\n(L2归一化)', fontsize=9, color='white', ha='center', va='center', fontweight='bold')

    # BM25检索
    rect = FancyBboxPatch((2.5, 0.8), 2.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#F39C12', edgecolor='#E67E22', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.5, 1.3, 'BM25检索\n(jieba分词)', fontsize=9, color='white', ha='center', va='center', fontweight='bold')

    # 融合
    rect = FancyBboxPatch((5.5, 2.0), 2.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#E74C3C', edgecolor='#C0392B', linewidth=2)
    ax.add_patch(rect)
    ax.text(6.5, 2.5, '自适应融合\nα·V+(1-α)·B', fontsize=9, color='white', ha='center', va='center', fontweight='bold')

    # 重排序
    rect = FancyBboxPatch((8.5, 2.0), 1.5, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#9B59B6', edgecolor='#8E44AD', linewidth=2)
    ax.add_patch(rect)
    ax.text(9.25, 2.5, '语义\n重排序', fontsize=9, color='white', ha='center', va='center', fontweight='bold')

    # 输出
    rect = FancyBboxPatch((10.5, 2.0), 1.3, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#1ABC9C', edgecolor='#16A085', linewidth=2)
    ax.add_patch(rect)
    ax.text(11.15, 2.5, 'Top-K\n结果', fontsize=9, color='white', ha='center', va='center', fontweight='bold')

    # 箭头
    arrows = [
        (1.7, 2.5, 2.5, 3.7),   # 查询→向量
        (1.7, 2.5, 2.5, 1.3),   # 查询→BM25
        (4.5, 3.7, 5.5, 2.7),   # 向量→融合
        (4.5, 1.3, 5.5, 2.3),   # BM25→融合
        (7.5, 2.5, 8.5, 2.5),   # 融合→重排序
        (10.0, 2.5, 10.5, 2.5), # 重排序→输出
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # α标注
    ax.text(5.2, 3.0, 'α', fontsize=14, color='#E74C3C', fontweight='bold', ha='center')
    ax.text(5.2, 1.7, '1-α', fontsize=12, color='#E74C3C', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_flow.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ 检索流程图已生成")


# ==================== 图3：实验结果柱状图 ====================
def draw_experiment_results():
    # 加载实验数据
    report_path = Path("data/validation_report.json")
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    else:
        report = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 子图1：稳定性测试
    ax1 = axes[0]
    categories = ['长时间运行\n(50次)', '并发查询\n(15次)']
    success_rates = [100, 100]
    avg_times = [0.65, 2.59]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax1.bar(x - width/2, success_rates, width, label='成功率(%)', color='#2ECC71', alpha=0.85)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, avg_times, width, label='平均响应时间(s)', color='#3498DB', alpha=0.85)

    ax1.set_ylabel('成功率 (%)', fontsize=10)
    ax1_twin.set_ylabel('响应时间 (s)', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1_twin.set_ylim(0, 4)
    ax1.set_title('系统稳定性', fontsize=12, fontweight='bold')

    # 添加数值标注
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{bar.get_height():.0f}%', ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{bar.get_height():.2f}s', ha='center', fontsize=9, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # 子图2：问题类型覆盖
    ax2 = axes[1]
    qtypes = ['故障诊断', '维护操作', '系统原理', '性能优化']
    success = [100, 100, 100, 100]
    resp_times = [0.69, 0.65, 0.62, 0.67]

    x = np.arange(len(qtypes))
    bars = ax2.bar(x, resp_times, color=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71'], alpha=0.85, width=0.6)
    ax2.set_ylabel('平均响应时间 (s)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(qtypes, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title('问题类型覆盖 (成功率100%)', fontsize=12, fontweight='bold')

    for bar, t in zip(bars, resp_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{t:.2f}s', ha='center', fontsize=9, fontweight='bold')

    # 子图3：鲁棒性测试
    ax3 = axes[2]
    robust_types = ['空输入', '超长输入', '特殊字符', '乱码', '英文', '混合']
    robust_times = [0.66, 0.82, 0.62, 0.62, 0.61, 0.64]

    colors = ['#95A5A6', '#E74C3C', '#F39C12', '#7F8C8D', '#3498DB', '#9B59B6']
    bars = ax3.barh(robust_types, robust_times, color=colors, alpha=0.85, height=0.6)
    ax3.set_xlabel('响应时间 (s)', fontsize=10)
    ax3.set_xlim(0, 1.2)
    ax3.set_title('鲁棒性测试 (全部通过)', fontsize=12, fontweight='bold')

    for bar, t in zip(bars, robust_times):
        ax3.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f'{t:.2f}s ✓', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_results.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ 实验结果图已生成")


# ==================== 图4：数据覆盖雷达图 ====================
def draw_data_coverage():
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))

    categories = ['故障案例', '技术文档', '运维手册', '问答对',
                  '部件覆盖', '故障类型', '文档类别', '知识图谱节点']
    values = [30, 10, 10, 10, 12, 30, 13, 162]

    # 归一化到0-1
    max_vals = [50, 20, 20, 20, 15, 50, 20, 200]
    norm_values = [v/m for v, m in zip(values, max_vals)]
    norm_values += norm_values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.fill(angles, norm_values, color='#3498DB', alpha=0.25)
    ax.plot(angles, norm_values, color='#3498DB', linewidth=2, marker='o', markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)

    # 标注实际值
    for angle, val, cat in zip(angles[:-1], values, categories):
        ax.text(angle, 1.15, f'{val}', ha='center', fontsize=9, fontweight='bold', color='#E74C3C')

    ax.set_title('数据覆盖度', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_coverage.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ 数据覆盖雷达图已生成")


# ==================== 图5：自适应α分布图 ====================
def draw_alpha_distribution():
    training_path = Path("data/adaptive_fusion_training.json")
    if training_path.exists():
        with open(training_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        alphas = [d["optimal_alpha"] for d in data]
    else:
        alphas = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # α值分布直方图
    if alphas:
        ax1.hist(alphas, bins=np.arange(0.05, 1.0, 0.1), color='#3498DB', alpha=0.8, edgecolor='white')
        ax1.axvline(x=0.7, color='#E74C3C', linestyle='--', linewidth=2, label='固定α=0.7')
        ax1.axvline(x=np.mean(alphas), color='#2ECC71', linestyle='-', linewidth=2, label=f'均值α={np.mean(alphas):.2f}')
    ax1.set_xlabel('α值', fontsize=11)
    ax1.set_ylabel('查询数量', fontsize=11)
    ax1.set_title('最优α值分布', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)

    # 不同查询类型的推荐α
    qtypes = ['含标准编号\n(ISO/GB)', '含数值规格\n(mm/℃)', '术语密集\n(≥3个)',
              '故障诊断类', '维护操作类', '系统原理类', '性能优化类']
    recommended_alpha = [0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.75]
    colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C']

    bars = ax2.barh(qtypes, recommended_alpha, color=colors, alpha=0.85, height=0.6)
    ax2.axvline(x=0.7, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.5, label='固定α=0.7')
    ax2.set_xlabel('推荐α值', fontsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.set_title('查询类型与推荐α值', fontsize=13, fontweight='bold')

    for bar, a in zip(bars, recommended_alpha):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'α={a}', va='center', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ α分布图已生成")


# ==================== 图6：知识图谱可视化 ====================
def draw_knowledge_graph():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis('off')

    # 部件节点（中心环形布局）
    components = ['齿轮箱', '发电机', '叶片', '变桨系统', '偏航系统',
                  '液压系统', '制动系统', '塔筒', '变压器', '控制系统', '轴承', '电气系统']

    n = len(components)
    radius = 3.5
    comp_pos = {}
    for i, comp in enumerate(components):
        angle = 2 * np.pi * i / n - np.pi / 2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        comp_pos[comp] = (x, y)

        circle = plt.Circle((x, y), 0.55, facecolor='#3498DB', edgecolor='#2980B9',
                           linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, comp, fontsize=8, ha='center', va='center',
               color='white', fontweight='bold')

    # 故障节点（外围）
    fault_types = ['齿轮磨损', '轴承过热', '叶片裂纹', '偏航偏差', '变桨卡涩',
                   '液压泄漏', '制动失效', '绝缘下降', '振动异常', '温度过高',
                   '油质变质', '腐蚀损伤']
    fault_radius = 5.2
    fault_pos = {}
    for i, fault in enumerate(fault_types):
        angle = 2 * np.pi * i / len(fault_types) - np.pi / 2 + 0.15
        x = fault_radius * np.cos(angle)
        y = fault_radius * np.sin(angle)
        fault_pos[fault] = (x, y)

        circle = plt.Circle((x, y), 0.45, facecolor='#E74C3C', edgecolor='#C0392B',
                           linewidth=1.5, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x, y, fault, fontsize=6.5, ha='center', va='center',
               color='white', fontweight='bold')

    # 连线（部件→故障）
    connections = [
        ('齿轮箱', '齿轮磨损'), ('齿轮箱', '油质变质'), ('齿轮箱', '温度过高'),
        ('发电机', '绝缘下降'), ('发电机', '温度过高'), ('发电机', '振动异常'),
        ('叶片', '叶片裂纹'), ('叶片', '腐蚀损伤'),
        ('偏航系统', '偏航偏差'),
        ('变桨系统', '变桨卡涩'),
        ('液压系统', '液压泄漏'),
        ('制动系统', '制动失效'),
        ('轴承', '轴承过热'), ('轴承', '振动异常'),
    ]
    for comp, fault in connections:
        if comp in comp_pos and fault in fault_pos:
            x1, y1 = comp_pos[comp]
            x2, y2 = fault_pos[fault]
            ax.plot([x1, x2], [y1, y2], color='#BDC3C7', linewidth=1, alpha=0.6, zorder=0)

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#3498DB', edgecolor='#2980B9', label='部件节点 (12)'),
        mpatches.Patch(facecolor='#E74C3C', edgecolor='#C0392B', label='故障节点 (30+)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_title('风电领域知识图谱', fontsize=15, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'knowledge_graph.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ 知识图谱可视化已生成")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("WindRAG 科研绘图生成")
    print("=" * 50)

    draw_architecture()
    draw_retrieval_flow()
    draw_experiment_results()
    draw_data_coverage()
    draw_alpha_distribution()
    draw_knowledge_graph()

    print(f"\n所有图片已保存到 {output_dir}/")
