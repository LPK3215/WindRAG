"""
Gradio Web界面
WindRAG - 风电领域智能问答系统
"""

import gradio as gr
import numpy as np
import time
import json
import sys
import os
sys.path.append('.')

from run_rag import WindTurbineRAG


# 初始化RAG系统
print("正在初始化RAG系统...")
rag = WindTurbineRAG()
print("RAG系统初始化完成！")

# 获取系统信息
llm_status = "已启用" if rag.use_llm else "模板生成（LLM未配置）"
if rag.use_llm and rag.model_manager.models:
    llm_model = rag.model_manager.models[0].name
else:
    llm_model = "模板生成"

adaptive_status = "已加载" if rag.adaptive_fusion_loaded else "未加载（使用固定α=0.7）"


def query_rag(question, top_k, method, history):
    """RAG查询函数，返回结构化结果"""
    if not question.strip():
        return history, "", "", "⚠️ 请输入问题", ""

    start_time = time.time()

    try:
        # 获取自适应α值
        alpha_info = ""
        if method == "hybrid" and rag.adaptive_fusion_loaded:
            query_emb = rag.model.encode([question])[0]
            alpha_val = rag.adaptive_fusion.predict_alpha(question, query_emb)
            alpha_info = f"🎯 自适应α = {alpha_val:.3f}"
            if alpha_val < 0.5:
                alpha_info += "（偏向BM25精确匹配）"
            elif alpha_val > 0.7:
                alpha_info += "（偏向语义检索）"
            else:
                alpha_info += "（均衡融合）"
        elif method == "hybrid":
            alpha_info = "🔧 固定α = 0.700"

        result = rag.query(question, top_k=int(top_k), method=method)
        elapsed = time.time() - start_time

        answer = result['answer']

        # 格式化检索文档为Markdown表格
        docs_md = "| # | 评分 | 来源 | 部件 | 内容摘要 |\n|---|------|------|------|----------|\n"
        for i, doc in enumerate(result['retrieved_docs'], 1):
            score = f"{doc['score']:.4f}"
            source = doc['metadata'].get('source', '-')
            component = doc['metadata'].get('component', '-')
            content_preview = doc['content'][:80].replace('\n', ' ').replace('|', '\\|')
            docs_md += f"| {i} | {score} | {source} | {component} | {content_preview}... |\n"

        # 知识图谱上下文（从检索结果中提取）
        kg_info = ""
        for doc in result['retrieved_docs']:
            if doc.get('metadata', {}).get('source') == 'knowledge_graph':
                kg_info = f"📊 **知识图谱命中**\n\n{doc['content'][:300]}"
                break

        status = f"✅ 查询成功 | 耗时 {elapsed:.2f}s | 检索 {len(result['retrieved_docs'])} 篇文档"

        # 更新聊天历史（Gradio 6.x messages格式）
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return history, docs_md, alpha_info, status, kg_info

    except Exception as e:
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ 查询失败: {str(e)}"})
        return history, "", "", f"❌ 错误: {str(e)}", ""


def get_system_info():
    """获取系统状态信息"""
    kg_info = "未加载"
    if rag.kg:
        edge_count = sum(len(v) for v in rag.kg.edges.values())
        kg_info = f"{len(rag.kg.nodes)} 节点 / {edge_count} 边"
    emb_count = rag.embeddings.shape[0] if rag.embeddings is not None else 0
    info = f"""### 系统状态
| 项目 | 状态 |
|------|------|
| 嵌入模型 | `text2vec-base-chinese` (768维) |
| LLM | {llm_model} ({llm_status}) |
| 自适应融合 | {adaptive_status} |
| 知识图谱 | {kg_info} |
| 文档块数 | {emb_count} |
| BM25索引 | {'已构建' if rag.bm25 else '未构建'} |
| 向量索引 | {'已构建' if rag.embeddings is not None else '未构建'} |
"""
    return info


def clear_chat():
    """清空聊天"""
    return [], "", "", "", ""


# 自定义CSS
custom_css = """
.gradio-container { max-width: 1200px !important; }
#chatbot { min-height: 400px; }
.status-bar { font-size: 0.9em; }
"""

# 创建Gradio界面
with gr.Blocks(title="WindRAG - 风电智能问答") as app:

    gr.Markdown("""
    # 🌬️ WindRAG - 风电领域智能问答系统
    **基于多策略混合检索与知识图谱增强的检索增强生成（RAG）系统** | 支持自适应融合权重预测
    """)

    with gr.Tabs():
        # ===== Tab 1: 问答 =====
        with gr.TabItem("💬 智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话",
                        elem_id="chatbot",
                        height=400
                    )

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="输入问题",
                            placeholder="例如：风电机的齿轮箱温度过高应该怎么处理？",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="检索文档数", scale=1
                        )
                        method_dropdown = gr.Dropdown(
                            choices=["hybrid", "vector", "bm25"],
                            value="hybrid",
                            label="检索方法", scale=1
                        )
                        clear_btn = gr.Button("🗑️ 清空对话", scale=1)

                with gr.Column(scale=2):
                    alpha_display = gr.Textbox(
                        label="🎯 自适应融合权重",
                        interactive=False, lines=2
                    )
                    status_display = gr.Textbox(
                        label="状态", interactive=False,
                        elem_classes="status-bar"
                    )
                    docs_display = gr.Markdown(
                        label="检索文档",
                        value="*查询后显示检索到的文档*"
                    )
                    kg_display = gr.Markdown(
                        label="知识图谱",
                        value=""
                    )

            gr.Markdown("### 📝 示例问题")
            examples = gr.Examples(
                examples=[
                    ["风电机的齿轮箱温度过高应该怎么处理？", 3, "hybrid"],
                    ["如何判断风电机叶片是否需要更换？", 3, "hybrid"],
                    ["ISO VG 320齿轮油更换周期", 3, "hybrid"],
                    ["风电机组的偏航系统有什么作用？", 5, "hybrid"],
                    ["轴承温度超过85℃报警阈值", 3, "hybrid"],
                    ["变桨系统的工作原理是什么？", 3, "vector"],
                    ["齿轮箱振动值超过7mm/s怎么处理？", 3, "bm25"],
                ],
                inputs=[question_input, top_k_slider, method_dropdown],
            )

            # 绑定事件
            submit_btn.click(
                fn=query_rag,
                inputs=[question_input, top_k_slider, method_dropdown, chatbot],
                outputs=[chatbot, docs_display, alpha_display, status_display, kg_display]
            ).then(lambda: "", outputs=question_input)

            question_input.submit(
                fn=query_rag,
                inputs=[question_input, top_k_slider, method_dropdown, chatbot],
                outputs=[chatbot, docs_display, alpha_display, status_display, kg_display]
            ).then(lambda: "", outputs=question_input)

            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, docs_display, alpha_display, status_display, kg_display]
            )

        # ===== Tab 2: 系统信息 =====
        with gr.TabItem("⚙️ 系统信息"):
            sys_info = gr.Markdown(value=get_system_info())
            refresh_btn = gr.Button("🔄 刷新状态")
            refresh_btn.click(fn=get_system_info, outputs=sys_info)

            gr.Markdown("""
            ### 检索方法说明
            | 方法 | 说明 | 适用场景 |
            |------|------|----------|
            | `hybrid` | 向量 + BM25 混合检索（支持自适应α） | **推荐**，综合效果最佳 |
            | `vector` | 纯向量语义检索 | 口语化、语义描述类问题 |
            | `bm25` | 纯BM25关键词检索 | 含精确术语、编号的问题 |

            ### 自适应融合权重（α）
            系统根据查询特征自动预测最优融合权重：
            - **α → 0**：偏向BM25精确匹配（适合含标准编号、数值规格的查询）
            - **α → 1**：偏向向量语义检索（适合口语化、概念描述类查询）
            - **α ≈ 0.5**：均衡融合（适合含领域术语的常规查询）
            """)

        # ===== Tab 3: 关于 =====
        with gr.TabItem("ℹ️ 关于"):
            gr.Markdown("""
            ### WindRAG
            基于多策略混合检索与知识图谱增强的风电领域智能诊断问答系统。

            **核心特性：**
            - 三路混合检索：向量语义 + BM25关键词 + 知识图谱结构
            - 查询自适应融合权重预测（规则先验 + MLP残差）
            - 知识图谱语义查询（embedding余弦相似度匹配）
            - 线程安全的LLM模型池管理器（自动切换 + 优雅降级）

            **技术栈：**
            `Python` · `SentenceTransformers` · `BM25` · `NetworkX` · `Gradio` · `ModelScope API`

            **开源协议：** MIT License
            """)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
