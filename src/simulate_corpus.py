"""
规模化测试语料生成器（零依赖、确定性）

项目自带的合成数据只有 66 篇，指标差异常常只有 1~3 条样本，不足以支撑结论。
这里生成一套规模大一个数量级的运维语料，用于测试检索内核在真实数据量下的表现。

保真度的边界（重要）：
- **风场名**取自 `data/OpenWindSCADA/communiy_annotations/` 里真实存在的 25 个风场
  （公开数据集 OpenWindSCADA 收录的场站）
- **机型**使用真实存在的风机型号命名
- **部件与故障类型**沿用项目知识图谱已有的 9 个部件 / 25 个故障，保证
  「部件 --HAS_FAULT--> 故障」的归属与语料一致（否则组合校验会自相矛盾）
- **其余全部是模板生成的**：机组编号、时间、症状描述、停机时长、备件、报告数字……
  各故障的相对发生频次按公开的机组故障统计粗略设定，只为让语料"像真的"，不是实测分布

用法：
    python src/simulate_corpus.py                     # 默认规模（约 1400 篇）
    python src/simulate_corpus.py --scale 0.2         # 小规模，便于快速验证
    python src/simulate_corpus.py --seed 7 --out data/generated
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# 真实存在的风场（取自 OpenWindSCADA 的社区标注文件清单）
FARMS: List[str] = [
    "anholt", "beberibe", "bjorko", "c2cB", "c2cC", "delabole", "dfw1a", "dfw1b",
    "dfw2a", "dfw2b", "dundalk", "edp", "engie", "furhlander", "ietost", "kaggle",
    "kelmarsh", "norre", "pcwg", "pedra", "penmanshiel", "saopaulo", "smart",
    "westermost", "winji",
]

# 真实机型命名（型号与实际场站的对应关系是虚构的）
MODELS: List[str] = [
    "Vestas V90-2.0MW", "Vestas V112-3.0MW", "Vestas V126-3.45MW",
    "Siemens SWT-2.3-93", "Siemens SWT-3.6-120", "Goldwind GW82/2000",
    "Goldwind GW121/2500", "Envision EN-2.0", "Envision EN-3.0",
    "Mingyang MY-2.0", "Nordex N90-2.5MW", "Enercon E-82 E4",
    "Gamesa G114-2.0MW", "Suzlon S88-2.1MW", "GE 2.75-120", "Senvion MM92",
]

# 部件 -> 故障类型（与 data/knowledge_graph.json 的 HAS_FAULT 边一致）
COMPONENT_FAULTS: Dict[str, List[str]] = {
    "齿轮箱": ["齿轮磨损", "齿轮断齿", "油封泄漏"],
    "发电机": ["发电机过热", "绝缘击穿"],
    "变桨系统": ["变桨轴承卡死", "变桨电机过热"],
    "偏航系统": ["偏航偏差过大", "偏航电机异响"],
    "叶片": ["叶片裂纹", "叶片前缘腐蚀", "叶片结冰", "叶片表面凹陷"],
    "液压系统": ["液压油泄漏", "液压泵故障"],
    "控制系统": ["PLC通信故障", "传感器漂移"],
    "变压器": ["变压器过热", "绝缘油老化"],
    "制动系统": ["制动器失效", "制动盘磨损"],
    "塔筒": ["塔筒螺栓松动"],
}

# 各故障的相对发生频次（按公开的机组故障统计粗略设定，仅用于让语料像真的）
FAULT_WEIGHTS: Dict[str, int] = {
    "叶片前缘腐蚀": 8, "传感器漂移": 8, "PLC通信故障": 6, "变桨电机过热": 6,
    "齿轮磨损": 5, "液压油泄漏": 5, "偏航电机异响": 5, "叶片表面凹陷": 4,
    "发电机过热": 4, "制动盘磨损": 4, "变桨轴承卡死": 3, "齿轮断齿": 3,
    "液压泵故障": 3, "绝缘油老化": 3, "变压器过热": 3, "偏航偏差过大": 3,
    "塔筒螺栓松动": 3, "油封泄漏": 3, "叶片裂纹": 2, "叶片结冰": 2,
    "绝缘击穿": 2, "制动器失效": 2, "主轴承损坏": 2,
}

# 字段模板：每个字段给多个说法，避免语料高度重复导致 IDF 失真
SYMPTOMS: Dict[str, List[str]] = {
    "齿轮磨损": [
        "齿轮箱振动幅值由 {low} mm/s 升至 {high} mm/s，伴随周期性异响",
        "机组报「齿轮箱振动超限」，频谱在啮合频率处出现明显边频带",
        "巡检发现齿轮箱油液铁屑含量超标，振动烈度较上月增长 {pct}%",
    ],
    "齿轮断齿": [
        "齿轮箱内部传出金属撞击声，振动瞬时峰值超过 {high} mm/s",
        "SCADA 报齿轮箱振动突变，机组触发安全停机",
    ],
    "油封泄漏": ["齿轮箱输出轴端面渗油，机舱地板出现油迹", "齿轮箱油位持续下降，日补油量约 {low} L"],
    "发电机过热": [
        "发电机绕组温度升至 {high} ℃（报警值 130 ℃），风机限功率运行",
        "发电机前后轴承温度偏差达 {pct}%，伴随焦糊气味",
    ],
    "绝缘击穿": ["发电机绝缘电阻降至 {low} MΩ 以下，合闸时报接地故障", "发电机三相绝缘不平衡度超过 {pct}%"],
    "变桨轴承卡死": ["单个叶片变桨角度滞后 {pct}%，变桨电流显著上升", "变桨动作报过载，叶片角度停在 {deg}° 不再变化"],
    "变桨电机过热": ["变桨电机温度达到 {high} ℃，连续变桨后触发过温保护", "变桨电机电流异常波动，温升速率偏高"],
    "偏航偏差过大": ["机组对风偏差持续大于 {deg}°，发电量损失约 {pct}%", "偏航液压制动后仍出现缓慢回转，风向偏差报警"],
    "偏航电机异响": ["偏航过程中电机侧传出周期性摩擦声，偏航速度不稳", "偏航减速器振动值升至 {high} mm/s"],
    "叶片裂纹": ["无人机巡检发现叶片距叶根 {deg} m 处有长度约 {low} cm 的横向裂纹", "叶片运行时传出异常气动噪声，内窥检查发现结构裂纹"],
    "叶片前缘腐蚀": ["叶片前缘防雷条与胶衣层剥落，粗糙度明显增大", "无人机巡检发现叶片前缘腐蚀面积约 {pct}%"],
    "叶片结冰": ["环境温度低于 {deg} ℃ 且湿度饱和，叶片表面结冰触发振动保护停机", "叶片结冰导致质量不平衡，机组自动降载"],
    "叶片表面凹陷": ["吊装检修发现叶片表面有 {low} cm 凹陷，探伤确认未穿透", "叶片表面雷击点周边胶衣破损并伴轻微凹陷"],
    "液压油泄漏": ["液压站油位下降，站体底部发现积液", "液压系统压力保持能力下降，日补油约 {low} L"],
    "液压泵故障": ["液压泵出口压力不足，变桨与制动动作迟缓", "液压泵运行噪声增大，压力脉动超过 {pct}%"],
    "PLC通信故障": ["主控与变桨子站通信中断 {low} 次/日，机组频繁复位", "PLC 总线丢包率超过 {pct}%，远程监控数据缺失"],
    "传感器漂移": ["风速仪测值与相邻机组偏差达 {pct}%，功率曲线右移", "机舱振动传感器零点漂移，报警阈值失真"],
    "变压器过热": ["箱变油温升至 {high} ℃，风机限功率运行", "变压器负载率超过 {pct}% 时油温报警"],
    "绝缘油老化": ["箱变绝缘油击穿电压降至 {low} kV，油色加深", "绝缘油微水含量超标，介损值上升 {pct}%"],
    "制动器失效": ["安全制动器动作后叶轮未在设定时间内停稳", "制动器摩擦片磨损至限位，制动力矩不足"],
    "制动盘磨损": ["制动盘表面出现沟槽，制动时抖动明显", "制动盘厚度减薄至 {low} mm，接近报废限值"],
    "主轴承损坏": ["主轴承温度升至 {high} ℃ 并伴随低频振动", "主轴承润滑脂铁屑含量超标，振动包络值上升"],
    "塔筒螺栓松动": ["塔筒法兰连接螺栓力矩抽检不合格，最大偏差 {pct}%", "螺栓标记线错位，塔筒摆动幅值增大"],
}

CAUSES: Dict[str, List[str]] = {
    "齿轮磨损": ["润滑油长期未按周期更换，齿面润滑不足", "长期在额定功率以上运行，齿面接触应力偏高"],
    "齿轮断齿": ["齿轮存在材料缺陷并在冲击载荷下扩展", "并网冲击导致扭矩突变，齿根疲劳断裂"],
    "油封泄漏": ["油封老化失去弹性", "输出轴对中偏差导致油封偏磨"],
    "发电机过热": ["冷却风道积尘堵塞，散热能力下降", "长期过载运行，绕组温升偏高"],
    "绝缘击穿": ["绕组绝缘受潮老化", "变频器谐波长期作用加速绝缘劣化"],
    "变桨轴承卡死": ["变桨轴承润滑脂缺失导致滚道点蚀", "密封失效进水，润滑脂乳化"],
    "变桨电机过热": ["变桨电机抱闸未完全释放形成拖拽", "连续调桨工况下电机过热保护整定偏严"],
    "偏航偏差过大": ["风向标标定漂移，控制偏差累积", "偏航制动器打滑导致机舱缓慢回转"],
    "偏航电机异响": ["偏航减速器润滑不足，齿面出现早期磨损", "偏航齿圈与驱动齿轮啮合间隙偏大"],
    "叶片裂纹": ["制造过程中的层间缺陷在交变载荷下扩展", "雷击造成结构层损伤"],
    "叶片前缘腐蚀": ["长期受雨蚀与砂蚀，前缘防护层磨损", "紫外线老化导致胶衣粉化"],
    "叶片结冰": ["低温高湿工况下叶片表面结冰", "除冰系统未投运或加热功率不足"],
    "叶片表面凹陷": ["吊装过程中发生磕碰", "运输支撑点受力过大导致局部变形"],
    "液压油泄漏": ["密封件老化、接头松动", "液压油污染导致密封件加速磨损"],
    "液压泵故障": ["液压泵内部磨损导致容积效率下降", "液压油清洁度不达标，泵体拉伤"],
    "PLC通信故障": ["通信光纤接头松动或光纤衰减超标", "子站电源模块老化导致通信复位"],
    "传感器漂移": ["传感器长期运行零点漂移", "信号电缆屏蔽层接地不良引入干扰"],
    "变压器过热": ["箱变散热片积尘、通风不畅", "负载率长期偏高，油温上升"],
    "绝缘油老化": ["绝缘油长期运行氧化，酸值升高", "呼吸器硅胶失效导致油中水分增加"],
    "制动器失效": ["制动器摩擦片磨损超限", "液压压力不足导致制动力矩下降"],
    "制动盘磨损": ["制动盘长期摩擦磨损超限", "制动器间隙调整不当导致拖磨"],
    "主轴承损坏": ["主轴承润滑脂加注周期过长", "轴电流腐蚀导致滚道损伤"],
    "塔筒螺栓松动": ["交变载荷下螺栓预紧力衰减", "安装时力矩未按标准执行"],
}

SOLUTIONS: Dict[str, List[str]] = {
    "齿轮磨损": ["停机更换齿轮箱高速轴轴承与齿轮油，复测振动恢复至 {low} mm/s"],
    "齿轮断齿": ["更换损坏齿轮副并检查同箱体其他齿轮，复测合格后并网"],
    "油封泄漏": ["更换输出轴油封并复测对中，补油至标准油位"],
    "发电机过热": ["清理冷却风道并更换滤网，复测绕组温度降至 {low} ℃ 以下"],
    "绝缘击穿": ["返厂重绕绕组并做耐压试验，绝缘电阻恢复至 {low} MΩ 以上"],
    "变桨轴承卡死": ["更换变桨轴承并加注指定润滑脂，三支叶片同步标定"],
    "变桨电机过热": ["更换变桨电机风机并重新整定过温保护曲线，复测变桨电流正常"],
    "偏航偏差过大": ["重新标定风向标并按标准执行偏航对风测试，偏差恢复至 {deg}° 以内"],
    "偏航电机异响": ["更换偏航减速器润滑油并调整啮合间隙，异响消除"],
    "叶片裂纹": ["按厂家工艺打磨补强并做真空灌注修复，修复后做超声检测"],
    "叶片前缘腐蚀": ["打磨前缘并重新涂覆防护涂层，恢复气动外形"],
    "叶片结冰": ["投运除冰系统并调整加热策略，结冰工况下降载运行"],
    "叶片表面凹陷": ["打磨找平后局部补强，受力部位加装防护层"],
    "液压油泄漏": ["更换密封件与接头并补充液压油，系统保压试验合格"],
    "液压泵故障": ["更换液压泵并更换液压油与滤芯，压力恢复至 {low} MPa"],
    "PLC通信故障": ["重新熔接光纤并更换电源模块，通信中断次数降至 0"],
    "传感器漂移": ["更换风速仪并完成风向风速标定，功率曲线恢复正常"],
    "变压器过热": ["清理散热片并加装强制通风，油温控制在 {low} ℃ 以下"],
    "绝缘油老化": ["更换绝缘油并更换呼吸器硅胶，击穿电压恢复至 {low} kV 以上"],
    "制动器失效": ["更换制动器摩擦片并调整间隙，制动力矩试验合格"],
    "制动盘磨损": ["更换制动盘并按标准调整制动器间隙"],
    "主轴承损坏": ["更换主轴承并按周期加注润滑脂，振动值恢复至 {low} mm/s"],
    "塔筒螺栓松动": ["按标准力矩重新紧固并做标记，复测合格"],
}

PREVENTIONS: List[str] = [
    "按周期检查并更换易损件，建立台账跟踪劣化趋势",
    "加装在线监测并设置分级报警，提前介入处理",
    "把该故障纳入月度专项巡检清单，重点复核",
    "优化运行策略，避免长期处于恶劣工况",
]

DISCOVERY: List[str] = [
    "SCADA 报警", "定期巡检", "在线振动监测", "无人机巡检", "人工登塔检查", "状态监测系统预警",
]

# 诊断手段与备件按部件给，避免出现「偏航故障用电气绝缘测试」「偏航换液压泵」这类外行搭配
COMPONENT_DIAGNOSIS: Dict[str, List[str]] = {
    "齿轮箱": ["振动频谱分析", "油液铁屑检测", "内窥镜检查"],
    "发电机": ["绕组温度趋势分析", "绝缘电阻与耐压试验", "振动包络分析"],
    "变桨系统": ["变桨动作时间测试", "变桨电流曲线分析", "轴承内窥检查"],
    "偏航系统": ["对风精度测试", "偏航速度与电流测试", "偏航制动压力测试"],
    "叶片": ["无人机外观巡检", "超声探伤", "内窥检查"],
    "液压系统": ["系统保压试验", "油液清洁度检测", "压力脉动分析"],
    "控制系统": ["通信报文抓包分析", "子站电源与接地检查", "传感器标定复核"],
    "变压器": ["红外测温", "绝缘油色谱分析", "负载率核算"],
    "制动系统": ["制动时间与力矩试验", "摩擦片厚度测量", "液压压力测试"],
    "塔筒": ["螺栓力矩抽检", "法兰间隙测量", "塔筒摆动监测"],
}

COMPONENT_SPARES: Dict[str, List[str]] = {
    "齿轮箱": ["高速轴轴承", "齿轮油 ISO VG 320", "输出轴油封", "齿轮副"],
    "发电机": ["发电机冷却风机", "绕组绝缘材料", "轴承润滑脂"],
    "变桨系统": ["变桨轴承", "变桨电机", "变桨润滑脂"],
    "偏航系统": ["偏航减速器润滑油", "偏航电机", "风向标", "偏航制动器摩擦片"],
    "叶片": ["叶片前缘防护涂层", "补强材料套件", "叶片螺栓"],
    "液压系统": ["液压泵", "液压油", "密封件套件", "液压滤芯"],
    "控制系统": ["PLC 电源模块", "通信光纤跳线", "振动传感器", "风速仪"],
    "变压器": ["箱变绝缘油", "呼吸器硅胶", "散热风机"],
    "制动系统": ["制动器摩擦片", "制动盘", "液压密封件"],
    "塔筒": ["塔筒螺栓", "法兰密封胶", "防腐涂料"],
}

# 严重程度 -> 停机时长区间，保证两者不矛盾
SEVERITY_DOWNTIME: Dict[str, List[int]] = {
    "高": [48, 72, 120, 168],
    "中": [12, 18, 24, 36],
    "低": [2, 4, 6, 8],
}

SPARE_PARTS: List[str] = [
    "高速轴轴承", "齿轮油 ISO VG 320", "变桨轴承", "偏航减速器润滑油", "制动器摩擦片",
    "发电机冷却风机", "液压泵", "密封件套件", "风速仪", "振动传感器", "箱变绝缘油",
    "主轴承润滑脂", "塔筒螺栓", "PLC 电源模块", "变桨电机", "齿轮副",
]

INSPECTION_ITEMS: List[tuple] = [
    ("齿轮箱油位与油质", "齿轮箱"), ("发电机绕组温度", "发电机"), ("叶片表面状态", "叶片"),
    ("变桨系统动作测试", "变桨系统"), ("偏航系统对风测试", "偏航系统"), ("液压系统保压试验", "液压系统"),
    ("制动器动作与间隙", "制动系统"), ("塔筒螺栓力矩抽检", "塔筒"), ("控制系统通信检查", "控制系统"),
    ("箱变油温与油位", "变压器"),
]

SAFETY_TOPICS: List[str] = [
    "高处作业与安全带使用", "机舱作业锁定挂牌", "带电作业安全距离", "吊装作业指挥与警戒",
    "有限空间作业", "恶劣天气登塔限制", "应急逃生与救援",
]


def _fill(template: str, rng: random.Random) -> str:
    """填充模板里的数值占位符，让同一故障的不同记录不雷同"""
    return (template
            .replace("{low}", str(rng.choice([1.2, 1.5, 2.1, 2.4, 3.0, 3.5, 6.0, 12.0, 18.0])))
            .replace("{high}", str(rng.choice([75.0, 88.0, 96.0, 110.0, 118.0, 128.0, 142.0])))
            .replace("{pct}", str(rng.choice([8, 11, 15, 18, 22, 27, 33, 41, 55])))
            .replace("{deg}", str(rng.choice([3.5, 5.0, 7.5, 8.0, 11.0, 13.5, 21.0])))
            .replace("{speed}", str(rng.choice([0.3, 0.5, 0.8, 1.2, 2.0]))))


def _weighted_fault(rng: random.Random) -> tuple:
    """按频次权重抽一个（部件, 故障）"""
    pool = [(component, fault, FAULT_WEIGHTS.get(fault, 1))
            for component, faults in COMPONENT_FAULTS.items() for fault in faults]
    total = sum(weight for _c, _f, weight in pool)
    point = rng.uniform(0, total)
    cursor = 0.0
    for component, fault, weight in pool:
        cursor += weight
        if point <= cursor:
            return component, fault
    return pool[-1][0], pool[-1][1]


def _work_orders(rng: random.Random, turbines: List[Dict], count: int) -> List[Dict]:
    """故障工单：字段格式与项目自带故障案例一致，便于复用解析与评测逻辑"""
    documents = []
    for index in range(1, count + 1):
        turbine = rng.choice(turbines)
        component, fault = _weighted_fault(rng)
        symptom = _fill(rng.choice(SYMPTOMS[fault]), rng)
        cause = _fill(rng.choice(CAUSES[fault]), rng)
        solution = _fill(rng.choice(SOLUTIONS[fault]), rng)
        year = rng.choice([2021, 2022, 2023, 2024, 2025])
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        severity = rng.choice(["高", "中", "中", "低"])
        downtime = rng.choice(SEVERITY_DOWNTIME[severity])
        fault_id = f"WO-{index:04d}"
        content = (
            f"故障ID：{fault_id}\n"
            f"风场：{turbine['farm']}\n"
            f"机组：{turbine['id']}\n"
            f"机型：{turbine['model']}\n"
            f"部件：{component}\n"
            f"故障类型：{fault}\n"
            f"发现时间：{year}-{month:02d}-{day:02d}\n"
            f"发现方式：{rng.choice(DISCOVERY)}\n"
            f"症状：{symptom}\n"
            f"原因：{cause}\n"
            f"诊断：结合 {rng.choice(COMPONENT_DIAGNOSIS[component])} 确认\n"
            f"解决方案：{solution}\n"
            f"备件：{rng.choice(COMPONENT_SPARES[component])} ×{rng.randint(1, 4)}\n"
            f"停机时长：{downtime} 小时\n"
            f"严重程度：{severity}\n"
            f"预防措施：{rng.choice(PREVENTIONS)}"
        )
        documents.append({
            "source": "work_orders",
            "content": content,
            "metadata": {
                "title": f"{turbine['farm']} {turbine['id']} {fault}",
                "component": component,
                "category": fault,
                "farm": turbine["farm"],
                "turbine": turbine["id"],
                "fault_id": fault_id,
            },
        })
    return documents


def _inspections(rng: random.Random, turbines: List[Dict], count: int) -> List[Dict]:
    documents = []
    for index in range(1, count + 1):
        turbine = rng.choice(turbines)
        item, component = rng.choice(INSPECTION_ITEMS)
        abnormal = rng.random() < 0.35
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        if abnormal:
            result = "异常"
            detail = _fill(rng.choice([
                "实测值偏离标准区间约 {pct}%", "发现轻微渗漏与积尘", "力矩抽检偏差 {pct}%",
                "动作时间比标准值长 {low} s",
            ]), rng)
            action = "已记录并安排下次停机窗口处理"
        else:
            result, detail, action = "正常", "各项参数在标准范围内", "无需处理"
        documents.append({
            "source": "inspections",
            "content": (
                f"巡检记录：{turbine['farm']} {turbine['id']}\n"
                f"巡检日期：2023-{month:02d}-{day:02d}\n"
                f"巡检项：{item}\n"
                f"所属部件：{component}\n"
                f"结果：{result}\n"
                f"异常描述：{detail}\n"
                f"处理：{action}"
            ),
            "metadata": {
                "title": f"{turbine['farm']} {turbine['id']} {item}巡检记录",
                "component": component,
                "category": item,
                "farm": turbine["farm"],
                "turbine": turbine["id"],
            },
        })
    return documents


def _monthly_reports(rng: random.Random, farms: List[str], count: int) -> List[Dict]:
    documents = []
    for index in range(1, count + 1):
        farm = rng.choice(farms)
        year, month = rng.choice([2023, 2024, 2025]), rng.randint(1, 12)
        availability = round(rng.uniform(93.0, 99.6), 2)
        generation = rng.randint(1800, 12000)
        faults = rng.randint(3, 26)
        top = [_weighted_fault(rng)[1] for _ in range(3)]
        documents.append({
            "source": "monthly_reports",
            "content": (
                f"运维月报：{farm} {year}年{month}月\n"
                f"可利用率：{availability}%\n"
                f"发电量：{generation} 万 kWh\n"
                f"故障次数：{faults} 次\n"
                f"停机损失电量：{round(generation * rng.uniform(0.01, 0.06), 1)} 万 kWh\n"
                f"高发故障：{ '、'.join(sorted(set(top))) }\n"
                f"下月重点：{'、'.join(rng.sample([item for item, _ in INSPECTION_ITEMS], 2))}"
            ),
            "metadata": {
                "title": f"{farm} {year}年{month}月运维月报",
                "category": "运维月报",
                "farm": farm,
            },
        })
    return documents


def _spare_parts(rng: random.Random) -> List[Dict]:
    documents = []
    for index, part in enumerate(SPARE_PARTS, 1):
        component = rng.choice([c for c, faults in COMPONENT_FAULTS.items()])
        documents.append({
            "source": "spare_parts",
            "content": (
                f"备件名称：{part}\n"
                f"适用部件：{component}\n"
                f"规格型号：{part[:2]}-{index:03d}\n"
                f"参考更换周期：{rng.choice([6, 12, 24, 36, 48])} 个月\n"
                f"安全库存：{rng.randint(2, 40)} 件\n"
                f"存放要求：{rng.choice(['阴凉干燥、避免挤压', '防潮防锈、竖直放置', '远离热源、密封保存'])}\n"
                f"验收要点：{rng.choice(['核对出厂合格证与批次', '检查外观无磕碰锈蚀', '抽检关键尺寸'])}"
            ),
            "metadata": {"title": f"{part} 备件规格", "category": "备件", "component": component},
        })
    return documents


def _sops(rng: random.Random) -> List[Dict]:
    documents = []
    for index, (component, faults) in enumerate(COMPONENT_FAULTS.items(), 1):
        for operation in ("例行维护", "故障处理"):
            documents.append({
                "source": "sops",
                "content": (
                    f"作业指导书：{component}{operation}\n"
                    f"适用范围：{component} 的{operation}作业\n"
                    f"安全措施：锁定挂牌、验电确认、佩戴安全带、专人监护\n"
                    f"工器具：{'、'.join(rng.sample(['力矩扳手', '振动分析仪', '红外测温仪', '绝缘电阻表', '内窥镜', '液压泵站', '万用表', '激光对中仪'], 3))}\n"
                    f"作业步骤：1) 停机并确认安全；2) 记录初始状态；3) 按清单检查{component}；"
                    f"4) 按标准处理异常项；5) 复测并恢复运行\n"
                    f"验收标准：振动 ≤ {rng.choice([2.5, 3.5, 4.5])} mm/s、温度 ≤ {rng.choice([65, 75, 85])} ℃、"
                    f"动作时间 ≤ {rng.choice([8, 12, 20])} s\n"
                    f"关注故障：{'、'.join(faults)}"
                ),
                "metadata": {"title": f"{component}{operation}作业指导书", "category": "作业指导书",
                             "component": component},
            })
    return documents


def _safety_rules(rng: random.Random) -> List[Dict]:
    documents = []
    for index, topic in enumerate(SAFETY_TOPICS, 1):
        documents.append({
            "source": "safety_rules",
            "content": (
                f"安全规程：{topic}\n"
                f"适用范围：所有风电场现场作业人员与承包商\n"
                f"基本要求：{rng.choice(['作业前完成风险辨识与工前会', '两人以上作业并指定监护人', '作业票审批后方可开工'])}\n"
                f"禁止事项：{rng.choice(['禁止无票作业', '禁止大风天气登塔', '禁止跨越安全围栏'])}\n"
                f"应急处置：立即停止作业、撤离至安全区域并按应急流程上报\n"
                f"检查频率：{rng.choice(['每季度', '每半年', '每年'])} 复核一次"
            ),
            "metadata": {"title": f"{topic}安全规程", "category": "安全规程"},
        })
    return documents


def _qa_pairs(rng: random.Random, work_orders: List[Dict], count: int) -> List[Dict]:
    """从工单派生问答对：问题带风场/机组/部件，答案取自对应工单"""
    documents = []
    for index, order in enumerate(rng.sample(work_orders, min(count, len(work_orders))), 1):
        metadata = order["metadata"]
        question = (f"{metadata['farm']} 的 {metadata['turbine']} 机组"
                    f"{rng.choice(['出现过什么', '发生过哪种', '报过什么'])}{metadata['component']}故障？")
        documents.append({
            "source": "qa_pairs",
            "content": f"问题：{question}\n答案：{order['content'].splitlines()[4]}，"
                       f"{order['content'].splitlines()[8]}；处理措施见工单 {metadata['fault_id']}。",
            "metadata": {"title": metadata["component"], "category": "故障诊断",
                         "farm": metadata["farm"], "turbine": metadata["turbine"],
                         "fault_id": metadata["fault_id"]},
        })
    return documents


def generate(scale: float = 1.0, seed: int = 20261016) -> Dict[str, List[Dict]]:
    """按比例生成全部语料（确定性：同一 seed 得到同一份数据）"""
    rng = random.Random(seed)
    farms = FARMS
    turbines = []
    for farm in farms:
        model = rng.choice(MODELS)
        for slot in range(1, rng.randint(4, 12) + 1):
            turbines.append({"farm": farm, "id": f"WT{slot:02d}", "model": model})

    def scaled(base: int) -> int:
        return max(1, int(base * scale))

    work_orders = _work_orders(rng, turbines, scaled(600))
    return {
        "work_orders": work_orders,
        "inspections": _inspections(rng, turbines, scaled(240)),
        "monthly_reports": _monthly_reports(rng, farms, scaled(120)),
        "spare_parts": _spare_parts(rng),
        "sops": _sops(rng),
        "safety_rules": _safety_rules(rng),
        "qa_pairs": _qa_pairs(rng, work_orders, scaled(200)),
    }


def save(corpus: Dict[str, List[Dict]], out_dir: str = "data/generated") -> List[str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, documents in corpus.items():
        if not documents:
            continue
        path = output / f"{name}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({name: documents}, handle, ensure_ascii=False, indent=1)
        written.append(f"{path}（{len(documents)} 篇）")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成规模化风电运维测试语料")
    parser.add_argument("--scale", type=float, default=1.0, help="规模系数（默认 1.0）")
    parser.add_argument("--seed", type=int, default=20261016, help="随机种子")
    parser.add_argument("--out", default="data/generated", help="输出目录")
    args = parser.parse_args(argv)

    corpus = generate(scale=args.scale, seed=args.seed)
    total = sum(len(documents) for documents in corpus.values())
    for line in save(corpus, args.out):
        print(f"[simulate] {line}")
    print(f"[simulate] 共 {total} 篇 → {args.out}（scale={args.scale} seed={args.seed}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
