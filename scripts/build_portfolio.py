"""Build the six-page Chinese portfolio from verified public project artifacts."""
from pathlib import Path
import csv, json, os
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/modeling-analytics-portfolio.pdf'
W,H=595.276,841.89; M=45; CW=W-2*M
NAVY=HexColor('#243C55');TEAL=HexColor('#137C8B');MUTED=HexColor('#627284');BG=HexColor('#EFF4F6');ORANGE=HexColor('#C47B35')
URL='https://github.com/lpwei-quant/modeling-analytics-portfolio'

def setup_fonts():
    normal=os.environ.get('PORTFOLIO_FONT','C:/Windows/Fonts/msyh.ttc')
    bold=os.environ.get('PORTFOLIO_BOLD_FONT','C:/Windows/Fonts/msyhbd.ttc')
    if Path(normal).is_file():
        pdfmetrics.registerFont(TTFont('CN',normal,subfontIndex=0))
        pdfmetrics.registerFont(TTFont('CNBold',bold if Path(bold).is_file() else normal,subfontIndex=0))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        return 'STSong-Light','STSong-Light'
    return 'CN','CNBold'

FONT,BOLD=setup_fonts()
pdfmetrics.registerFontFamily(FONT,normal=FONT,bold=BOLD,italic=FONT,boldItalic=BOLD)
STYLE=ParagraphStyle('body',fontName=FONT,fontSize=10.2,leading=16.5,textColor=NAVY,wordWrap='CJK',spaceAfter=0)
SMALL=ParagraphStyle('small',parent=STYLE,fontSize=8,leading=12,textColor=MUTED)
CAP=ParagraphStyle('cap',parent=STYLE,fontSize=8.6,leading=13.2,textColor=MUTED)
used=[]

def para(c,text,y,width=CW,x=M,style=STYLE):
    p=Paragraph(text,style);_,height=p.wrap(width,H)
    if y-height<47:raise ValueError(f'Page content overflows: {text[:50]} at {y-height}')
    p.drawOn(c,x,y-height);return y-height-10

def title(c,kicker,title,subtitle=None):
    c.setFillColor(TEAL);c.setFont('Helvetica-Bold',9);c.drawString(M,H-43,kicker)
    c.setFillColor(NAVY);c.setFont(BOLD,23);c.drawString(M,H-80,title)
    y=H-99
    if subtitle:y=para(c,subtitle,y,style=CAP)
    return y-7

def footer(c,page):
    c.setStrokeColor(HexColor('#D7E0E7'));c.line(M,39,W-M,39)
    c.setFillColor(MUTED);c.setFont(FONT,7);c.drawString(M,25,'魏来平 | 数据分析与策略建模 | 2026.09')
    c.setFont('Helvetica',8);c.drawRightString(W-M,25,f'{page} / 6')
    c.showPage()

def heading(c,text,y):
    c.setFillColor(TEAL);c.rect(M,y-13,3,13,fill=1,stroke=0)
    c.setFillColor(NAVY);c.setFont(BOLD,12);c.drawString(M+10,y-11,text)
    return y-26

def figure(c,name,y,height):
    path=ROOT/'assets'/name;used.append(path.relative_to(ROOT).as_posix())
    img=ImageReader(str(path));iw,ih=img.getSize();scale=min(CW/iw,height/ih)
    h=ih*scale;w=iw*scale;c.drawImage(img,M+(CW-w)/2,y-h,width=w,height=h,mask='auto')
    return y-h-7

def table(c,rows,widths,y,small=9.3):
    sty=ParagraphStyle('table',parent=STYLE,fontSize=small,leading=14)
    data=[[Paragraph(str(v),sty) for v in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BG),('TEXTCOLOR',(0,0),(-1,0),TEAL),
      ('LINEBELOW',(0,0),(-1,0),.7,HexColor('#C6D5DD')),('LINEBELOW',(0,1),(-1,-1),.35,HexColor('#E3E9ED')),
      ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
      ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    _,h=t.wrap(CW,H)
    if y-h<48:raise ValueError('Table overflow')
    t.drawOn(c,M,y-h);return y-h-15

def callout(c,big,text,y):
    h=70;c.setFillColor(BG);c.roundRect(M,y-h,CW,h,8,fill=1,stroke=0)
    c.setFont(BOLD,19);c.setFillColor(TEAL);c.drawString(M+15,y-28,big)
    para(c,text,y-39,width=CW-30,x=M+15,style=CAP)
    return y-h-18

def load_csv(p):
    return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
    c.setTitle('魏来平 | 数据分析与策略建模作品集');c.setAuthor('魏来平');c.setSubject('2026 CUMCM C与2024 C赛题训练：方法、证据、个人贡献与复现')
    micro={r['strategy']:r for r in load_csv('projects/microgrid-2026/data/strategy_summary.csv')}
    agri={r['plan']:r for r in load_csv('projects/agriculture-2024/data/q3_common_paths_summary.csv') if r['surplus_rule']=='half_price_surplus'}
    q1=json.loads((ROOT/'projects/microgrid-2026/verification/representative.json').read_text('utf-8'))
    y=title(c,'SELECTED WORK / 2026','数据分析与策略建模','魏来平 · 上海财经大学 · 投资学 × 数学双学位本科生')
    y=para(c,'从问题与指标出发，用可复现的数据处理、策略比较和结果核验支持判断。这里收录一次正式竞赛和一次赛题训练，展示方法选择、分析结果及其适用边界。',y)
    y=heading(c,'我的贡献',y-8)
    y=para(c,'<b>建模方案审查、研究推进与成果整理。</b>参与问题拆解和假设取舍，追问关键约束与结论依据，组织审阅意见和成果表达。AI 实质参与模型草案、代码、求解和绘图，具体贡献按项目记录说明。',y)
    y=heading(c,'01 / 微网购电与储能调度：预测不确定性下的策略分析',y-8)
    y=para(c,'<b>2026 年 9 月 · 全国大学生数学建模竞赛 C 题 · 已提交参赛作品</b><br/>以 334 日历史回放比较计划、反馈与改约策略。将费用拆成可核算的组成部分，并同时报告期末库存。',y)
    y=callout(c,'3.59%','F-fast 相对场景固定调度的现金费用下降；回顾性比较，库存条件见第 3 页。',y)
    y=heading(c,'02 / 农业种植规划：收益与风险的情景分析',y)
    y=para(c,'<b>2026 年 8 月 · 使用 2024 年 C 题开展训练</b><br/>覆盖 54 个地块、41 种作物。用 60 个优化情景构建方案，在 3000 条模拟样本外路径上评价收益与风险。',y)
    y=callout(c,'保留更有证据支持的基线','相关风险扩展方案未超过冻结 Q2，复杂度不作为采用新方案的理由。',y)
    y=para(c,'<b>阅读入口</b><br/>第 2-3 页：微网案例　·　第 4-5 页：农业案例　·　第 6 页：贡献与复现<br/><link href="'+URL+'" color="#137C8B">github.com/lpwei-quant/modeling-analytics-portfolio</link>',y,style=CAP)
    footer(c,1)

    y=title(c,'CASE 01 / PROBLEM & MODEL','先核对信息，再制定决策','微网购电与储能调度 · 2026 C 题')
    y=para(c,'负载和光伏存在预测偏差，提前买少了会产生高价紧急购电，买多了又可能闲置。分析需要同时回答：当时能知道什么、动作是否可执行、费用怎样结算。',y)
    y=table(c,[['输入','决策','检查'],['365 天 × 144 格<br/>负载 / 光伏 / 价格','日前购电、储能动作<br/>日内合同调整','发布时间与可用信息<br/>能量、库存和现金账本']],[CW*.32,CW*.34,CW*.34],y)
    y=heading(c,'结构审查：从互斥变量到状态增量',y)
    y=para(c,'我要求重新审查初始充放电互斥处理，推动第一问重做。最终模型利用储电净变化恢复充放电动作，在既定条件下由 865 个变量的模型转为 289 个连续变量的精确 LP；保留显式互斥模型作对照。',y)
    y=figure(c,'microgrid-dispatch.png',y,244)
    y=para(c,'图：题给代表日的一组最优调度。功率以 MW 显示，库存以 MWh 显示；源数据为 144 格冻结派生表。',y,style=CAP)
    y=callout(c,f'{q1["q1_cost_yuan"]:,.2f} 元 / 26.90%','代表日优化费用 / 相对无储能参照的费用下降；不代表全年节费率。',y)
    y=para(c,'<b>适用条件：</b>单程效率各 0.9、正电价、允许弃光及指定库存边界。存在同费多解；变量减少没有被包装为经过测试的运行加速。',y,style=CAP)
    footer(c,2)

    y=title(c,'CASE 01 / STRATEGY EVALUATION','把费用变化拆成可解释的部分','2025-02-01 至 12-31 · 334 日历史回放 · 相同共同预热')
    y=figure(c,'microgrid-costs.png',y,235)
    rows=[['策略','总现金 / 万元','末库存 / kWh']]
    for k,label in [('q2/D','D：确定性计划'),('q2/S','S：场景固定调度'),('q2/F_fast','F-fast：日内反馈'),('q3/Z2_111','Q3：融合预报与改约')]:
        r=micro[k];rows.append([label,f'{float(r["total_cash_yuan"])/1e4:,.2f}',f'{float(r["final_inventory_kwh"]):,.2f}'])
    y=table(c,rows,[CW*.48,CW*.26,CW*.26],y)
    y=heading(c,'变化来自紧急购电费下降',y)
    y=para(c,'F-fast 相对 S 的计划费增加约 <b>0.57 万元</b>，紧急购电费减少约 <b>51.81 万元</b>，现金合计下降 <b>3.59%</b>。Q3 主配置再比 F-fast 减少 <b>27.29 万元</b>。拆分后的账本让改善来源可以独立核算。',y)
    y=heading(c,'预测精度不等于最终策略价值',y)
    y=para(c,'研究保留了 18 点官方预报误差高于历史预测、而完整策略费用较低的结果。预报融合、剩余窗口、储能和改约成本共同影响决策，应分别评价预测指标和策略成本。',y)
    y=para(c,'<b>边界：</b>策略是在已比较配置中回顾性选择；末库存不同。当期十分钟实测近似快速量测条件。费用差额不能直接解释为现实运营收益、纯信息价值或因果效果。<br/>来源：strategy_summary.csv、F06_q2_annual.csv；完整口径见仓库案例。',y,style=CAP)
    footer(c,3)

    y=title(c,'CASE 02 / DATA & RISK','把收益和较差情形一起纳入评价','2026 年 8 月训练 · 2024 年 C 题 · 54 个地块 / 41 种作物 / 1213 亩')
    y=para(c,'在地类、季次、轮作和豆类要求下制定 2024-2030 年种植方案。销量、亩产、价格和成本存在不确定性，平均利润之外，还需要观察较差情形下的表现。',y)
    y=table(c,[['数据与口径','建模与评价'],['附件与 2023 种植基线<br/>按作物、地块、年份、季次对齐','60 个优化情景产生固定方案<br/>3000 条新模拟路径作样本外评价'],['销量代理、分布、相关结构均标明性质','平均利润 + 90% 下尾 CVaR<br/>同一组路径比较 Q2 与 Q3']],[CW/2,CW/2],y)
    y=figure(c,'agriculture-distribution.png',y,235)
    y=para(c,'图：两方案在同一相关风险环境中的 3000 条模拟利润分布；超产按半价销售。横轴为七年名义利润（百万元），不是已实现收益。',y,style=CAP)
    y=heading(c,'比较口径先于模型复杂度',y)
    y=para(c,'Q2 表示冻结的基准方案，Q3 表示相关风险扩展候选。这里将二者放在同一个 Q3 相关环境中评价，避免把“换了评价环境”误写成“模型带来提升”。',y)
    y=para(c,'<b>风险指标：</b>90% 下尾 CVaR 描述模拟利润中较差约 10% 情形的平均水平。它依赖所设定的随机模型，不能替代真实的联合历史数据。',y,style=CAP)
    footer(c,4)

    y=title(c,'CASE 02 / FINDING & DECISION','新方案没有超过基线，也值得报告','统一模拟样本外环境 · 固定方案评价 · 结果不反过来修改观测')
    rows=[['方案','平均利润 / 万元','下尾 CVaR / 万元']]
    for key,label in [('q2_frozen','冻结 Q2'),('q3_final','Q3 候选')]:
        r=agri[key];rows.append([label,f'{float(r["mean_profit_yuan"])/1e4:,.2f}',f'{float(r["cvar_90_yuan"])/1e4:,.2f}'])
    y=table(c,rows,[CW*.30,CW*.35,CW*.35],y)
    y=para(c,'Q3 候选的均值与下尾风险指标都未超过冻结 Q2。在本组假设下，保留 Q2 更有依据；Q3 的价值在于展示共同冲击对尾部风险的影响。',y)
    y=figure(c,'agriculture-risk.png',y,240)
    y=para(c,'图：基准弹性、超产半价规则下，假设依赖强度从弱到强的敏感性。两面板使用相同纵轴；下尾 CVaR 比均值更敏感。',y,style=CAP)
    y=heading(c,'我的研究与表达工作',y)
    y=para(c,'参与模型和假设审查，按阶段确认采纳范围与冻结结果；推动章节、图表和结论对应真实证据。保留扩展方案的负结果，明确 60 个优化情景与 3000 条评价路径的不同用途。',y)
    y=para(c,'<b>局限：</b>共同因子和弹性来自透明建模设定，未经本地数据经验识别；超产市场容量会影响结果。模拟名义利润不是现实收益。<br/>来源：q3_common_paths_summary.csv、q3_sensitivity_summary.csv。',y,style=CAP)
    footer(c,5)

    y=title(c,'EVIDENCE / CONTRIBUTION / REPRODUCIBILITY','让每项展示都能继续追问','贡献边界、计算依据与公开入口')
    y=heading(c,'个人贡献与辅助工具',y)
    y=para(c,'<b>个人：</b>建模方案审查、研究推进与成果整理。2026 年主动追问互斥处理并推动重做，参与修订和最终核验；2024 C 题训练中参与假设取舍、阶段采纳、证据与论文表达整理。<br/><b>AI 辅助：</b>参与模型草案、程序实现、求解、图表及文稿辅助。可运行成果与个人独立编码能力分别说明。',y)
    y=heading(c,'本次实际复现与检查',y)
    y=table(c,[['项目','本次检查'],['微网','重新求解 Q1；核对 144 格能量、库存、费用；14 个题目变体 + 90 个构造小例与显式互斥对照；复算年度公开汇总。'],['农业','从公开包重新生成同种子 3000 条相关路径，评价冻结 Q2 / Q3，核对利润、下尾风险与农艺约束。'],['展示材料','图表从公开派生表生成；核对短版、长版、案例与本 PDF 的数字和角色口径。']],[CW*.18,CW*.82],y)
    y=para(c,'完整年度优化与全部种植优化保留原入口和冻结证据，本次没有全量重跑。农业利润数值一致至浮点精度，历史情景数组的二进制指纹未完全一致。具体范围与误差见仓库验证记录。',y,style=CAP)
    y=heading(c,'打开作品集',y)
    y=para(c,'<link href="'+URL+'" color="#137C8B">github.com/lpwei-quant/modeling-analytics-portfolio</link><br/>首页提供两个案例、精选研究记录、代码、数据来源、完整论文与投递文案。官方原始附件按来源和哈希获取，不在公开包中重复分发。',y)
    y=heading(c,'两条复现路径',y)
    y=para(c,'<b>快速阅读：</b>从冻结派生结果重建图表，运行代表性求解和固定方案评价。<br/><b>深入复算：</b>按说明获取官方输入，安装锁定依赖，运行原计算入口。输入与模型哈希用于追溯版本。',y)
    y=para(c,'2026 年为正式参赛且已提交；2024 年为在 2026 年完成的赛题训练。作品集不声明尚无依据的奖项、场站部署、实际增收或互联网实验经历。',y,style=CAP)
    footer(c,6)
    c.save();print(str(OUT))

if __name__=='__main__':main()
