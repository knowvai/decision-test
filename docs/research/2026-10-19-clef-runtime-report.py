"""Render all audited Clef scheduling results without model/service access."""
import argparse
import json
from pathlib import Path


def render(data):
    assert data["passed"]
    result = data["performance"]
    out = ["# Clef-Flash Runtime 参数实测", "",
           f"独立离线审计通过：{data['total_calls']:,} 次 valid/HTTP200 调用，{data['blocks']} 个块，"
           f"其中预热 {data['calls']['warmup']:,} 次，实测 {data['calls']['performance']:,} 次。"
           "零失败、超时和未尝试。", "",
           "## 测试口径", "",
           "- 同一张 A100-PCIE-40GB GPU0，BF16/TP1、8 CPU quota、96GiB RAM，原生 SGLang joint schema head。",
           "- 固定镜像、权重、8192上下文、0.85显存预算、原始提示、native choice；不量化、不截断、不启用结果缓存。",
           "- 两任务各128个复用开发集输入；每配置两轮正反顺序，七次冷部署，首尾相同基线。不是新的留出质量测试。",
           "- RPS按有效请求数/两轮完整block wall；延迟合并逐调用后求mean/P95/P99，不平均百分位。",
           "- 计时包含HTTP、鉴权和服务端排队，不含等客户端空位、装载及预热。全部配置启用相同metrics。",
           "- 每任务预热32条长度分散输入，在n1/n64各两轮；n64预热实际最多32条在途，不能声称覆盖所有测量执行形状。",
           "- 短程闭环负载，不能外推最大可持续并发或生产SLA；慢请求和两轮均保留，不只取最快轮。",
           "- 无reranker或额外主机推理；代理机业务embedding/reranker/proxy未参与本轮负载。", "",
           "## 参数配置", "",
           "n是客户端同时在途请求数，不是GPU批大小。running限制调度中的请求数，prefill限制完整输入合批token预算。", "",
           "| 档位 | max-running-requests | max-prefill-tokens | 装载至就绪s | GPU0最大采样MiB |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for profile, config in data["profiles"].items():
        d = data["deployments"][profile]
        out.append(f"| {profile} | {config['seqs']} | {config['tokens']} | {d['loading_seconds']:.1f} | {d['max_sampled_gpu0_memory_mib']:.0f} |")
    out += ["", "原生模型路径自动禁用radix/prefix复用与chunked prefill，以完整非因果前向读出joint head。"
            "这些限制没有被调参绕开；实际解析配置保存在完整审计JSON。", "",
            "## 候选筛选", "",
            "事前规则：相同n的两轮RPS都至少提升5%，合并P95不超过基线110%，零失败、零判断变化，"
            "首尾基线RPS漂移不超过10%。通过只代表复验候选，不是生产认证。", "",
            "| 档位 | 场景 | n | 吞吐提升% | P95变化% | 首尾基线漂移% |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    found = False
    for profile, tasks in result.items():
        for task, levels in tasks.items():
            for n, v in levels.items():
                if v["promising_candidate"]:
                    found = True
                    out.append(f"| {profile} | {task} | {n} | {(v['rps_ratio']-1)*100:+.1f} | {(v['p95_ratio']-1)*100:+.1f} | {v['baseline_end_rps_drift']*100:+.1f} |")
    if not found:
        out.append("| 无满足全部规则的候选 | | | | | |")
    out += ["", "## 全部性能结果", "",
            "每格256次尝试/有效，来自128个独立输入的两次重复；所有格子失败/超时/未尝试均为0。"
            "吞吐比是对相同并发baseline。判断变化按输入ID和轮次配对，不按文件行号配对。", ""]
    for task in ("classification", "evidence"):
        out += [f"### {task}", "", "| 档位 | n | RPS | 两轮RPS | Mean ms | P95 ms | P99 ms | 吞吐比 | 对基线变化/256 | 对n1变化/256 |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for profile, tasks in result.items():
            for n, v in tasks[task].items():
                rounds = " / ".join(f"{r['rps']:.2f}" for r in v["rounds"])
                out.append(f"| {profile} | {n} | {v['rps']:.2f} | {rounds} | {v['mean_ms']:.2f} | {v['p95_ms']:.2f} | "
                           f"{v['p99_ms']:.2f} | {v['rps_ratio']:.3f} | {v['vs_baseline']['decision_changes']} | {v['vs_n1']['decision_changes']} |")
        out.append("")
    out += ["## 输出稳定性与审计", "",
            "性能子集质量仅用于判断runtime改变是否影响输出，不用于重新排名模型；"
            "片段标签仍是T2Ranking L2/L3正例的代理标准，不能当作逐条事实真伪或最终RAG回答质量。", "",
            "| 场景 | baseline n1 Accuracy% | Precision% | Recall% | F1% |", "| --- | ---: | ---: | ---: | ---: |"]
    for task in ("classification", "evidence"):
        q = result["baseline"][task]["1"]["quality"]
        out.append(f"| {task} | {q['accuracy']*100:.2f} | " + " | ".join(f"{q[k]*100:.2f}" if k in q else "N/A" for k in ("precision", "recall", "f1")) + " |")
    out += ["", f"核对{data['preserved_files']}个历史文件和{data['frozen_files']}个冻结源码不变；"
            "逐行复算ID/input/request hash、一次调用、原生解析、完整token数、在途区间、块间无重叠、"
            "延迟分位数、吞吐、CPU累计值、镜像/命令/实际参数及原容器恢复证据。", "",
            "完整JSON保留每格两轮、全部尝试累计耗时、输出概率漂移、样本质量、原始metrics增量与文件SHA256。"
            "累计请求耗时不能替代并发block wall。GPU采样不是精确峰值；不能用配置并发数推定真实批大小。", "",
            "- [完整审计JSON](2026-10-19-clef-runtime-results.json)",
            "- [执行记录与外部服务核验](2026-10-19-clef-runtime-execution.md)",
            "- [冻结实验方案](../plans/2026-10-19-clef-runtime-parameters.md)", ""]
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(render(json.loads(args.input.read_text())))
