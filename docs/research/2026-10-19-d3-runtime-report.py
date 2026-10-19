"""Render the independent scheduling audit without querying a model or service."""
import argparse
import json
from pathlib import Path


def render(data):
    assert data["passed"]
    out = ["# D3 Runtime 参数实测", "",
           "这轮专门验证 runtime 参数，不做权重、提示或阈值调优。固定原始 baseline 提示和原生 choice，"
           "与上一轮性能测试使用的开发集选定提示不同，不能将两轮绝对 RPS 直接相减。", "",
           f"独立离线审计通过：{data['total_calls']:,} 次 HTTP 推理调用，{data['blocks']} 个块，"
           f"其中预热 {data['calls']['warmup']:,} 次、实测 {data['calls']['performance']:,} 次，全部 valid/HTTP 200；"
           f"零失败、超时和未尝试。{data['preserved_files']:,} 个历史文件、{data['frozen_files']} 个冻结源码文件不变。", "",
           "## 测试口径", "",
           "- 两款 D3，固定 vllm-srun 0.2.0 / 原生计算、同一张 A100-PCIE-40GB、8 CPU quota、96 GiB 主机内存限制。",
           "- 每任务 128 个冻结输入，来自上一轮性能子集；两轮正反输入/并发顺序。不是新的独立质量样本。",
           "- 每档冷启动后独立预热；基线首尾各测一次，flash 反转七个候选部署的顺序。每个候选只有一次部署。",
           "- n 是客户端同时在途请求上限；max-num-seqs 是 GPU forward 行数上限，两者不是同一参数。",
           "- RPS=有效请求数/两轮整块 wall；mean/P95/P99 用合并请求计算，不平均百分位。包含 HTTP、鉴权与服务端排队，"
           "不含客户端等空位、装载、校准和预热。闭环短程负载，不是开放流量、长期 SLA 或生产容量。",
           "- 自动模式的启动校准和所有档位的原生启动检查有内部 GPU 工作，单独计入启动成本，不伪称仅有上述 HTTP 推理。",
           "- 不调用 reranker、不动代理机 embedding/reranker/proxy；没有新增 GPU/CPU、缓存或降低精度。", "",
           "## 参数与实际解析", "",
           "所有档位 threads=2；baseline 为 seq32/tokens32768/window2ms。auto 省略 seq/token 标志。", "",
           "| 模型 | 档位 | 实际 seq 上限 | 实际 token 上限 | 窗口 ms | 启动至就绪 s | 来源 |",
           "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for model, result in data["models"].items():
        for profile, deploy in result["deployments"].items():
            out.append(f"| {model} | {profile} | {deploy['resolved_seqs']} | {deploy['resolved_tokens']} | "
                       f"{data['profiles'][profile]['batch_window_ms']} | {deploy['loading_seconds']:.1f} | {deploy['source']} |")
    out += ["", "这些是启动后的外层预算，Decision3 内部还会自适应约束 coalesced batch；"
            "本轮没有观测到内部实时预算，不能仅凭设置值宣称某个限制已经消除。auto 的 seq 上限若高于 32，"
            "超出客户端 n32 的部分没有被本次测试覆盖。", "", "## 预注册候选规则", "",
            "同一任务、同一 n，两轮吞吐均至少提高 5%，合并 P95 不超过基线 110%，零失败、零决策变化，"
            "且首尾基线吞吐漂移不超过 10%，才标为“候选”。这只是待复验候选，不是生产推荐。", "",
            "| 模型 | 任务 | 档位 | n | 吞吐变化 % | P95 变化 % | 基线首尾漂移 % |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    found = False
    for model, result in data["models"].items():
        for profile, tasks in result["performance"].items():
            for task, levels in tasks.items():
                for n, v in levels.items():
                    if v["promising_candidate"]:
                        found = True
                        out.append(f"| {model} | {task} | {profile} | {n} | {(v['rps_ratio']-1)*100:+.1f} | "
                                   f"{(v['p95_ratio']-1)*100:+.1f} | {v['baseline_end_rps_drift']*100:+.1f} |")
    if not found:
        out.append("| 本轮没有满足全部规则的候选 | | | | | | |")
    out += ["", "## 全部性能结果", "",
            "每格 256 次尝试；保留全部两轮，不挑最快轮。变化数以 ID 配对，对照相同并发的 baseline；概率变化单独保留在 JSON。", ""]
    for task in ("classification", "evidence"):
        out += [f"### {task}", "", "| 模型 | 档位 | n | RPS | 两轮 RPS | Mean ms | P95 ms | P99 ms | 吞吐比 | 决策变化 /256 |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
        for model, result in data["models"].items():
            for profile, tasks in result["performance"].items():
                for n, v in tasks[task].items():
                    rounds = " / ".join(f"{x['rps']:.2f}" for x in v["rounds"])
                    out.append(f"| {model} | {profile} | {n} | {v['rps']:.2f} | {rounds} | {v['mean_ms']:.2f} | "
                               f"{v['p95_ms']:.2f} | {v['p99_ms']:.2f} | {v['rps_ratio']:.3f} | {v['changes_vs_baseline']} |")
        out.append("")
    out += ["## 合批与资源观测", "",
            "每批条数、forward 和排队取 Prometheus 实际计数器增量；CPU 为 cgroup 累计使用时间/HTTP block wall，"
            "GPU 为每秒采样均值。采样不是精确峰值；各阶段耗时不能加起来冒充端到端 wall。", "",
            "| 模型 | 档位 | 任务 | n | 条/批 | forward ms/批 | 排队 ms/请求 | CPU 核均值 | GPU 利用率均值 % | 显存最大采样 MiB |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, result in data["models"].items():
        for profile, tasks in result["performance"].items():
            for task, levels in tasks.items():
                for n, v in levels.items():
                    gpu = f"{v['gpu_util_mean']:.1f}" if v["gpu_util_mean"] is not None else "N/A"
                    out.append(f"| {model} | {profile} | {task} | {n} | {v['batch_rows_mean']:.2f} | "
                               f"{v['forward_ms_mean']:.2f} | {v['queue_ms_mean']:.2f} | {v['cpu_cores_mean']:.2f} | {gpu} | "
                               f"{v['gpu_memory_mib_max_sampled']} |")
    out += ["", "## 质量稳定性", "",
            "下面是固定性能子集的基线结果，只用于 runtime 回归。证据是候选级 P/R/F1，"
            "不是完整池指标；重复性能调用不扩充独立样本量，也不证明模型泛化准确率提高。", "",
            "| 模型 | 分类 Accuracy % | 证据 Accuracy % | 证据 P % | 证据 R % | 证据 F1 % | 全档位对基线变化次数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, result in data["models"].items():
        c = result["performance"]["baseline"]["classification"]["1"]["quality"]
        e = result["performance"]["baseline"]["evidence"]["1"]["quality"]
        changes = sum(v["changes_vs_baseline"] for tasks in result["performance"].values() for levels in tasks.values() for v in levels.values())
        out.append(f"| {model} | {c['accuracy']*100:.2f} | {e['accuracy']*100:.2f} | {e['precision']*100:.2f} | "
                   f"{e['recall']*100:.2f} | {e['f1']*100:.2f} | {changes} |")
    out += ["", "运行时参数主要优化调度，并不保证改善准确率。上一轮独立留出的提示/阈值结果请单独阅读，"
            "不将本轮样本 Accuracy 与旧全量测试排名混排。T2Ranking 的 L2/L3 仍只是答案证据代理标签。", "",
            "## 审计与范围", "",
            "逐行核验输入与请求哈希、一次调用、原生解码、GPU 元数据、并发在途区间及块间无重叠，重算 wall/百分位/指标，"
            "核验镜像、启动参数、实际解析预算、历史保护清单及原容器恢复。原始响应仅保存原生字段与原始字节 hash，"
            "不声称可以重建原始 HTTP 字节。systemd 终态与跨主机保护服务身份在执行记录中外部复核。", "",
            "没有重跑旧测试、没有生产上线、没有自动追加下一轮实验。若候选需要采用，还需更长负载、完整长输入回归及真实业务流量复验。", "",
            "- [本轮方案](../plans/2026-10-19-d3-runtime-parameters.md)",
            "- [本轮执行记录](2026-10-19-d3-runtime-execution.md)",
            "- [本轮完整审计 JSON](2026-10-19-d3-runtime-results.json)",
            "- [上一轮质量与并发试验](2026-10-19-d3-tuning-results.md)", ""]
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(render(json.loads(args.input.read_text())))
