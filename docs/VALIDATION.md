# 归档验证记录

验证日期：2026-10-20（上海时间）。本次只整理和核对已有材料，没有重新运行模型、部署容器或修改历史得分。

## 已完成

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 文件完整性 | 5,070 条来源文件记录、173 个归档；已发布文件及归档成员的 SHA256 全部匹配 | [逐条核对结果](../manifests/validation/verification.json) |
| 分类结果重算 | 16 个配置、47,584 条正式记录；Accuracy、macro-F1、平均耗时和 P99 与原汇总一致 | 同上 |
| 片段筛选结果重算 | 15 个模型、56 个条件；195,580 次调用、14,952 条完整筛选记录；正式条件的混淆计数、P/R/F1 点估计及平均/P99 耗时一致 | 同上 |
| 调用记录检查 | 各阶段调用数、成功状态、唯一调用标识、标签对应及 195,580 个请求的全局串行时间区间通过 | 同上 |
| 离线回归测试 | 282 项通过，0 失败；包括原评测/部署测试 266 项及归档工具检查 16 项；网络连接被禁用 | [测试明细](../manifests/validation/offline-tests.json) |
| 历史输入重建 | 20 份输入文件逐字节恢复，全部与历史 SHA256 相同；包括分类、片段筛选及相关补充实验 | [重建明细](../manifests/validation/reconstruction.json) |
| 数据下载入口 | 实际下载固定版本 MASSIVE test 文件，201,108 字节，SHA256 匹配；未为此次整理重复下载完整大语料 | [辅助检查](../manifests/validation/packaging.json) |
| 解包与报告生成 | 173 个归档可解包到独立目录；主片段 HTML 重新生成后，与发布版内容一致（仅归档 JSON 下载链接后缀不同） | 同上 |
| 公开内容检查 | 已扫描归档内容、正文及压缩 JSON；未检出高置信凭据特征、非空敏感 JSON 字段或需脱敏的主机/用户路径；入口文档及三个 HTML 的本地链接存在 | 同上 |

5,070 条来源记录中，4,649 条按收集时字节发布，288 条经过命名整理、脱敏或下载链接调整，133 条仅保留哈希与未发布原因。
其中 `preflight-20261015/vela-source/LICENSE` 是本次从官方项目补入的许可文本，不是历史执行文件；清单中单独标记为 `publication_supplement`。

Git 空白检查指出 3 个导入文件原有的 8 处行尾空白或末尾空行（两个 HTML 和 reranker 模板）。为保留原模板与报告字节，未做格式清理；本次新增的文档和工具没有空白检查问题。

名称整理覆盖仓库文件与压缩包内部，不只修改顶层目录。两个主场景的 47 份原始响应/请求/筛选记录文件逐字节未变。主审计 JSON 只修改了 `dataset.kind` 显示名称，所有数字和指标未变；HTML 生成器校验的是发布副本的 SHA256，原始 SHA256 在发布清单中保留。
另外对 309 个排序输入及两条随机数流各 100 次抽样做了修改前后对照，计算结果一致；全文件及解压内容扫描未发现原命名残留。见[命名整理复核](../manifests/validation/namespace.json)。

## 复核命令

先按照根目录 README 安装工具依赖，再执行：

```sh
.venv/bin/python tools/verify.py --output verification-new.json
.venv/bin/python tools/test_offline.py -q
.venv/bin/python tools/materialize.py --output work
.venv/bin/python work/docs/research/2026-10-19-answer-evidence-summary.py --replace
.venv/bin/python tools/reconstruct_inputs.py \
  --sources-dir downloads --output work/evidence --report reconstruction-new.json
```

`work`、新验证 JSON 必须使用尚不存在的路径；`downloads` 需先按[数据集说明](../datasets/README.md)准备固定来源文件。
新工具的依赖版本见根目录 `requirements.txt`，不是对全部历史依赖版本完全相同的声明。

## 未验证的范围

- 没有重新计算 bootstrap 置信区间。报告里的区间来自保留的原审计，当前重算检查的是质量和耗时点估计。
- 没有再次执行原主机 systemd、GPU 或代理身份的现场审计。旧审计的严格失败、修订规则和通过记录都保留，不能以此次归档核对覆盖它们。
- 没有在新 GPU 主机重建镜像、重新下载权重或重跑并发测试。5 项不可变权重版本缺口见[部署说明](DEPLOYMENT.md)。
- 凭据扫描是发布检查，不是对任意历史自由文本的安全证明。运行日志、私有环境及生成的鉴权配置不公开；原文件哈希和省略原因可查。
- 没有增加人工标签或新的事实正确性判定，所有主结果仍以原数据集标签为准。
