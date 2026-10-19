"""Protect the evaluation credential when rendering the native proxy config."""

from pathlib import Path
import json
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / "decision-eval"
sys.path.insert(0, str(SOURCE))
from launch_reviewed import write_proxy_config
import launch_reviewed as launcher


class ProxyConfigTests(unittest.TestCase):
    def test_shared_resources_require_exact_identity_explicit_limits_and_reranker(self):
        valid = dict(cohost_container_id="a" * 64, gpu_memory_utilization=0.6,
                     memory_gib=32, cpus=4)
        self.assertEqual(launcher.resource_settings(SimpleNamespace(), "worker"),
                         {"cohost": None, "gpu": "0.90", "memory_gib": 96, "cpus": 8})
        for changed in ({"cohost_container_id": "embedding-service"}, {"memory_gib": None},
                        {"gpu_memory_utilization": float("nan")}, {"gpu_memory_utilization": 1},
                        {"memory_gib": True}, {"cpus": -1}, {"cpus": float("inf")}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                launcher.resource_settings(SimpleNamespace(**{**valid, **changed}), "reranker")
        with self.assertRaises(ValueError):
            launcher.resource_settings(SimpleNamespace(**valid), "llm")

    def test_cohost_guard_rejects_unknown_processes_stops_and_identity_changes(self):
        cid = "a" * 64
        identity = dict(id=cid, image_id="sha256:" + "b" * 64, running=True,
                        pid=41, started_at="2026-10-17T00:00:00Z", owned=None)
        for active, after, passes in (("41\n42", identity, True),
                                      ("41\n99", identity, False),
                                      ("41", {**identity, "pid": 45}, False),
                                      ("41", {**identity, "running": False}, False)):
            calls = iter([identity, after])
            def docker(*parts):
                return "PID\n41\n42" if parts[0] == "top" else json.dumps(next(calls))
            with self.subTest(active=active, after=after), \
                 patch.object(launcher, "docker", side_effect=docker), \
                 patch.object(launcher.subprocess, "check_output", return_value=active):
                if passes:
                    self.assertEqual(launcher.check_gpu_processes(cid)["gpu_pids"], [41, 42])
                else:
                    with self.assertRaises(RuntimeError):
                        launcher.check_gpu_processes(cid)
        with patch.object(launcher.subprocess, "check_output", return_value="41"), \
             self.assertRaises(RuntimeError):
            launcher.check_gpu_processes()
        with patch.object(launcher, "docker", return_value=json.dumps({**identity, "owned": "true"})), \
             self.assertRaises(RuntimeError):
            launcher.cohost_identity(cid)

    def test_shared_reranker_records_limits_without_mutating_cohost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = "qwen3-reranker-8b"
            (root / launcher.MODELS[model][1]).mkdir(parents=True)
            args = SimpleNamespace(root=root, model=model, env_file=root / ".env",
                native_port=30888, public_port=30889, seqs=16, tokens=16384, enforce_eager=False,
                cohost_container_id="a" * 64, gpu_memory_utilization=0.6, memory_gib=32, cpus=4)
            identity = dict(id=args.cohost_container_id, image_id="sha256:" + "b" * 64,
                            running=True, pid=41, started_at="2026-10-17T00:00:00Z", owned=None)
            def docker(*parts):
                if parts[0] == "inspect":
                    return json.dumps(identity)
                if parts[0] == "top":
                    return "PID\n41"
                return "" if parts[-1] == "{{.Config.User}}" else "sha256:" + "c" * 64
            with patch.object(launcher, "docker", side_effect=docker) as mocked, \
                 patch.object(launcher, "remove_owned") as remove, \
                 patch.object(launcher, "read_key", return_value="A1" * 16), \
                 patch.object(launcher.socket, "socket"), \
                 patch.object(launcher.subprocess, "check_output", return_value="41"):
                launcher.launch(args)
            self.assertNotIn(args.cohost_container_id, [call.args[0] for call in remove.call_args_list])
            record = json.loads((root / "acceptance-20261015/evidence" / (model + ".deployment.json")).read_text())
            self.assertEqual(record["cohost"]["id"], args.cohost_container_id)
            self.assertEqual(record["cpu_budget"], 4)
            self.assertEqual(record["host_memory_limit_gib"], 32)
            self.assertIn("qwen3_reranker.jinja", record["source_sha256"])
            command = record["command"]
            self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.6")
            self.assertEqual(command[command.index("--runner") + 1], "pooling")
            launch = next(call.args for call in mocked.call_args_list if call.args[:4] ==
                          ("run", "-d", "--name", "decision-acceptance-model"))
            self.assertEqual(launch[launch.index("--cpus") + 1], "4")
            self.assertEqual(launch[launch.index("--memory") + 1], "32g")
            self.assertEqual(launch[launch.index("--gpus") + 1], "device=0")
            with patch.object(launcher, "remove_owned") as remove, \
                 patch.object(launcher, "read_key", return_value="A1" * 16), \
                 patch.object(launcher, "check_gpu_processes", side_effect=RuntimeError("unknown process")), \
                 self.assertRaises(RuntimeError):
                launcher.launch(args)
            remove.assert_not_called()

    def test_laya_admission_override_is_opt_in_and_not_gpu_batching(self):
        for admission in (None, 32):
            with self.subTest(admission=admission), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / launcher.MODELS["laya-multilingual"][1]).mkdir(parents=True)
                args = SimpleNamespace(root=root, model="laya-multilingual", env_file=root / ".env",
                    native_port=30888, public_port=30889, seqs=32, tokens=32768,
                    enforce_eager=False, laya_max_concurrent=admission)

                def docker(*parts):
                    return "" if parts[-1] == "{{.Config.User}}" else "sha256:" + "a" * 64

                with patch.object(launcher, "docker", side_effect=docker), \
                     patch.object(launcher, "remove_owned"), \
                     patch.object(launcher, "read_key", return_value="A1" * 16), \
                     patch.object(launcher.socket, "socket"), \
                     patch.object(launcher.subprocess, "check_output", return_value=""):
                    launcher.launch(args)
                record = json.loads((root / "acceptance-20261015/evidence/laya-multilingual.deployment.json").read_text())
                if admission is None:
                    self.assertNotIn("LAYA_MAX_CONCURRENT", record["env"])
                    self.assertNotIn("http_admission_limit", record["scheduling"])
                else:
                    self.assertEqual(record["env"]["LAYA_MAX_CONCURRENT"], "32")
                    self.assertEqual(record["scheduling"]["http_admission_limit"], 32)
                    self.assertEqual(record["scheduling"]["single_request_executor_workers"], 1)
                    self.assertFalse(record["scheduling"]["cross_request_batching"])
                self.assertNotIn("LAYA_API_KEY", record["env"])

    def test_invalid_admission_is_rejected_before_container_mutation(self):
        for model, admission in (("laya-multilingual", 0), ("laya-multilingual", 257),
                                  ("laya-multilingual", True), ("d3-edge", 32)):
            with self.subTest(model=model, admission=admission), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(root=Path(directory), model=model, laya_max_concurrent=admission)
                with patch.object(launcher, "remove_owned") as remove, self.assertRaises(ValueError):
                    launcher.launch(args)
                remove.assert_not_called()

    def test_key_match_is_case_sensitive_and_config_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            key = "A1" * 16
            path = write_proxy_config(SOURCE, Path(directory), key, 30888, 30889)
            self.assertIn('"~^Bearer ' + key + '$" 1;', path.read_text())
            self.assertNotIn("${", path.read_text())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_rejects_regex_or_config_metacharacters_in_key(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in ("A1" * 16 + ".*", "A1" * 16 + '"', "A1" * 16 + "\n"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    write_proxy_config(SOURCE, Path(directory), key, 30888, 30889)
            self.assertFalse((Path(directory) / ".nginx.conf").exists())

    def test_pageable_lora_profile_is_jev_only(self):
        for model, image in (("jev-27b", launcher.JEV_VLLM),
                             ("qwen3-30b-a3b-instruct-2507", launcher.VLLM),
                             ("qwen3.8-27b", launcher.VLLM),
                             ("qwen3.8-27b-thinking", launcher.VLLM),
                             ("pplx-decider-v1.1-27b", launcher.SGLANG)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / launcher.MODELS[model][1]).mkdir(parents=True)
                args = SimpleNamespace(root=root, model=model, env_file=root / ".env",
                                       native_port=30888, public_port=30889, seqs=16,
                                       tokens=16384, enforce_eager=False)

                def docker(*parts):
                    if parts[:2] == ("image", "inspect"):
                        return "" if parts[-1] == "{{.Config.User}}" else "sha256:" + "a" * 64
                    return "test-container"

                with patch.object(launcher, "docker", side_effect=docker) as mocked, \
                     patch.object(launcher, "remove_owned"), \
                     patch.object(launcher, "read_key", return_value="A1" * 16), \
                     patch.object(launcher.socket, "socket"), \
                     patch.object(launcher.subprocess, "check_output", return_value=""):
                    launcher.launch(args)
                mocked.assert_any_call("image", "inspect", image, "--format", "{{.Id}}")
                record = json.loads((root / "acceptance-20261015/evidence" / (model + ".deployment.json")).read_text())
                self.assertEqual(record["tp"], 2)
                self.assertNotIn("DECISION_EVAL_API_KEY", record["env"])
                self.assertNotIn("VLLM_API_KEY", record["env"])
                if model in ("qwen3.8-27b", "qwen3.8-27b-thinking"):
                    self.assertIn("--language-model-only", record["command"])
                    flag = record["command"].index("--default-chat-template-kwargs")
                    template = json.loads(record["command"][flag + 1])
                    if model.endswith("-thinking"):
                        self.assertEqual(template, {"enable_thinking": True, "reasoning_effort": "xhigh"})
                        self.assertEqual(record["classification_profile"], launcher.CLASSIFICATION_PROFILES[model])
                        self.assertEqual(launcher.MODELS[model], launcher.MODELS["qwen3.8-27b"])
                        self.assertEqual(record["command"][record["command"].index("--reasoning-parser") + 1], "qwen3")
                    else:
                        self.assertEqual(template, {"enable_thinking": False})
                        self.assertNotIn("--reasoning-parser", record["command"])
                        self.assertNotIn("classification_profile", record)
                    self.assertNotIn("--enforce-eager", record["command"])
                else:
                    self.assertNotIn("--default-chat-template-kwargs", record["command"])
                    self.assertNotIn("--language-model-only", record["command"])
                if model == "jev-27b":
                    self.assertEqual(record["env"]["VLLM_LORA_DISABLE_PIN_MEMORY"], "1")
                    self.assertEqual(record["env"]["VLLM_LORA_VERIFY_TRANSFER"], "1")
                    self.assertIn("patch_vllm_lora.py", record["source_sha256"])
                    self.assertIn("--enforce-eager", record["command"])
                else:
                    self.assertNotIn("VLLM_LORA_DISABLE_PIN_MEMORY", record["env"])
                    self.assertNotIn("VLLM_LORA_VERIFY_TRANSFER", record["env"])
                    self.assertNotIn("patch_vllm_lora.py", record["source_sha256"])


if __name__ == "__main__":
    unittest.main()
