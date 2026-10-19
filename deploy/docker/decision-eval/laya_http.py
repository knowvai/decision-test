"""Official Laya HTTP app with the existing evaluation no-truncation guard."""

import os
import warnings

from server import validate_laya_input


def guard_input(context):
    agent = context.agent
    state_limit = context.max_len if context.max_len is not None else agent.cfg["max_len"]
    head_limit = context.head_max_len if context.head_max_len is not None else agent.cfg["head_max_len"]
    for state in context.states:
        validate_laya_input(agent, {"state": state, "questions": context.questions},
                            state_limit, head_max_length=head_limit)


def create_app():
    import laya
    import torch
    from laya.router import Router
    from laya.serve import create_app as official_app

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not an evaluation path")
    torch.set_num_threads(2)
    warnings.filterwarnings("error", message="Warning: GPU memory exceeded.*", category=RuntimeWarning)
    name, path = os.environ["MODEL_NAME"], os.environ["MODEL_PATH"]
    agent = laya.load(path, device="cuda", fast=False, compile=False,
                      on_predict_start=guard_input, hooks_raise=True)
    router = Router(models={name: path}, device="cuda", default=name, max_loaded=1)
    router.models = {router.resolve(name): path}
    router.attach(name, agent)
    return official_app(router)
