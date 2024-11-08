# type: ignore
# pylint: disable=unused-argument,missing-function-docstring
from kfp import dsl

from ..consts import PYTHON_IMAGE


@dsl.component(base_image=PYTHON_IMAGE, use_venv=True)
def kubectl_apply_op(manifest: str):
    return


@dsl.component(base_image=PYTHON_IMAGE, use_venv=True)
def kubectl_wait_for_op(condition: str, kind: str, name: str):
    return


@dsl.component(base_image=PYTHON_IMAGE, use_venv=True)
def huggingface_importer_op(repo_name: str, model_path: str = "/model"):
    return


@dsl.component(base_image=PYTHON_IMAGE, use_venv=True)
def pvc_to_mt_bench_op(mt_bench_output: dsl.Output[dsl.Artifact], pvc_path: str):
    return


@dsl.component(base_image=PYTHON_IMAGE, use_venv=True)
def pvc_to_model_op(model: dsl.Output[dsl.Model], pvc_path: str):
    return
