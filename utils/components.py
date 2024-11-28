# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member,missing-function-docstring

from kfp import dsl

from .consts import PYTHON_IMAGE, RHELAI_IMAGE, TOOLBOX_IMAGE


@dsl.container_component
def pvc_to_mt_bench_op(mt_bench_output: dsl.Output[dsl.Artifact], pvc_path: str):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {mt_bench_output.path}"],
    )


@dsl.container_component
def pvc_to_model_op(model: dsl.Output[dsl.Model], pvc_path: str):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {model.path}"],
    )


@dsl.container_component
def model_to_pvc_op(model: dsl.Input[dsl.Model], pvc_path: str = "/data/model"):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"mkdir -p {pvc_path} && cp -r {model.path}/* {pvc_path}"],
    )


@dsl.container_component
def ilab_importer_op(repository: str, release: str, base_model: dsl.Output[dsl.Model]):
    return dsl.ContainerSpec(
        RHELAI_IMAGE,
        ["/bin/sh", "-c"],
        [
            f"ilab --config=DEFAULT model download --repository {repository} --release {release} --model-dir {base_model.path}"
        ],
    )


@dsl.component(
    base_image=PYTHON_IMAGE,
    install_kfp_package=False,
)
def create_volume_snapshot_op(
    snapshot_name: str, volume_name: str, volume_snapshot_class: str
):
    import kubernetes
    import kubernetes.client

    try:
        kubernetes.config.load_kube_config()
        print("Loaded kube config")
    except kubernetes.config.ConfigException:
        print("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    api = kubernetes.client.CoreV1Api()

    body = {
        "apiVersion": "snapshot.storage.k8s.io/v1beta1",
        "kind": "VolumeSnapshot",
        "metadata": {"name": snapshot_name},
        "spec": {
            "volumeSnapshotClassName": volume_snapshot_class,
            "source": {"persistentVolumeClaimName": volume_name},
        },
    }

    try:
        api.create_namespaced_custom_object("default", body=body)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            print("Snapshot already exists")
        else:
            raise


# Remove me once https://github.com/kubeflow/pipelines/issues/11420 merged
@dsl.component(
    base_image=PYTHON_IMAGE,
    install_kfp_package=False,
)
def create_pvc_from_snapshot_op(
    volume_name: str, snapshot_name: str, access_modes: list, storage: str
) -> str:
    import uuid

    import kubernetes
    import kubernetes.client

    try:
        kubernetes.config.load_kube_config()
        print("Loaded kube config")
    except kubernetes.config.ConfigException:
        print("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    api = kubernetes.client.CoreV1Api()
    volume_full_name = volume_name + uuid.uuid4().hex[:8]

    body = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": volume_full_name},
        "spec": {
            "dataSource": {
                "name": snapshot_name,
                "kind": "VolumeSnapshot",
                "apiGroup": "snapshot.storage.k8s.io",
            },
            "accessModes": access_modes,
            "resources": {"requests": {"storage": storage}},
        },
    }

    try:
        # create the resource
        api.create_namespaced_persistent_volume_claim(
            namespace="admin",
            body=body,
        )
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            print("Snapshot already exists")
        else:
            raise

    return volume_full_name


@dsl.component(base_image=PYTHON_IMAGE, install_kfp_package=False)
def list_phase1_final_model_op() -> str:
    import os

    model_dir = "/output/phase_1/model/hf_format"
    models = os.listdir(model_dir)
    newest_idx = max(
        (os.path.getmtime(f"{model_dir}/{model}"), i) for i, model in enumerate(models)
    )[-1]
    newest_model = models[newest_idx]
    return f"{model_dir}/{newest_model}"
