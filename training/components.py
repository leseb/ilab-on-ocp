# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring

from typing import NamedTuple, Optional

from kfp import dsl

from utils.consts import PYTHON_IMAGE, RHELAI_IMAGE, TOOLBOX_IMAGE


@dsl.component(base_image=RHELAI_IMAGE, install_kfp_package=False)
def data_processing_op(
    model_path: str = "/model",
    sdg_path: str = "/data/sdg",
    skills_path: str = "/data/skills",
    knowledge_path: str = "/data/knowledge",
    max_seq_len: Optional[int] = 4096,
    max_batch_len: Optional[int] = 20000,
):
    import os

    import instructlab.training.data_process as dp
    from instructlab.training import (
        DataProcessArgs,
        TrainingArgs,
    )

    # define training-specific arguments
    skill_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model_path,
        data_path=f"{sdg_path}/skills_train_msgs*.jsonl",
        data_output_dir=skills_path,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    knowledge_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model_path,
        data_path=f"{sdg_path}/knowledge_train_msgs*.jsonl",
        data_output_dir=knowledge_path,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    def data_processing(train_args: TrainingArgs) -> None:
        # early validation logic here
        if train_args.max_batch_len < train_args.max_seq_len:
            raise ValueError(
                f"the 'max_batch_len' cannot be less than 'max_seq_len': {train_args.max_batch_len=} < {train_args.max_seq_len=}"
            )

            # process the training data
        if not os.path.exists(train_args.data_output_dir):
            os.makedirs(train_args.data_output_dir, exist_ok=True)
        dp.main(
            DataProcessArgs(
                # XXX(osilkin): make a decision here, either:
                #   1. the CLI is fully responsible for managing where the data is written
                #   2. we never cache it and simply write it to a tmp file every time.
                #
                # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
                # where the user has a defined place for new temporary data to be written.
                data_output_path=train_args.data_output_dir,
                model_path=train_args.model_path,
                data_path=train_args.data_path,
                max_seq_len=train_args.max_seq_len,
                chat_tmpl_path=train_args.chat_tmpl_path,
            )
        )

    data_processing(train_args=skill_training_args)
    data_processing(train_args=knowledge_training_args)


@dsl.container_component
def skills_processed_data_to_artifact_op(
    skills_processed_data: dsl.Output[dsl.Dataset],
    pvc_path: str = "/data/skills",
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {skills_processed_data.path}"],
    )


@dsl.container_component
def knowledge_processed_data_to_artifact_op(
    knowledge_processed_data: dsl.Output[dsl.Dataset],
    pvc_path: str = "/data/knowledge",
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {knowledge_processed_data.path}"],
    )


@dsl.component(base_image=PYTHON_IMAGE, install_kfp_package=False)
def pytorchjob_manifest_op(
    input_pvc_name: str,
    output_pvc_name: str,
    phase_num: int,
    path_to_model: str = None,
    nproc_per_node: int = 3,
    nnodes: int = 2,
    num_epochs: int = 2,
    effective_batch_size: int = 3840,
    learning_rate: float = 1e-4,
    num_warmup_steps: int = 800,
    save_samples: int = 0,
    max_batch_len: int = 20000,
    seed: int = 42,
):
    import inspect
    import time

    import kubernetes
    import urllib3
    import yaml

    name = f"train-phase-{phase_num}"

    if phase_num == 1:
        path_to_model = "/data/model"
        path_to_data = "/data/knowledge/data.jsonl"
    elif phase_num == 2:
        # If None, this means the caller forgot to pass the path_to_model
        # We need path_to_data to point to the epoch7 directory
        if path_to_model is None:
            raise RuntimeError("path_to_model is required for phase 2")
        path_to_data = "/data/skills/data.jsonl"
    else:
        raise RuntimeError(f"Unsupported value of {phase_num=}")

    image = "quay.io/redhat-et/ilab:1.2"

    manifest = inspect.cleandoc(
        f"""
        apiVersion: kubeflow.org/v1
        kind: PyTorchJob
        metadata:
          name: {name}
        spec:
          nprocPerNode: \"{nproc_per_node}\"
          pytorchReplicaSpecs:
            Master:
              replicas: 1
              restartPolicy: OnFailure
              template:
                metadata:
                  annotations:
                    sidecar.istio.io/inject: 'false'
                spec:
                  containers:
                    - args:
                        - |
                          echo "Running phase {phase_num}"
                          echo "Using {path_to_model} model for training"
                          echo "Using {path_to_data} data for training"
                          mkdir -p /output/phase_{phase_num}/model;
                          mkdir -p /output/data;
                          torchrun --nnodes {nnodes} \
                              --nproc_per_node {nproc_per_node} \
                              --node_rank \$(RANK) \
                              --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                              -m instructlab.training.main_ds \
                              --model_name_or_path={path_to_model} \
                              --data_path={path_to_data} \
                              --output_dir=/output/phase_{phase_num}/model \
                              --num_epochs={num_epochs} \
                              --effective_batch_size={effective_batch_size} \
                              --learning_rate={learning_rate} \
                              --num_warmup_steps={num_warmup_steps} \
                              --save_samples={save_samples} \
                              --log_level=INFO \
                              --max_batch_len={max_batch_len} \
                              --seed={seed} \
                              --cpu_offload_optimizer \
                              --cpu_offload_params \
                              --distributed_training_framework fsdp \
                              --is_granite \
                              --checkpoint_at_epoch
                      command:
                        - /bin/bash
                        - '-c'
                        - '--'
                      image: {image}
                      name: pytorch
                      volumeMounts:
                        - mountPath: /data
                          name: input-data
                          readOnly: true
                        - mountPath: /output
                          name: output
                      env:
                        - name: NNODES
                          value: \"{nnodes}\"
                        - name: NPROC_PER_NODE
                          value: \"{nproc_per_node}\"
                        - name: XDG_CACHE_HOME
                          value: /tmp
                        - name: TRITON_CACHE_DIR
                          value: /tmp
                        - name: HF_HOME
                          value: /tmp
                        - name: TRANSFORMERS_CACHE
                          value: /tmp
                      resources:
                        requests:
                          cpu: 8
                          "nvidia.com/gpu": {nproc_per_node}
                        limits:
                          cpu: 8
                          "nvidia.com/gpu": {nproc_per_node}
                  volumes:
                    - name: input-data
                      persistentVolumeClaim:
                        claimName: {input_pvc_name}
                    - name: output
                      persistentVolumeClaim:
                        claimName: {output_pvc_name}
            Worker:
              replicas: {nnodes-1}
              restartPolicy: OnFailure
              template:
                metadata:
                  annotations:
                    sidecar.istio.io/inject: 'false'
                spec:
                  containers:
                    - args:
                        - |
                          echo "Running phase {phase_num}"
                          echo "Using {path_to_model} model for training"
                          echo "Using {path_to_data} data for training"
                          mkdir -p /tmp/model;
                          torchrun --nnodes {nnodes} \
                            --nproc_per_node {nproc_per_node} \
                            --node_rank \$(RANK) \
                            --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                            -m instructlab.training.main_ds \
                            --model_name_or_path={path_to_model} \
                            --data_path={path_to_data} \
                            --output_dir=/tmp/model \
                            --num_epochs={num_epochs} \
                            --effective_batch_size={effective_batch_size} \
                            --learning_rate={learning_rate} \
                            --num_warmup_steps={num_warmup_steps} \
                            --save_samples={save_samples} \
                            --log_level=INFO \
                            --max_batch_len={max_batch_len} \
                            --seed={seed} \
                            --cpu_offload_optimizer \
                            --cpu_offload_params \
                            --distributed_training_framework fsdp \
                            --is_granite \
                            --checkpoint_at_epoch
                      command:
                        - /bin/bash
                        - '-c'
                        - '--'
                      image: {image}
                      name: pytorch
                      volumeMounts:
                        - mountPath: /data
                          name: input-data
                          readOnly: true
                      env:
                        - name: NNODES
                          value: \"{nnodes}\"
                        - name: NPROC_PER_NODE
                          value: \"{nproc_per_node}\"
                        - name: XDG_CACHE_HOME
                          value: /tmp
                        - name: TRITON_CACHE_DIR
                          value: /tmp
                        - name: HF_HOME
                          value: /tmp
                        - name: TRANSFORMERS_CACHE
                          value: /tmp
                      resources:
                        requests:
                          cpu: 8
                          "nvidia.com/gpu": {nproc_per_node}
                        limits:
                          cpu: 8
                          "nvidia.com/gpu": {nproc_per_node}
                  volumes:
                    - name: input-data
                      persistentVolumeClaim:
                        claimName: {input_pvc_name}

        """
    )

    try:
        manifest_yaml = yaml.safe_load(manifest)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error parsing manifest: {exc}") from exc

    # Discover the namespace in which the pod is running
    with open(
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r", encoding="utf-8"
    ) as f:
        namespace = f.read().strip()
        print(f"The pod is running in the namespace: {namespace}")

    try:
        kubernetes.config.load_kube_config()
        print("Loaded kube config")
    except kubernetes.config.ConfigException:
        print("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    api = kubernetes.client.CustomObjectsApi()
    try:
        api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=namespace,
            plural="pytorchjobs",
            body=manifest_yaml,
        )
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            print(
                "{} '{}/{}' already exists.".format(
                    manifest_yaml["kind"],
                    namespace,
                    manifest_yaml["metadata"]["name"],
                )
            )
        else:
            raise

    # Get the CR status and wait for it to be completed
    w = kubernetes.watch.Watch()
    exit_flag = False
    start_time = time.time()
    timeout_seconds = 24 * 60 * 60  # 24 hours

    while not exit_flag:  # Keep the watch active
        if time.time() - start_time > timeout_seconds:
            raise RuntimeError(
                "Timeout (24h) reached waiting for the PytorchJob to complete."
            )

        try:
            print("Watching for PytorchJob")
            for event in w.stream(
                api.list_namespaced_custom_object,
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
                timeout_seconds=60,  # Timeout after 1 minute
            ):
                pytorchjob_event = event["object"]
                if (
                    pytorchjob_event["metadata"]["name"]
                    != manifest_yaml["metadata"]["name"]
                ):
                    continue
                pytorchjob_name = pytorchjob_event["metadata"]["name"]

                if (
                    "status" not in pytorchjob_event
                    or "conditions" not in pytorchjob_event["status"]
                ):
                    continue
                print(
                    f"PytorchJob: {pytorchjob_name} - {pytorchjob_event['status'].get('conditions', 'No conditions yet')}"
                )
                for job_condition in reversed(pytorchjob_event["status"]["conditions"]):
                    if job_condition["type"] == "Succeeded":
                        print(
                            f"PytorchJob '{pytorchjob_name}' completed successfully: {job_condition['reason']}"
                        )
                        print(f"Training phase {phase_num} completed.")
                        w.stop()
                        exit_flag = True
                        # Break here to avoid going into other conditions, we are done
                        break
                    elif job_condition["type"] == "Failed":
                        print(
                            f"PytorchJob '{pytorchjob_name}' failed: {job_condition['reason']}"
                        )
                        w.stop()
                        raise RuntimeError("Job failed.")
        except kubernetes.client.exceptions.ApiException as e:
            print(f"API exception occurred: {str(e)}")
            time.sleep(5)  # Backoff before retrying
        # Catches the following error:
        # urllib3.exceptions.ProtocolError: ("Connection broken: InvalidChunkLength
        except urllib3.exceptions.ProtocolError as e:
            print(f"Connection broken reconnecting the watcher {str(e)}")
            time.sleep(5)  # Backoff before retrying
        finally:
            w.stop()
