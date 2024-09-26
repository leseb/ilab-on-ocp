#!/usr/bin/env python3

"""
Standalone Distributed training script

This script provides a standalone version of the pipeline.py script, designed to be used when
pipelines are not available. It includes a command-line interface (CLI) for setting up and running
SDG (Sustainable Development Goals) data preprocessing tasks on a Kubernetes cluster.

Modules:
    click: A package for creating command-line interfaces. kubernetes: A package for interacting
    with Kubernetes clusters.

Functions:
    cli(): Command Line Interface (CLI) entry point. run(namespace): Execute the run command for
    setting up SDG. create_job_object(namespace, job_name, container_image): Create a Kubernetes Job
    object. run_job(namespace, job): Create and run a Kubernetes job in the specified namespace, and
    wait for its completion. create_pvc(namespace): Create a PersistentVolumeClaim (PVC) in the
    specified namespace. preprocess_sdg_data(namespace): Preprocesses SDG data by creating a
    Persistent Volume Claim (PVC) and initiating a job to run a pod for SDG data preprocessing.

Usage:
    This script can be executed directly from the command line. Ensure that the Kubernetes client is
    properly configured before running the script.

Dependencies:
    kubernetes: The Kubernetes Python client library. click: A package for creating command-line
    interfaces. kfp: The Kubeflow Pipelines SDK.

TODO:
    - Make sure ressources get cleaned up after the job is done. (configmap, secret etc) using a
      finalizer.
    - See if we can use KServe to deploy the model and serve it for SDG Data Generation.
      kubernetes_yaml/mixtral_serve/mixtral_serve.yaml
"""

import logging
import typing

import click
import kubernetes
import kubernetes.client
import kubernetes.config
import kubernetes.client.rest

logger = logging.getLogger(__name__)

SDG_PVC_NAME = "sdg-data"


@click.group()
def cli():
    """
    Command Line Interface (CLI) entry point.

    This function serves as the main entry point for the command line interface.
    It currently does not perform any operations.
    """


@click.command()
@click.option(
    "--namespace", type=str, default="default", help="Kubernetes namespace to use"
)
@click.option(
    "--taxonomy-repo-url",
    type=str,
    default="",
    help="URL of the taxonomy repository",
)
@click.option(
    "--taxonomy-repo-branch",
    type=typing.Optional[str] | None,
    default=None,
    help="Branch of the taxonomy repository",
)
@click.option(
    "--taxonomy-repo-pr",
    type=typing.Optional[int] | None,
    default=0,
    help="Pull request number of the taxonomy repository",
)
@click.option(
    "--storage-class",
    type=str,
    default="standard",
    help="Storage class to use for the PersistentVolumeClaim",
)
def run(
    namespace: typing.Optional[str] = None,
    taxonomy_repo_url: typing.Optional[str] = None,
    taxonomy_repo_branch: typing.Optional[str] = None,
    taxonomy_repo_pr: typing.Optional[str] = None,
    storage_class: typing.Optional[str] = "standard",
):
    """
    Execute the distributed training on Kubernetes.

    Args:
        namespace (str): The namespace to use for the setup process.
        taxonomy_repo_url (str): The URL of the taxonomy repository.
        taxonomy_repo_branch (str): The branch of the taxonomy repository.
        taxonomy_repo_pr (int): The pull request number of the taxonomy repository.
        storage_class (str): The storage class to use for the PersistentVolumeClaim.

    Returns:
        None
    """
    # Implement the functionality for the run command
    logger.info("Running setup for SDG")
    # Add your pipeline execution logic here
    preprocess_sdg_data(
        namespace,
        taxonomy_repo_url,
        taxonomy_repo_branch,
        taxonomy_repo_pr,
        storage_class,
    )


cli.add_command(run)


def create_sdg_job(
    namespace: str,
    job_name: str,
    exec_git_clone_op_repo_url: typing.Optional[str] = None,
    exec_git_clone_op_repo_branch: typing.Optional[str] = None,
    exec_git_clone_op_repo_pr: typing.Optional[str] = None,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run a specified
    container image with a predefined command. The job is configured to not restart
    on failure and has a backoff limit and TTL after completion.

    Steps:
        1. InitContainer to fetch the taxonomy data. - EmptyDir volume to share data between
           containers.
        2. InitContainer to generate synthetic data. - Stored on EmptyDir volume. (Option to push to
           S3?)
        3. Main container to pre-process the data before training. From the EmptyDir volume and copy
           the result to the PVC.
    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.
        exec_git_clone_op_repo_url (str): The URL of the taxonomy repository.
        exec_git_clone_op_repo_branch (str, optional): The branch of the taxonomy repository.
        exec_git_clone_op_repo_pr (str, optional): The pull request number of the taxonomy repository.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """
    # Configureate Pod template container
    init_containers = (
        [
            kubernetes.client.V1Container(
                name="sdg-op-fetch-taxonomy-data",
                image="registry.access.redhat.com/ubi9/toolbox",
                command=["/bin/sh", "-c"],
                args=['git clone {exec_git_clone_op_repo_url} /tmp && cd /tmp && if [ ! -z "{exec_git_clone_op_repo_branch}" ]; then git fetch origin {exec_git_clone_op_repo_branch} && git checkout {exec_git_clone_op_repo_branch}; elif [ ! -z "{exec_git_clone_op_repo_pr}" ]; then git fetch origin pull/{exec_git_clone_op_repo_pr}/head:{exec_git_clone_op_repo_pr} && git checkout {exec_git_clone_op_repo_pr}; fi '],
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name="shared-data", mount_path="/mnt/shared-data"
                    )
                ],
            ),
            kubernetes.client.V1Container(
                name="sdg-op-generate-synthetic-data",
                image="quay.io/tcoufal/ilab-sdg:latest",
                command=['sh', '-c', '\nif ! [ -x "$(command -v pip)" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location \'kfp==2.6.0\' \'--no-deps\' \'typing-extensions>=3.7.4,<5; python_version<"3.9"\' && "$0" "$@"\n', 'sh', '-ec', 'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n', '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef sdg_op(\n    num_instructions_to_generate: int,\n    taxonomy: dsl.Input[dsl.Dataset],\n    sdg: dsl.Output[dsl.Dataset],\n    repo_branch: Optional[str],\n    repo_pr: Optional[int],\n):\n    import openai\n    from instructlab.sdg import generate_data\n    from instructlab.sdg.utils.taxonomy import read_taxonomy\n    from os import getenv\n\n    api_key = getenv("api_key")\n    model = getenv("model")\n    endpoint = getenv("endpoint")\n    client = openai.OpenAI(base_url=endpoint, api_key=api_key)\n\n    taxonomy_base = "main" if repo_branch or repo_pr else "empty"\n\n    print("Generating syntetic dataset for:")\n    print()\n    print(read_taxonomy(taxonomy.path, taxonomy_base))\n\n    # generate_data has a magic word for its taxonomy_base argument - `empty`\n    # it allows generating from the whole repo, see:\n    # https://github.com/instructlab/sdg/blob/c6a9e74a1618b1077cd38e713b8aaed8b7c0c8ce/src/instructlab/sdg/utils/taxonomy.py#L230\n    generate_data(\n        client=client,\n        num_instructions_to_generate=num_instructions_to_generate,\n        output_dir=sdg.path,\n        taxonomy=taxonomy.path,\n        taxonomy_base=taxonomy_base,\n        model_name=model,\n    )\n\n'],
                args=['--executor_input', '{"inputs": {"parameterValues": {"repo_name": "some-huggingface-repo"}}}', '--function_to_execute', 'sdg_op'],
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name="shared-data", mount_path="/mnt/shared-data"
                    )
                ],
            ),
        ],
    )

    # Format each string in the args list of each init container
    for container in init_containers:
        container.args = [
            arg.format(
                exec_git_clone_op_repo_url=exec_git_clone_op_repo_url,
                exec_git_clone_op_repo_branch=exec_git_clone_op_repo_branch,
                exec_git_clone_op_repo_pr=exec_git_clone_op_repo_pr,
            )
            for arg in container[0].args
        ]

    container = kubernetes.client.V1Container(
        name="sdg-preprocess",
        image="registry.access.redhat.com/ubi9/python-311:latest",
        # This command will:
        # 1. Fetch the SDG data
        # 2. Preprocess the SDG data
        # 3. Save the preprocessed data on the volume
        command=['sh', '-c', '\nif ! [ -x "$(command -v pip)" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location \'kfp==2.6.0\' \'--no-deps\' \'typing-extensions>=3.7.4,<5; python_version<"3.9"\'  &&  python3 -m pip install --quiet --no-warn-script-location \'instructlab-training@git+https://github.com/instructlab/training.git\' && "$0" "$@"\n', 'sh', '-ec', 'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n', '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef data_processing_op(\n    sdg: dsl.Input[dsl.Dataset],\n    processed_data: dsl.Output[dsl.Dataset],\n    model: dsl.Input[dsl.Artifact],\n    max_seq_len: Optional[int] = 4096,\n    max_batch_len: Optional[int] = 20000\n):\n    import instructlab.training.data_process as dp\n    import os\n    from instructlab.training import (\n        TrainingArgs,\n        DataProcessArgs,\n        )\n        # define training-specific arguments\n    training_args = TrainingArgs(\n        # define data-specific arguments\n        model_path = model.path,\n        data_path = f"{sdg.path}/*_train_msgs*.jsonl",\n        data_output_dir = processed_data.path,\n\n        # define model-trianing parameters\n        max_seq_len = max_seq_len,\n        max_batch_len = max_batch_len,\n\n       # XXX(shanand): We don\'t need the following arguments \n       # for data processing. Added them for now to avoid\n       # Pydantic validation errors for TrainingArgs\n        ckpt_output_dir = "data/saved_checkpoints",\n        num_epochs = 2,\n        effective_batch_size = 3840,\n        save_samples = 0,\n        learning_rate = 2e-6,\n        warmup_steps = 800,\n        is_padding_free = True,\n    )\n    def data_processing(train_args: TrainingArgs) -> None:\n      # early validation logic here\n      if train_args.max_batch_len < train_args.max_seq_len:\n          raise ValueError(\n              f"the `max_batch_len` cannot be less than `max_seq_len`: {train_args.max_batch_len=} < {train_args.max_seq_len=}"\n          )\n\n          # process the training data\n      if not os.path.exists(train_args.data_output_dir):\n          os.makedirs(train_args.data_output_dir, exist_ok=True)\n      dp.main(\n          DataProcessArgs(\n              # XXX(osilkin): make a decision here, either:\n              #   1. the CLI is fully responsible for managing where the data is written\n              #   2. we never cache it and simply write it to a tmp file every time.\n              #\n              # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux\n              # where the user has a defined place for new temporary data to be written.\n              data_output_path=train_args.data_output_dir,\n              model_path=train_args.model_path,\n              data_path=train_args.data_path,\n              max_seq_len=train_args.max_seq_len,\n              chat_tmpl_path=train_args.chat_tmpl_path,\n          )\n      )\n    data_processing(train_args=training_args)\n\n'],
        args=['--executor_input', '{"inputs": {"parameterValues": {"repo_name": "some-huggingface-repo"}}}', '--function_to_execute', 'data_processing_op'],
        # TODO: need another volume to read the model?
        volume_mounts=[
            kubernetes.client.V1VolumeMount(
                name=SDG_PVC_NAME, mount_path="/mnt/sdg-data"
            )
        ],
    )

    volumes = (
        [
            kubernetes.client.V1Volume(
                name="shared-data", empty_dir=kubernetes.client.V1EmptyDirVolumeSource()
            ),
            kubernetes.client.V1Volume(
                name=SDG_PVC_NAME,
                persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=SDG_PVC_NAME
                ),
            ),
        ],
    )

    # TODO: append a command to copy the preprocessed data to the PVC

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "sdg-preprocess"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[container],
            volumes=volumes,
        ),
    )

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
        backoff_limit=4,
        ttl_seconds_after_finished=100,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def run_job(namespace: str, job: kubernetes.client.V1Job):
    """
    Create and run a Kubernetes job in the specified namespace, and wait for its completion.

    Args:
        namespace (str): The namespace in which to create the job.
        job (kubernetes.client.V1Job): The job object to be created and run.

    Prints:
        str: The name of the created job.
        str: The status of the job during its execution.
        str: The logs of the pod if the job fails.

    The function will print the job's status as it progresses and will stop watching once the job
    either succeeds or fails. If the job fails, it will also print the logs of the failed pod.
    """
    # Create a job
    batch_v1 = kubernetes.client.BatchV1Api()
    resp = batch_v1.create_namespaced_job(body=job, namespace=namespace)
    logger.info("Job created. status='%s'", resp.metadata.name)

    # Wait for the job to complete
    w = kubernetes.watch.Watch()
    for event in w.stream(batch_v1.list_namespaced_job, namespace=namespace):
        job = event["object"]
        logger.info("Job: %s - %s", job.metadata.name, job.status.phase)
        if job.status.phase == "Succeeded":
            logger.info("Job completed successfully.")
            w.stop()
        elif job.status.phase == "Failed":
            core_v1 = kubernetes.client.CoreV1Api()
            logger.error()("Job failed. Pod logs:")
            pod_name = job.status.conditions[0].message.split()[-1]
            pod_log = core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=namespace
            )
            logger.info(pod_log)
            w.stop()


def create_pvc(
    name: str,
    namespace: str,
    storage_class: str,
    access_modes: list,
    size: str,
) -> kubernetes.client.V1PersistentVolumeClaim:
    """
    Create a PersistentVolumeClaim (PVC) in the specified namespace.

    Args:
        namespace (str): The namespace in which to create the PVC.
        storage_class (str): The storage class for the PVC.
        access_modes (list): The access modes for the PVC.
        size (str): The size of the PVC.

    Returns:
        kubernetes.client.V1PersistentVolumeClaim: The created PVC object.
    """
    # Create a PVC
    return kubernetes.client.V1PersistentVolumeClaim(
        metadata=kubernetes.client.V1ObjectMeta(name=name, namespace=namespace),
        spec=kubernetes.client.V1PersistentVolumeClaimSpec(
            access_modes=access_modes,
            storage_class_name=storage_class,
            resources=kubernetes.client.V1ResourceRequirements(
                requests={"storage": size}
            ),
        ),
    )


def preprocess_sdg_data(
    namespace: typing.Optional[str] = None,
    taxonomy_repo_url: typing.Optional[str] = None,
    taxonomy_repo_branch: typing.Optional[str] = None,
    taxonomy_repo_pr: typing.Optional[str] = None,
    storage_class: typing.Optional[str] = "standard",
):
    """
    Preprocesses SDG data by creating a Persistent Volume Claim (PVC) and
    initiating a job to run a pod for SDG data preprocessing.

    Args:
        namespace (str): The namespace in which the PVC and job will be created.

    Steps:
        1. Creates a PVC to hold SDG data and transformed SDG data.
        2. Initiates a job to run a pod for SDG data preprocessing.

    Note:
        - Ensure that the Kubernetes client is properly configured.
        - Add the logic for PVC creation and job initiation where indicated.
    """
    logger.info("Creating PVC for SDG data")

    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    # list of PVCs to create and their details
    pvcs = [
        {
            "name": "sdg-data",
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "1Gi",
        },
        {
            "name": "model",
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "50Gi",
        },
    ]
    for pvc in pvcs:
        try:
            v1.create_namespaced_persistent_volume_claim(
                namespace=namespace, body=create_pvc(namespace, **pvc)
            )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC `%s` already exists.", pvc["name"])
            else:
                raise
        logger.info("Successfully creayed PVC `%s` created.", SDG_PVC_NAME)

    # Create the job to run the pod to execute the SDG data preprocessing
    # Example usage
    job = create_sdg_job(
        namespace=namespace,
        job_name="sdg-preprocess",
        exec_git_clone_op_repo_url=taxonomy_repo_url,
        exec_git_clone_op_repo_branch=taxonomy_repo_branch,
        exec_git_clone_op_repo_pr=taxonomy_repo_pr,
    )
    run_job(namespace, job)


if __name__ == "__main__":
    # Configs can be set in Configuration class directly or using helper utility
    try:
        kubernetes.config.load_kube_config()
    except kubernetes.config.ConfigException:
        logger.info("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    try:
        cli()
    except Exception as e:
        logger.info("An error occurred: %s", str(e))