#!/usr/bin/env python3

"""
Standalone Distributed training script

This script provides a standalone version of the pipeline.py script, designed to be used when
Kubeflow pipelines are not available.

Usage:
    This script can be executed directly from the command line. Ensure that the Kubernetes client is
    properly configured before running the script.

Dependencies:
    kubernetes: The Kubernetes Python client library.
    click: A package for creating command-line interfaces.

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
    default="{{exec_git_clone_op_repo_url}}",
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

    This function generates a Kubernetes Job object configured to run SDG steps.

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
                image="{{exec_git_clone_op_image}}",
                command=["/bin/sh", "-c"],
                args={{exec_git_clone_op_args}},
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name="shared-data", mount_path="/mnt/shared-data"
                    )
                ],
            ),
            kubernetes.client.V1Container(
                name="sdg-op-generate-synthetic-data",
                image="{{exec_sdg_op_image}}",
                command={{exec_sdg_op_command}},
                args={{exec_sdg_op_args}},
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
        image="{{exec_data_processing_op_image}}",
        # This command will:
        # 1. Fetch the SDG data
        # 2. Preprocess the SDG data
        # 3. Save the preprocessed data on the volume
        command={{exec_data_processing_op_command}},
        args={{exec_data_processing_op_args}},
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
        taxonomy_repo_url (str): The URL of the taxonomy repository.
        taxonomy_repo_branch (str): The branch of the taxonomy repository.
        taxonomy_repo_pr (int): The pull request number of the taxonomy repository.
        storage_class (str): The storage class to use for the PersistentVolumeClaim.

    Steps:
        1. Creates a PVC to hold SDG data and transformed SDG data.
        2. Initiates a job to run a pod for SDG data preprocessing.
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
