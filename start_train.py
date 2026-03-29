import os
import sys
import modal
import subprocess
import wandb

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install([
        "uv", "pyyaml", "wandb", "pytest",
        "humanfriendly", "matplotlib", "numpy", "regex",
        "torch", "tqdm", "pandas", "ty",
        "tensorboard", "einops", "einx", "jaxtyping",
        "psutil", "submitit", "tiktoken", "tokenizers"
    ])
    .add_local_dir(".", "/app")
)


app = modal.App("llm-training", image=image)
volume = modal.Volume.from_name("llm-data", create_if_missing=True)



def train_loop_wrapper():
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY 没有从 Modal Secret 注入")
    wandb.login(key=wandb_key)

    workdir = "/app/cs336_systems"
    os.chdir(workdir)

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/cs336-basics")
    sys.path.insert(0, "/app/cs336-basics/cs336_basics")

    import cs336_systems.train as train_mod
    if hasattr(train_mod, 'data_args'):
        for key in ['train_data_path', 'val_data_path', 'test_data_path']:
            if key in train_mod.data_args:
                train_mod.data_args[key] = os.path.join("/app", train_mod.data_args[key])

    from cs336_systems.train import train_loop
    train_loop()


@app.function(
    gpu="B200",
    volumes={"/data": volume},
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train():
    subprocess.run(["nvidia-smi"])
    train_loop_wrapper()
    volume.commit()


@app.function(
    gpu=None,
    timeout=60 * 30
)
def run_tests():
    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/cs336-basics")
    sys.path.insert(0, "/app/cs336-basics/cs336_basics")
    subprocess.run(["uv", "run", "pytest"], check=True)


@app.local_entrypoint()
def main():
    train.remote()