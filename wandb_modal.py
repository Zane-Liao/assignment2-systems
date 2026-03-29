import os

import modal

app = modal.App()


@app.function(secrets=[modal.Secret.from_name("wandb-secret")])
def f():
    print(os.environ["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"])
