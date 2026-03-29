import modal

# 1) Define a Modal Image that includes NumPy
image = modal.Image.debian_slim().pip_install("numpy")

# 2) Attach the image
app = modal.App("example-custom-container", image=image)


@app.function()
def square(x=2):
    # 3) Inside the container, import and use the library
    import numpy as np

    print(f"The square of {x} is {np.square(x)}")