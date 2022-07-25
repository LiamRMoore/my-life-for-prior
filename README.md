# PyMC fun with Jax + GPU

This is a fork of [the excellent PyMC port of the book exercises for Statistical Rethinking, 2nd Edition](https://github.com/pymc-devs/pymc-resources/tree/main/Rethinking_2).

It contains my own takes, adaptations and segues around these exercises, and uses JAX for sampling with the GPU (at the time of writing a powerful and very new experimental feature of pymc - you may find the [associated dockerfile](./Dockerfile.gpu)) useful).

Start the environment with:

```
docker-compose up -d
```

Which will launch a Jupyter notebook server in the background. You can read the URL by running:

```
docker-compose logs pymc_gpu
```

You should have an nvidia driver installed, and have a CUDA-capable GPU.
