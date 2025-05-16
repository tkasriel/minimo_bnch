# Conjecturing Theorems from Minimal Axioms

Based on Minimo. You can find it on [arXiv](https://arxiv.org/abs/2407.00695). 

### Compiling the environment

The Peano enviroment is written in Rust and has a Python API via [PyO3](https://pyo3.rs/v0.18.2/).

To compile it, you'll first need to install the Rust toolchain. For that, use [rustup](https://rustup.rs/).

The environment compiles two targets: a `peano` binary, that can be used stand-alone, as well as a `peano` Python library, which we'll use to interact with agents. To build, we'll use `maturin`, a PyO3 build system. Make sure you have Python version at least 3.9. Then, first make a Python virtual environment (conda works too):

```sh
[peano] $ python -m venv /path/to/new/virtual/environment
[peano] $ source /path/to/new/virtual/environment/bin/activate
```

Now, within your environment, install maturin with:

```sh
[peano] $ pip install maturin
```

With `maturin` you can now compile both the Peano environment and library with it:

```sh
[peano] $ cd environment
[environment] $ maturin dev --release  # This will compile the Peano library.
[...]
[environment] $ cargo build --bin peano --release  # This compiles the peano executable.
```

If this works without errors, you're ready to use Peano from Python. Before running any scripts, make sure to install dependencies with:

```sh
[environment] $ cd ../learning
[learning] $ pip install -r requirements.txt
```

The entry point for the conjecture-prove loop is in [learning/bootstrap.py](bootstrap.py). It should suffice to pass it one of the domain configuration files, such as:

```sh
[learning] $ python bootstrap.py theory=nat-mul
```

We use hydra for configuration -- the relevant file here is [config/bootstrap.yaml](config/bootstrap.yaml). This will run the loop in "sequential" mode, in a single process. By default, the program will run using multiprocessing. To disable this, set `use_multiprocessing` option to false

```sh
[learning] $ python bootstrap.py use_multiprocessing=False
```