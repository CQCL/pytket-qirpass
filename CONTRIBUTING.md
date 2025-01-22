The package is built with the `flit` backend, so it is first necessary to
install this:

```shell
pip install flit
```

Then to install from the top-level directory (containing `pyproject.toml`):

```shell
flit install
```

To run unit tests:

```shell
python -m unittest
```

To run the full set of tests, which take a few minutes, run the above command
with the environment variable `PYTKET_QIRPASS_RUN_ALL_TESTS` set.
