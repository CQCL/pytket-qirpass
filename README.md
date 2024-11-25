# pytket-qirpass

This module provides a method to optimize QIR using pytket, and a method to
convert QIR to pytket for simple circuits.

## Installation

Python 3.10, 3.11 or 3.12 is required.

To install from PyPI:

```shell
pip install pytket-qirpass
```

## Usage

### Optimizing QIR with `apply_qirpass`

The function `apply_qirpass` takes as input

- some QIR bitcode
- a pytket compilation pass
- a target gateset

and outputs some new QIR bitcode, where the pass has been applied to the basic
blocks in the input program, followed by a rebase to the target gateset.

For example:

```python
from pytket_qirpass import apply_qirpass
from pytket.circuit import OpType
from pytket.passes import FullPeepholeOptimise

qir_out = apply_qirpass(
    qir_bitcode=qir_in,
    comp_pass=FullPeepholeOptimise(allow_swaps=False),
    target_1q_gates={OpType.Rx, OpType.Rz},
    target_2q_gates={OpType.ZZPhase},
)
```

Both the input and the output are Python `bytes` objects.

Provided the pass preserves the circuit semantics, `apply_qirpass` preserves
the QIR semantics.

### Converting QIR to pytket with `qir_to_pytket`

The function `qir_to_pytket` takes as input some QIR bitcode and outputs a
pytket circuit.

For example:

```python
from pytket_qirpass import qir_to_pytket

circ = qir_to_pytket(qir_bitcode=qir_in)
```

The program represented by the bitcode must consist of a single basic block
comprised of quantum operations, i.e. `__quantum__qis__*` instructions; any
`__quantum__rt__*` instructions are accepted but ignored.
