# pytket-qirpass

This module provides a method to optimize QIR using pytket.

## Installation

Python 3.10, 3.11, 3.12 or 3.13 is required.

To install from PyPI:

```shell
pip install pytket-qirpass
```

## Usage

This module provides a single function, `apply_qirpass`, which takes as input

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
