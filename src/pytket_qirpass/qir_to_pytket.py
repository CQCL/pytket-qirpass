from .utils import bc_to_module, is_entry_point, to_circuit

from pytket.circuit import Circuit


def qir_to_pytket(qir_bitcode: bytes) -> Circuit:
    """Convert QIR to a pytket circuit.

    :param qir_bitcode: QIR bitcode
    :return: pytket circuit
    """
    module = bc_to_module(qir_bitcode)
    entries = [f for f in module.functions if is_entry_point(f)]
    assert len(entries) == 1
    blocks = list(entries[0].blocks)
    assert len(blocks) == 1
    instructions = list(
        filter(lambda instr: instr.opcode == "call", blocks[0].instructions)
    )
    return to_circuit(instructions)
