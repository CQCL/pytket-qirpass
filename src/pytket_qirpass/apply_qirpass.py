from math import pi
import struct
from typing import List, Optional, Set, Tuple

from llvmlite.binding import (
    create_context,
    parse_assembly,
    parse_bitcode,
    ModuleRef,
    ValueRef,
)
from pytket.circuit import Bit, Circuit, Node, OpType, Qubit
from pytket.passes import (
    BasePass,
    RemoveImplicitQubitPermutation,
    RemoveRedundancies,
    SequencePass,
    AutoRebase,
    AutoSquash,
)

# Gates taken from https://github.com/qir-alliance/qat/blob/main/targets/target_7ee0.yaml
opdata = {
    # Gates taken from https://github.com/qir-alliance/qat/blob/main/targets/target_7ee0.yaml:
    "__quantum__qis__cnot__body": (OpType.CX, "%Qubit*, %Qubit*"),
    "__quantum__qis__cz__body": (OpType.CZ, "%Qubit*, %Qubit*"),
    "__quantum__qis__h__body": (OpType.H, "%Qubit*"),
    "__quantum__qis__mz__body": (OpType.Measure, "%Qubit*, %Result* writeonly"),
    "__quantum__qis__reset__body": (OpType.Reset, "%Qubit*"),
    "__quantum__qis__rx__body": (OpType.Rx, "double, %Qubit*"),
    "__quantum__qis__ry__body": (OpType.Ry, "double, %Qubit*"),
    "__quantum__qis__rz__body": (OpType.Rz, "double, %Qubit*"),
    "__quantum__qis__rzz__body": (OpType.ZZPhase, "double, %Qubit*, %Qubit*"),
    "__quantum__qis__s__body": (OpType.S, "%Qubit*"),
    "__quantum__qis__t__body": (OpType.T, "%Qubit*"),
    "__quantum__qis__t__adj": (OpType.Tdg, "%Qubit*"),
    "__quantum__qis__x__body": (OpType.X, "%Qubit*"),
    "__quantum__qis__y__body": (OpType.Y, "%Qubit*"),
    "__quantum__qis__z__body": (OpType.Z, "%Qubit*"),
    # Additional gates:
    "__quantum__qis__phasedx__body": (OpType.PhasedX, "double, double, %Qubit*"),
    "__quantum__qis__zzmax__body": (OpType.ZZMax, "%Qubit*, %Qubit*"),
    "__quantum__qis__rxxyyzz__body": (
        OpType.TK2,
        "double, double, double, %Qubit*, %Qubit*",
    ),
}

tk_to_qir = {optype: (name, sig) for name, (optype, sig) in opdata.items()}


def encode_double(a: float) -> str:
    assert isinstance(a, float)
    encoding = struct.unpack("Q", struct.pack("d", a))
    assert len(encoding) == 1
    return f'double {"0x{:016X}".format(encoding[0])}'


def decode_double(s: str) -> float:
    assert isinstance(s, str)
    words = s.split(" ")
    assert len(words) == 2 and words[0] == "double"
    encoding = words[1]
    try:
        return float(encoding)
    except ValueError:
        n = int(encoding, 16)
        return struct.unpack("d", struct.pack("Q", n))[0]


def parse_instr(instr: ValueRef) -> Tuple[OpType, List[float], List[Qubit], List[Bit]]:
    assert instr.is_instruction
    assert instr.opcode == "call"
    assert str(instr.type) == "void"
    operands = list(instr.operands)
    assert len(operands) >= 1
    optype, _ = opdata[operands[-1].name]
    params = []
    q_args = []
    c_args = []
    for operand in operands[:-1]:
        typename = str(operand.type)
        if typename == "double":
            params.append(decode_double(str(operand)) / pi)
        elif typename == "%Qubit*":
            optext = str(operand).split(" ")
            assert optext[0] == "%Qubit*"
            if optext[1] == "null":
                assert len(optext) == 2
                q_args.append(Qubit(0))
            else:
                q_args.append(Qubit(int(optext[3])))
        else:
            assert typename == "%Result*"
            optext = str(operand).split(" ")
            assert optext[0] == "%Result*"
            if optext[1] == "null":
                assert len(optext) == 2
                c_args.append(Bit(0))
            else:
                c_args.append(Bit(int(optext[3])))
    return (optype, params, q_args, c_args)


def to_circuit(instrs: List[ValueRef]) -> Circuit:
    circuit = Circuit()
    for instr in instrs:
        optype, params, q_args, c_args = parse_instr(instr)
        for q in q_args:
            circuit.add_qubit(q, reject_dups=False)
        for c in c_args:
            circuit.add_bit(c, reject_dups=False)
        circuit.add_gate(optype, params, q_args + c_args)
    return circuit


def argrep(arg: Node) -> str:
    if arg.reg_name == "q":
        assert len(arg.index) == 1
        q = arg.index[0]
        if q == 0:
            return "%Qubit* null"
        return f"%Qubit* nonnull inttoptr (i64 {q} to %Qubit*)"
    else:
        assert arg.reg_name == "c"
        assert len(arg.index) == 1
        c = arg.index[0]
        if c == 0:
            return "%Result* null"
        return f"%Result* nonnull inttoptr (i64 {c} to %Result*)"


def is_known_type(instr: ValueRef) -> bool:
    assert instr.is_instruction
    if instr.opcode != "call":
        return False
    operands = list(instr.operands)
    assert len(operands) >= 1
    return operands[-1].name in opdata


def partition_instrs(
    instrs: List[ValueRef],
) -> List[Tuple[List[ValueRef], List[ValueRef]]]:
    # Organize the instructions into a list of pairs of lists, each pair consisting of
    # all-known and all-unknown, preserving the original order.
    n_instrs = len(instrs)
    if n_instrs == 0:
        return []
    i = 0
    known_sub_block = []
    while (i < n_instrs) and is_known_type(instrs[i]):
        known_sub_block.append(instrs[i])
        i += 1
    unknown_sub_block = []
    while (i < n_instrs) and not is_known_type(instrs[i]):
        unknown_sub_block.append(instrs[i])
        i += 1
    return [(known_sub_block, unknown_sub_block)] + partition_instrs(instrs[i:])


def compile_basic_block_ll(basic_block: ValueRef, comp_pass: BasePass):
    assert basic_block.is_block
    bb_ll = ""
    if basic_block.name != "":
        bb_ll += str(basic_block).split("\n")[1] + "\n"  # keep top line with label
    instructions = list(basic_block.instructions)
    # Take maximal blocks of "known" quantum instructions and convert them to circuits;
    # leave the rest (assignments, classical operations, branches etc) as they are.
    sub_blocks = partition_instrs(instructions)
    for known_sub_block, unknown_sub_block in sub_blocks:
        # Convert known instructions to a circuit
        circuit = to_circuit(known_sub_block)
        # Compile the circuit
        comp_pass.apply(circuit)
        # Convert the compiled circuit to QIR instructions and write them to new_ll
        for cmd in circuit:
            op, args = cmd.op, cmd.args
            optype, params = op.type, op.params
            name, sig = tk_to_qir[optype]
            assert len(params) + len(args) == len(sig.split(", "))
            paramreps = [encode_double(param * pi) for param in params]
            argreps = [argrep(arg) for arg in args]
            bb_ll += f"  call void @{name}({', '.join(paramreps + argreps)})\n"
        # Write out the remaining instructions
        for instr in unknown_sub_block:
            bb_ll += str(instr) + "\n"
    return bb_ll


def bc_to_module(bc: bytes) -> ModuleRef:
    ctx = create_context()
    module = parse_bitcode(bc, context=ctx)
    module.verify()
    return module


def ll_to_module(ll: str) -> ModuleRef:
    ctx = create_context()
    module = parse_assembly(ll, context=ctx)
    module.verify()
    return module


def ll_to_bc(ll: str) -> bytes:
    module = ll_to_module(ll)
    return module.as_bitcode()


def is_header_line(line: str) -> bool:
    if line == "":
        return True
    words = line.split(" ")
    return len(words) >= 3 and "=" in words[1:-1]


def is_entry_point(function: ValueRef) -> bool:
    assert function.is_function
    return any(b'"EntryPoint"' in attrs for attrs in function.attributes)


def apply_qirpass(
    qir_bitcode: bytes,
    comp_pass: Optional[BasePass],
    target_1q_gates: Set[OpType],
    target_2q_gates: Set[OpType],
) -> bytes:
    """Apply the given pass to basic blocls of the QIR.

    Only QIR conforming to the Quantinuum profile is accepted as input.

    The compilation pass is followed by a rebase to the target gateset (with redundancy
    removal and squashing of single-qubit gates).

    If the pass introduces implicit swaps, these are automatically replaced with
    explicit swaps afterwards, which enlarge the resulting circuits; it is therefore
    recommended to provide a pass that does not introduce implicit swaps.

    The compilation pass must be unitary-preserving, since it is applied to basic blocks
    within the larger program. (For example, it cannot assume anything about the initial
    state of the qubits.)

    :param qir_bitcode: QIR bitcode
    :param comp_pass: pytket compilation pass to apply to the basic blocks
    :return: transformed QIR bitcode
    """

    module = bc_to_module(qir_bitcode)

    new_ll = ""

    module_ll_lines = str(module).split("\n")
    for line in module_ll_lines:
        if is_header_line(line):
            new_ll += line + "\n"
        else:
            break

    functions = list(module.functions)
    entries = [f for f in functions if is_entry_point(f)]
    assert len(entries) == 1
    f0 = entries[0]
    f0_attrs = list(f0.attributes)
    assert len(f0_attrs) == 1
    f0_attr = f0_attrs[0].decode("utf-8")

    target_gates = target_1q_gates | target_2q_gates

    pass_list = [] if comp_pass is None else [comp_pass]
    pass_list.extend(
        [
            RemoveImplicitQubitPermutation(),
            AutoRebase(target_gates),
            AutoSquash(target_1q_gates),
            RemoveRedundancies(),
        ]
    )
    comp_pass = SequencePass(pass_list)

    for function in functions:
        new_ll += "\n"
        basic_blocks = list(function.blocks)
        if len(basic_blocks) == 0:
            # This is an external declaration
            decl = str(function)
            # if it's not one of the "known" functions, include verbatim
            if not any(name in decl for name in opdata):
                new_ll += decl
        else:
            # This is an inline definition. There must only be one.
            assert is_entry_point(function)
            lines = str(function).split("\n")
            assert len(lines) >= 2
            first_line = lines[0]
            assert " #0 " in first_line and first_line.endswith("{")
            new_ll += first_line + "\n"
            new_ll += "\n".join(
                compile_basic_block_ll(basic_block, comp_pass)
                for basic_block in basic_blocks
            )
            new_ll += "}\n"

    # Declarations of "known" functions
    for name, (optype, sig) in opdata.items():
        if optype in target_gates | {OpType.Measure, OpType.Reset}:
            if (
                name in new_ll
            ):  # Skip if e.g. circuit was all classical and Qubit undeclared.
                new_ll += f"\ndeclare void @{name}({sig}) local_unnamed_addr\n"

    # Function attributes:
    new_ll += f"\nattributes #0 = {{ {f0_attr} }}\n"

    # Metadata:
    for line in str(module).split("\n"):
        if line.startswith("!"):
            new_ll += line + "\n"

    return ll_to_bc(new_ll)
