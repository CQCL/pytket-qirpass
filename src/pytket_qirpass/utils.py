import re
import struct
import warnings
from math import pi

from llvmlite.binding import (  # type: ignore
    ModuleRef,
    ValueRef,
    create_context,
    parse_assembly,
    parse_bitcode,
)
from pytket.circuit import Bit, Circuit, OpType, Qubit, UnitID

opdata = {
    # Gates taken from:
    # https://github.com/qir-alliance/qat/blob/main/targets/target_4bf9.yaml
    # https://github.com/qir-alliance/qat/blob/main/targets/target_7ee0.yaml
    # https://github.com/qir-alliance/qat/blob/main/targets/target_b340.yaml
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
    "__quantum__qis__swap__body": (OpType.SWAP, "%Qubit*"),
    "__quantum__qis__t__adj": (OpType.Tdg, "%Qubit*"),
    "__quantum__qis__t__body": (OpType.T, "%Qubit*"),
    "__quantum__qis__x__body": (OpType.X, "%Qubit*"),
    "__quantum__qis__y__body": (OpType.Y, "%Qubit*"),
    "__quantum__qis__z__body": (OpType.Z, "%Qubit*"),
    # Additional gates:
    "__quantum__qis__phasedx__body": (OpType.PhasedX, "double, double, %Qubit*"),
    "__quantum__qis__rxxyyzz__body": (
        OpType.TK2,
        "double, double, double, %Qubit*, %Qubit*",
    ),
    "__quantum__qis__tk2__body": (
        OpType.TK2,
        "double, double, double, %Qubit*, %Qubit*",
    ),
    "__quantum__qis__u1q__body": (OpType.PhasedX, "double, double, %Qubit*"),
    "__quantum__qis__zz__body": (OpType.ZZMax, "%Qubit*, %Qubit*"),
    "__quantum__qis__zzmax__body": (OpType.ZZMax, "%Qubit*, %Qubit*"),
}


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
    return len(words) >= 3 and "=" in words[1:-1]  # noqa: PLR2004


def is_entry_point(function: ValueRef) -> bool:
    assert function.is_function
    return any(b'"EntryPoint"' in attrs for attrs in function.attributes)


def decode_double(s: str) -> float:
    assert isinstance(s, str)
    words = s.split(" ")
    assert len(words) == 2 and words[0] == "double"  # noqa: PLR2004
    encoding = words[1]
    try:
        return float(encoding)
    except ValueError:
        n = int(encoding, 16)
        return struct.unpack("d", struct.pack("Q", n))[0]


def qubit_from_operand(operand: ValueRef) -> Qubit:
    optext = str(operand).split(" ")
    assert optext[0] == "%Qubit*"
    if optext[1] == "null":
        assert len(optext) == 2  # noqa: PLR2004
        return Qubit(0)
    return Qubit(int(optext[3]))


def bit_from_operand(operand: ValueRef) -> Bit:
    optext = str(operand).split(" ")
    assert optext[0] == "%Result*"
    if optext[1] == "null":
        assert len(optext) == 2  # noqa: PLR2004
        return Bit(0)
    return Bit(int(optext[3]))


def parse_operands(
    operands: list[ValueRef],
) -> tuple[list[float], list[Qubit], list[Bit]]:
    params, q_args, c_args = [], [], []
    for operand in operands:
        typename = str(operand.type)
        if typename == "double":
            params.append(decode_double(str(operand)) / pi)
        elif typename == "%Qubit*":
            q_args.append(qubit_from_operand(operand))
        else:
            assert typename == "%Result*"
            c_args.append(bit_from_operand(operand))
    return params, q_args, c_args


BARRIER_PATTERN = re.compile("__quantum__qis__barrier(?P<n>.*?)__body")
GROUP_PATTERN = re.compile("__quantum__qis__group(?P<n>.*?)__body")
ORDER_PATTERN = re.compile("__quantum__qis__order(?P<n>.*?)__body")


def parse_instr(
    instr: ValueRef,
) -> tuple[OpType, list[float], list[Qubit], list[Bit], str] | None:
    assert instr.is_instruction
    assert instr.opcode == "call"
    assert str(instr.type) == "void"
    operands = list(instr.operands)
    assert len(operands) >= 1
    name = operands[-1].name
    if name.startswith("__quantum__rt__"):
        warnings.warn(f"Ignoring external call: '{name}'")  # noqa: B028
        return None

    match = BARRIER_PATTERN.match(name)
    if match is not None:
        assert len(operands) == int(match.group("n")) + 1
        params, q_args, c_args = parse_operands(operands[:-1])
        assert params == []
        return (OpType.Barrier, [], q_args, c_args, "")

    match = GROUP_PATTERN.match(name)
    if match is not None:
        n = int(match.group("n"))
        assert len(operands) == n + 1
        params, q_args, c_args = parse_operands(operands[:-1])
        assert params == []
        assert c_args == []
        return (OpType.Barrier, [], q_args, [], f"group{n}")

    match = ORDER_PATTERN.match(name)
    if match is not None:
        n = int(match.group("n"))
        assert len(operands) == n + 1
        params, q_args, c_args = parse_operands(operands[:-1])
        assert params == []
        assert c_args == []
        return (OpType.Barrier, [], q_args, [], f"order{n}")

    if name == "__quantum__qis__sleep__body":
        operand0, operand1, _ = operands
        assert str(operand0.type) == "%Qubit*"
        assert str(operand1.type) == "double"
        return (
            OpType.Barrier,
            [],
            [qubit_from_operand(operand0)],
            [],
            f"sleep({decode_double(str(operand1))})",
        )

    optype, _ = opdata[name]
    params, q_args, c_args = parse_operands(operands[:-1])
    return (optype, params, q_args, c_args, "")


def to_circuit(instrs: list[ValueRef]) -> Circuit:
    circuit = Circuit()
    for instr in instrs:
        data = parse_instr(instr)
        if data is not None:
            optype, params, q_args, c_args, barrier_data = data
            for q in q_args:
                circuit.add_qubit(q, reject_dups=False)
            for c in c_args:
                circuit.add_bit(c, reject_dups=False)
            args: list[UnitID] = []
            args.extend(q_args)
            args.extend(c_args)
            if optype == OpType.Barrier:
                circuit.add_barrier(units=args, data=barrier_data)
            else:
                circuit.add_gate(optype, params, args)
    return circuit
