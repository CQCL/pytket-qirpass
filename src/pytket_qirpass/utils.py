from math import pi
import struct
import warnings

from llvmlite.binding import (  # type: ignore
    create_context,
    parse_assembly,
    parse_bitcode,
    ModuleRef,
    ValueRef,
)
from pytket.circuit import Bit, Circuit, OpType, Qubit, UnitID

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
    # Additional gates from https://github.com/qir-alliance/qat/blob/main/targets/target_4bf9.yaml:
    "__quantum__qis__swap__body": (OpType.SWAP, "%Qubit*"),
    # Additional gates:
    "__quantum__qis__phasedx__body": (OpType.PhasedX, "double, double, %Qubit*"),
    "__quantum__qis__zzmax__body": (OpType.ZZMax, "%Qubit*, %Qubit*"),
    "__quantum__qis__rxxyyzz__body": (
        OpType.TK2,
        "double, double, double, %Qubit*, %Qubit*",
    ),
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
    return len(words) >= 3 and "=" in words[1:-1]


def is_entry_point(function: ValueRef) -> bool:
    assert function.is_function
    return any(b'"EntryPoint"' in attrs for attrs in function.attributes)


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


def parse_instr(
    instr: ValueRef,
) -> tuple[OpType, list[float], list[Qubit], list[Bit]] | None:
    assert instr.is_instruction
    assert instr.opcode == "call"
    assert str(instr.type) == "void"
    operands = list(instr.operands)
    assert len(operands) >= 1
    name = operands[-1].name
    if name.startswith("__quantum__rt__"):
        warnings.warn(f"Ignoring external call: '{name}'")
        return None
    optype, _ = opdata[name]
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


def to_circuit(instrs: list[ValueRef]) -> Circuit:
    circuit = Circuit()
    for instr in instrs:
        data = parse_instr(instr)
        if data is not None:
            optype, params, q_args, c_args = data
            for q in q_args:
                circuit.add_qubit(q, reject_dups=False)
            for c in c_args:
                circuit.add_bit(c, reject_dups=False)
            args: list[UnitID] = []
            args.extend(q_args)
            args.extend(c_args)
            circuit.add_gate(optype, params, args)
    return circuit
