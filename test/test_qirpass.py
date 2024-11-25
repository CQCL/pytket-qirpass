import os
from pathlib import Path
import unittest

from pytket_qirpass import apply_qirpass

from llvmlite.binding import create_context, parse_assembly, parse_bitcode, ModuleRef
from pyqir import Context, Module
from pytket.circuit import OpType
from pytket.passes import FullPeepholeOptimise, SynthesiseTK

QIR_DIR = Path(__file__).parent.resolve() / "qir"


def ll_to_module(ll: str) -> ModuleRef:
    ctx = create_context()
    module = parse_assembly(ll, context=ctx)
    module.verify()
    return module


def ll_to_bc(ll: str) -> bytes:
    module = ll_to_module(ll)
    return module.as_bitcode()


prognames_1 = [
    "add_only",
    "and_only",
    "classical_and_controlflow",
    "ClassicalCircuit",
    "ClassicalReg2ConstCircuit",
    "collapse_jump_instr",
    "collapse_jump_left",
    "collapse_jump_right",
    "collapse_nested_jump_instr",
    "collapse_simple_instr_chain",
    "cx_only",
    "div_only",
    "eq_only",
    "GenerateAnd",
    "GenerateOr",
    "GenerateXorAsBitNegation",
    "GenerateXorOr",
    "h_only",
    "lshr_only",
    "measure_only",
    "mul_only",
    "neg_only",
    "neq_only",
    "NestedConditionalsCircuit",
    "nested_conditionals_crossed",
    "nested_conditionals_else",
    "nested_conditionals",
    "nested_conditionals_then",
    "not_supported",
    "one_conditional_diamond",
    "one_conditional_diamond_opposite",
    "one_conditional_else",
    "one_conditional_if",
    "one_conditional",
    "one_conditional_then",
    "or_only",
    "phasedx_only",
    "purely_classical",
    "ReadResult",
    "RebasedCircuit",
    "reset_only",
    "RoundTripDiamondConditional",
    "RoundTripSimpleChain",
    "RtBoolRecordOutput",
    "RtIntRecordOutput",
    "RtResultRecordOutput",
    "RUSLoopXX-1",
    "rx_only",
    "rz_only",
    "select_and_controlflow",
    "select",
    "shl_only",
    "SimpleConditionalCircuit",
    "SimpleGroverBaseProfile",
    "SimpleGroverSample.BaseProfile",
    "SimpleGroverSampleOptimised",
    "sub_only",
    "tadj_only",
    "tagged_rt_functions",
    "teleportchain_baseprofile",
    "TestBellCircuit",
    "t_only",
    "uge_only",
    "ugt_only",
    "ule_only",
    "ult_only",
    "untagged_rt_functions",
    "wasm_and_controlflow",
    "WASM",
    "WASM_noinputs",
    "wasm_only_test",
    "x_only",
    "xor_only",
    "y_only",
    "zext",
    "z_only",
]

prognames_2 = [
    "1_PauliX_PauliX_",
    "2_PauliX_PauliX_",
    "3_PauliX_PauliX_",
    "4_PauliX_PauliX_",
    "add_only",
    "and_only",
    "ArithmeticOps.TargetedAlt_",
    "bad_div",
    "bool_tag",
    "classical_flow2",
    "classical_flow3",
    "classical_flow4",
    "classical_flow5",
    "classical_flow6",
    "classical_flow",
    "comparison_instrs",
    "conditional_test_2",
    "conditional-test-3",
    "conditional_test_3",
    "conditional_test_4",
    "conditional_test",
    "cond_ops",
    "control_flow_complex.baseProfile",
    "control_flow_long.baseProfile",
    "control_flow_nested.baseProfile",
    "control_flow_short.baseProfile",
    "cx_only",
    "Distillation",
    "Distillation_other",
    "Distillation_small",
    "eq_only",
    "fallthrough_simple",
    "gate_stress_5qb",
    "gate_stress_5qb_",
    "gate_stress_test_10k",
    "gate_stress_test_10k_",
    "gate_stress_test_10qb_10k",
    "gate_stress_test_10qb_10k_",
    "gate_stress_test_10qb",
    "gate_stress_test_10qb_",
    "gate_stress_test_15k",
    "gate_stress_test_15k_",
    "gate_stress_test_20k",
    "gate_stress_test_20k_",
    "gate_stress_test_25k",
    "gate_stress_test_25k_",
    "gate_stress_test_50k",
    "gate_stress_test",
    "gate_stress_test_",
    "ge_only",
    "ghz_fail_",
    "gt_only",
    "h_only",
    "IntegerComparison.TargetedAlt_",
    "IntegerComparison.Targeted_",
    "int_record_out",
    "int_result_with_tag",
    "int_tag_alt_",
    "iqpe",
    "iqpe_",
    "le_only",
    # "LogicalRb-BaseProfile.opt", # >1 defined function
    "loopRecursion.baseProfile",
    "lshr_only",
    "lt_only",
    "MagicStateDistillation-BaseProfile.10X",
    "MagicStateDistillation-BaseProfile.3",
    "MagicStateDistillation-BaseProfile-3X",
    "MagicStateDistillation-BaseProfile.3X",
    # "MagicStateDistillation.BaseProfile-4X", # >1 defined function
    "MagicStateDistillation-BaseProfile-5X",
    "MagicStateDistillation-BaseProfile-6X",
    "MagicStateDistillation-BaseProfile.6X",
    "MagicStateDistillation-BaseProfile-7X",
    "MagicStateDistillation-BaseProfile.8Y",
    "MagicStateDistillation-BaseProfile.9X",
    # "MagicStateDistillation-BaseProfile-NoInline.10X", # >1 defined function
    # "MagicStateDistillation-BaseProfile.opt", # >1 defined function
    "MagicStateDistillation-Grouping.10X",
    "MagicStateDistillation-Grouping.3Y",
    "many_gates.baseProfile",
    "many_ops",
    "measure_only",
    "measure_result",
    "most_features",
    # "most_features_", # "Instruction does not dominate all uses!"
    "mul_only",
    "multi_arith",
    "mutables_ops_branches.baseProfile",
    "native_gates",
    "negative_arith",
    "ne_only",
    "nested_conditions",
    "no_entry",
    "no-ops",
    "no_qubits",
    "no_results",
    "null_tag",
    "one_conditional",
    "or_only",
    "phi_test_1",
    "phi_test_2",
    "phi_test_3",
    "phi_test_4",
    "phi_test_5",
    "phi_test_6",
    "phi_test_7",
    "phi_test_8",
    "qir_hybrid",
    # "RandomWalkPE1.baseProfile", # contains Rz with non-constant parameter
    "reset_only",
    "result_tag_alt",
    # "result_tag_alt_", # invalid LLVM
    "result_tag",
    "result_with_constant_args",
    "RUSLoop10",
    "RUSLoop10_",
    "RUSLoop1",
    "RUSLoop1_",
    "RUSLoop2",
    "RUSLoop2_",
    "RUSLoop3_",
    "RUSLoop4_",
    "RUSLoop5_",
    "RUSLoop6_",
    "RUSLoop7_",
    "RUSLoop8_",
    "RUSLoop9_",
    "RUSLoopYY-10_",
    "RUSLoopYY-1_",
    "RUSLoopYY-2_",
    "RUSLoopYY-3_",
    "RUSLoopYY-4_",
    "RUSLoopYY-5_",
    "RUSLoopYY-6_",
    "RUSLoopYY-7_",
    "RUSLoopYY-8_",
    "RUSLoopYY-9_",
    "RUSLoopZZ-10_",
    "RUSLoopZZ-1_",
    "RUSLoopZZ-2_",
    "RUSLoopZZ-3_",
    "RUSLoopZZ-4_",
    "RUSLoopZZ-5_",
    "RUSLoopZZ-6_",
    "RUSLoopZZ-7_",
    "RUSLoopZZ-8_",
    "RUSLoopZZ-9_",
    "rx_only",
    "rz_only",
    "select",
    "select-nested-const-cond",
    "select-nested",
    "sequential_flow_1",
    "sequential_flow_2",
    "sge",
    "sgt",
    "shl_only",
    "SimpleGroverBaseProfile",
    "SimpleGroverGrouped",
    "SimpleGroverSample.BaseProfile",
    # "SimpleGroverSample", # >1 defined function
    # "SimpleGroverSample.O1", # >1 defined function
    "sle",
    "slt",
    "sub_only",
    "tadj_only",
    "tagged_duplicates",
    "tagged_result_control_flow",
    "tagged_result_control_flow_",
    "teleportchain.BaseProfile",
    "teleport-chain-grouping",
    "t_only",
    "udiv_only",
    "x_only",
    "xor_only",
    "XX_recursion_limit8",
    "y_only",
    "zext",
    "zext-reg",
    "zext-use",
    "z_only",
]


def verify_with_llvmlite(qir_bitcode: bytes) -> None:
    ctx = create_context()
    module = parse_bitcode(qir_bitcode, context=ctx)
    module.verify()


def verify_with_pyqir(qir_bitcode: bytes) -> None:
    ctx = Context()
    module = Module.from_bitcode(ctx, qir_bitcode)
    module.verify()


def check_compilation(qir_ll_in: str) -> None:
    passes = [
        None,
        SynthesiseTK(),
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2, allow_swaps=False),
    ]
    gatesets2q = [
        {OpType.ZZMax, OpType.ZZPhase},
        {OpType.ZZMax, OpType.ZZPhase, OpType.TK2},
    ]
    qir_in = ll_to_bc(qir_ll_in)
    for comp_pass in passes:
        for gateset2q in gatesets2q:
            qir_out = apply_qirpass(
                qir_in, comp_pass, {OpType.PhasedX, OpType.Rz}, gateset2q
            )
            verify_with_llvmlite(qir_out)
            verify_with_pyqir(qir_out)


class TestQirPass(unittest.TestCase):
    def test_qirpass(self):
        for progname in prognames_1:
            with self.subTest(msg=f"Compiling {progname}"):
                with open(
                    QIR_DIR / "batch_1" / f"{progname}.ll", encoding="utf-8"
                ) as f:
                    qir_ll_in = f.read()
                check_compilation(qir_ll_in)
        if os.getenv("PYTKET_QIRPASS_RUN_ALL_TESTS"):
            for progname in prognames_2:
                with self.subTest(msg=f"Compiling {progname}"):
                    with open(
                        QIR_DIR / "batch_2" / f"{progname}.ll", encoding="utf-8"
                    ) as f:
                        qir_ll_in = f.read()
                    check_compilation(qir_ll_in)


if __name__ == "__main__":
    unittest.main()
