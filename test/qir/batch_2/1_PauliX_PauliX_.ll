; ModuleID = '1_PauliX_PauliX.bc'
source_filename = "qat-link"

%Qubit = type opaque
%Result = type opaque

define void @Microsoft__Quantum__Samples__RepeatUntilSuccess__Run() #0 {
entry:
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* null)
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %0 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %0, label %then0__1, label %continue__6

then0__1:                                         ; preds = %entry
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__6

continue__6:                                      ; preds = %then0__1, %entry
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %1 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %1, label %continue__8, label %then0__2

then0__2:                                         ; preds = %continue__6
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %2 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  br i1 %2, label %then0__3, label %continue__12

then0__3:                                         ; preds = %then0__2
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__12

continue__12:                                     ; preds = %then0__3, %then0__2
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %3 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  br label %continue__8

continue__8:                                      ; preds = %continue__12, %continue__6
  call void @__quantum__qis__rz__body(double 0x4001B6E192EBBE42, %Qubit* null)
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 2 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 3 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 4 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* null)
  %4 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  br i1 %4, label %then0__4, label %continue__18

then0__4:                                         ; preds = %continue__8
  call void @__quantum__qis__x__body(%Qubit* null)
  br label %continue__18

continue__18:                                     ; preds = %then0__4, %continue__8
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__rt__tuple_start_record_output()
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 2 to %Result*))
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 3 to %Result*))
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 4 to %Result*))
  call void @__quantum__rt__tuple_end_record_output()
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__t__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare void @__quantum__qis__t__adj(%Qubit*)

declare %Result* @__quantum__rt__result_get_zero()

declare void @__quantum__rt__result_update_reference_count(%Result*, i32)

declare %Result* @__quantum__qis__m__body(%Qubit*)

declare void @__quantum__qis__reset__body(%Qubit*)

declare %Result* @__quantum__rt__result_get_one()

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare void @__quantum__qis__x__body(%Qubit*)

declare void @__quantum__qis__z__body(%Qubit*)

declare void @__quantum__qis__rz__body(double, %Qubit*)

declare void @__quantum__rt__qubit_release(%Qubit*)

declare void @__quantum__rt__tuple_start_record_output()

declare void @__quantum__rt__result_record_output(%Result*)

declare void @__quantum__rt__tuple_end_record_output()

declare void @__quantum__qis__mz__body(%Qubit*, %Result*)

declare i1 @__quantum__qis__read_result__body(%Result*)

attributes #0 = { "EntryPoint" "maxQubitIndex"="2" "maxResultIndex"="4" "requiredQubits"="3" "requiredResults"="5" }
