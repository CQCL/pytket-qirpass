; ModuleID = 'control_flow_nested.ll'
source_filename = "control_flow_nested.ll"

%Result = type opaque
%Qubit = type opaque
%String = type opaque

@0 = internal constant [3 x i8] c"()\00"

declare %Result* @__quantum__rt__result_get_one() local_unnamed_addr

declare i1 @__quantum__rt__result_equal(%Result*, %Result*) local_unnamed_addr

declare void @__quantum__rt__result_update_reference_count(%Result*, i32) local_unnamed_addr

declare %Qubit* @__quantum__rt__qubit_allocate() local_unnamed_addr

declare void @__quantum__rt__qubit_release(%Qubit*) local_unnamed_addr

declare %Result* @__quantum__qis__m__body(%Qubit*) local_unnamed_addr

declare void @__quantum__qis__reset__body(%Qubit*) local_unnamed_addr

declare void @__quantum__qis__h__body(%Qubit*) local_unnamed_addr

declare void @__quantum__qis__z__body(%Qubit*) local_unnamed_addr

declare %String* @__quantum__rt__string_create(i8*) local_unnamed_addr

define void @control_flow_nested__HelloQ() local_unnamed_addr #0 {
entry:
  br label %body__1.i.i

body__1.i.i:                                      ; preds = %exiting__1.i.i, %entry
  %i1.i.i = phi i64 [ 0, %entry ], [ %4, %exiting__1.i.i ]
  tail call void @__quantum__qis__h__body(%Qubit* null)
  tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  tail call void @__quantum__qis__reset__body(%Qubit* null)
  %0 = tail call i1 @__quantum__qir__read_result(%Result* null)
  br i1 %0, label %body__1.i.i.i, label %exiting__1.i.i

body__1.i.i.i:                                    ; preds = %body__1.i.i, %exiting__1.i.i.i
  %i1.i.i.i = phi i64 [ %2, %exiting__1.i.i.i ], [ 0, %body__1.i.i ]
  tail call void @__quantum__qis__h__body(%Qubit* null)
  tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 1 to %Result*))
  tail call void @__quantum__qis__reset__body(%Qubit* null)
  %1 = tail call i1 @__quantum__qir__read_result(%Result* nonnull inttoptr (i64 1 to %Result*))
  br i1 %1, label %then0__1.i.i.i, label %exiting__1.i.i.i

then0__1.i.i.i:                                   ; preds = %body__1.i.i.i
  tail call void @__quantum__qis__z__body(%Qubit* null)
  br label %exiting__1.i.i.i

exiting__1.i.i.i:                                 ; preds = %then0__1.i.i.i, %body__1.i.i.i
  %2 = add nuw nsw i64 %i1.i.i.i, 1
  %3 = icmp ult i64 %i1.i.i.i, 100
  br i1 %3, label %body__1.i.i.i, label %exiting__1.i.i

exiting__1.i.i:                                   ; preds = %exiting__1.i.i.i, %body__1.i.i
  %4 = add nuw nsw i64 %i1.i.i, 1
  %5 = icmp ult i64 %i1.i.i, 100
  br i1 %5, label %body__1.i.i, label %control_flow_nested__HelloQ__body.1.exit

control_flow_nested__HelloQ__body.1.exit:         ; preds = %exiting__1.i.i
  ret void
}

declare void @__quantum__rt__message(%String*) local_unnamed_addr

declare void @__quantum__rt__string_update_reference_count(%String*, i32) local_unnamed_addr

declare void @__quantum__qis__mz__body(%Qubit*, %Result*)

declare i1 @__quantum__qir__read_result(%Result*)

attributes #0 = { "EntryPoint" "requiredQubits"="1" }

