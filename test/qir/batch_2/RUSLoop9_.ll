; ModuleID = 'RUSLoop9.bc'
source_filename = "qat-link"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [6 x i8] c"0_t0r\00"
@1 = internal constant [6 x i8] c"1_t1r\00"
@2 = internal constant [6 x i8] c"2_t2r\00"

define void @Microsoft__Quantum__Samples__RepeatUntilSuccess__RepeatUntilSuccess() #0 {
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
  br i1 %0, label %then0__4, label %continue__7

then0__4:                                         ; preds = %entry
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__7

continue__7:                                      ; preds = %then0__4, %entry
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %1 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %1, label %else__1, label %then0__5

then0__5:                                         ; preds = %continue__7
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %2 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  br i1 %2, label %then0__7, label %continue__13

then0__7:                                         ; preds = %then0__5
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__13

continue__13:                                     ; preds = %then0__7, %then0__5
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %3 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  %4 = xor i1 %3, true
  br i1 %3, label %then0__8, label %continue__9

then0__8:                                         ; preds = %continue__13
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__9

else__1:                                          ; preds = %continue__7
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__9

continue__9:                                      ; preds = %else__1, %then0__8, %continue__13
  %result2.0 = phi i1 [ false, %else__1 ], [ %4, %then0__8 ], [ %4, %continue__13 ]
  %5 = xor i1 %result2.0, true
  %6 = select i1 %1, i1 true, i1 %5
  br i1 %6, label %then0__9, label %continue__16

then0__9:                                         ; preds = %continue__9
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 2 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %7 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
  br i1 %7, label %then0__11, label %continue__20

then0__11:                                        ; preds = %then0__9
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__20

continue__20:                                     ; preds = %then0__11, %then0__9
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %8 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
  br i1 %8, label %else__2, label %then0__12

then0__12:                                        ; preds = %continue__20
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 3 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %9 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
  br i1 %9, label %then0__14, label %continue__26

then0__14:                                        ; preds = %then0__12
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__26

continue__26:                                     ; preds = %then0__14, %then0__12
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %10 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
  %11 = xor i1 %10, true
  br i1 %10, label %then0__15, label %continue__16

then0__15:                                        ; preds = %continue__26
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__16

else__2:                                          ; preds = %continue__20
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__16

continue__16:                                     ; preds = %else__2, %then0__15, %continue__26, %continue__9
  %result1.0.in = phi i1 [ %1, %continue__9 ], [ %8, %continue__26 ], [ %8, %then0__15 ], [ %8, %else__2 ]
  %result2.2 = phi i1 [ %result2.0, %continue__9 ], [ %11, %continue__26 ], [ %11, %then0__15 ], [ %result2.0, %else__2 ]
  %result1.0 = xor i1 %result1.0.in, true
  %12 = select i1 %result1.0, i1 %result2.2, i1 false
  br i1 %12, label %continue__29, label %then0__16

then0__16:                                        ; preds = %continue__16
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 4 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %13 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  br i1 %13, label %then0__18, label %continue__33

then0__18:                                        ; preds = %then0__16
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__33

continue__33:                                     ; preds = %then0__18, %then0__16
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %14 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  %15 = xor i1 %14, true
  br i1 %14, label %else__3, label %then0__19

then0__19:                                        ; preds = %continue__33
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 5 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %16 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 5 to %Result*))
  br i1 %16, label %then0__21, label %continue__39

then0__21:                                        ; preds = %then0__19
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__39

continue__39:                                     ; preds = %then0__21, %then0__19
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %17 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 5 to %Result*))
  %18 = xor i1 %17, true
  br i1 %17, label %then0__22, label %continue__29

then0__22:                                        ; preds = %continue__39
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__29

else__3:                                          ; preds = %continue__33
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__29

continue__29:                                     ; preds = %else__3, %then0__22, %continue__39, %continue__16
  %result1.1 = phi i1 [ %result1.0, %continue__16 ], [ %15, %continue__39 ], [ %15, %then0__22 ], [ %15, %else__3 ]
  %result2.4 = phi i1 [ %result2.2, %continue__16 ], [ %18, %continue__39 ], [ %18, %then0__22 ], [ %result2.2, %else__3 ]
  %19 = select i1 %result1.1, i1 %result2.4, i1 false
  br i1 %19, label %continue__42, label %then0__23

then0__23:                                        ; preds = %continue__29
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 6 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %20 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 6 to %Result*))
  br i1 %20, label %then0__25, label %continue__46

then0__25:                                        ; preds = %then0__23
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__46

continue__46:                                     ; preds = %then0__25, %then0__23
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %21 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 6 to %Result*))
  %22 = xor i1 %21, true
  br i1 %21, label %else__4, label %then0__26

then0__26:                                        ; preds = %continue__46
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 7 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %23 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 7 to %Result*))
  br i1 %23, label %then0__28, label %continue__52

then0__28:                                        ; preds = %then0__26
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__52

continue__52:                                     ; preds = %then0__28, %then0__26
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %24 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 7 to %Result*))
  %25 = xor i1 %24, true
  br i1 %24, label %then0__29, label %continue__42

then0__29:                                        ; preds = %continue__52
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__42

else__4:                                          ; preds = %continue__46
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__42

continue__42:                                     ; preds = %else__4, %then0__29, %continue__52, %continue__29
  %result1.2 = phi i1 [ %result1.1, %continue__29 ], [ %22, %continue__52 ], [ %22, %then0__29 ], [ %22, %else__4 ]
  %result2.6 = phi i1 [ %result2.4, %continue__29 ], [ %25, %continue__52 ], [ %25, %then0__29 ], [ %result2.4, %else__4 ]
  %26 = select i1 %result1.2, i1 %result2.6, i1 false
  br i1 %26, label %continue__55, label %then0__30

then0__30:                                        ; preds = %continue__42
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 8 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %27 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 8 to %Result*))
  br i1 %27, label %then0__32, label %continue__59

then0__32:                                        ; preds = %then0__30
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__59

continue__59:                                     ; preds = %then0__32, %then0__30
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %28 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 8 to %Result*))
  %29 = xor i1 %28, true
  br i1 %28, label %else__5, label %then0__33

then0__33:                                        ; preds = %continue__59
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 9 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %30 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 9 to %Result*))
  br i1 %30, label %then0__35, label %continue__65

then0__35:                                        ; preds = %then0__33
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__65

continue__65:                                     ; preds = %then0__35, %then0__33
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %31 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 9 to %Result*))
  %32 = xor i1 %31, true
  br i1 %31, label %then0__36, label %continue__55

then0__36:                                        ; preds = %continue__65
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__55

else__5:                                          ; preds = %continue__59
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__55

continue__55:                                     ; preds = %else__5, %then0__36, %continue__65, %continue__42
  %result1.3 = phi i1 [ %result1.2, %continue__42 ], [ %29, %continue__65 ], [ %29, %then0__36 ], [ %29, %else__5 ]
  %result2.8 = phi i1 [ %result2.6, %continue__42 ], [ %32, %continue__65 ], [ %32, %then0__36 ], [ %result2.6, %else__5 ]
  %33 = select i1 %result1.3, i1 %result2.8, i1 false
  br i1 %33, label %continue__68, label %then0__37

then0__37:                                        ; preds = %continue__55
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 10 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %34 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 10 to %Result*))
  br i1 %34, label %then0__39, label %continue__72

then0__39:                                        ; preds = %then0__37
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__72

continue__72:                                     ; preds = %then0__39, %then0__37
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %35 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 10 to %Result*))
  %36 = xor i1 %35, true
  br i1 %35, label %else__6, label %then0__40

then0__40:                                        ; preds = %continue__72
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 11 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %37 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 11 to %Result*))
  br i1 %37, label %then0__42, label %continue__78

then0__42:                                        ; preds = %then0__40
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__78

continue__78:                                     ; preds = %then0__42, %then0__40
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %38 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 11 to %Result*))
  %39 = xor i1 %38, true
  br i1 %38, label %then0__43, label %continue__68

then0__43:                                        ; preds = %continue__78
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__68

else__6:                                          ; preds = %continue__72
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__68

continue__68:                                     ; preds = %else__6, %then0__43, %continue__78, %continue__55
  %result1.4 = phi i1 [ %result1.3, %continue__55 ], [ %36, %continue__78 ], [ %36, %then0__43 ], [ %36, %else__6 ]
  %result2.10 = phi i1 [ %result2.8, %continue__55 ], [ %39, %continue__78 ], [ %39, %then0__43 ], [ %result2.8, %else__6 ]
  %40 = select i1 %result1.4, i1 %result2.10, i1 false
  br i1 %40, label %continue__81, label %then0__44

then0__44:                                        ; preds = %continue__68
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 12 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %41 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 12 to %Result*))
  br i1 %41, label %then0__46, label %continue__85

then0__46:                                        ; preds = %then0__44
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__85

continue__85:                                     ; preds = %then0__46, %then0__44
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %42 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 12 to %Result*))
  %43 = xor i1 %42, true
  br i1 %42, label %else__7, label %then0__47

then0__47:                                        ; preds = %continue__85
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 13 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %44 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 13 to %Result*))
  br i1 %44, label %then0__49, label %continue__91

then0__49:                                        ; preds = %then0__47
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__91

continue__91:                                     ; preds = %then0__49, %then0__47
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %45 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 13 to %Result*))
  %46 = xor i1 %45, true
  br i1 %45, label %then0__50, label %continue__81

then0__50:                                        ; preds = %continue__91
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__81

else__7:                                          ; preds = %continue__85
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__81

continue__81:                                     ; preds = %else__7, %then0__50, %continue__91, %continue__68
  %result1.5 = phi i1 [ %result1.4, %continue__68 ], [ %43, %continue__91 ], [ %43, %then0__50 ], [ %43, %else__7 ]
  %result2.12 = phi i1 [ %result2.10, %continue__68 ], [ %46, %continue__91 ], [ %46, %then0__50 ], [ %result2.10, %else__7 ]
  %47 = select i1 %result1.5, i1 %result2.12, i1 false
  br i1 %47, label %continue__94, label %then0__51

then0__51:                                        ; preds = %continue__81
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 14 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %48 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 14 to %Result*))
  br i1 %48, label %then0__53, label %continue__98

then0__53:                                        ; preds = %then0__51
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__98

continue__98:                                     ; preds = %then0__53, %then0__51
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %49 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 14 to %Result*))
  %50 = xor i1 %49, true
  br i1 %49, label %else__8, label %then0__54

then0__54:                                        ; preds = %continue__98
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 15 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %51 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 15 to %Result*))
  br i1 %51, label %then0__56, label %continue__104

then0__56:                                        ; preds = %then0__54
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__104

continue__104:                                    ; preds = %then0__56, %then0__54
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %52 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 15 to %Result*))
  %53 = xor i1 %52, true
  br i1 %52, label %then0__57, label %continue__94

then0__57:                                        ; preds = %continue__104
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__94

else__8:                                          ; preds = %continue__98
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__94

continue__94:                                     ; preds = %else__8, %then0__57, %continue__104, %continue__81
  %result1.6 = phi i1 [ %result1.5, %continue__81 ], [ %50, %continue__104 ], [ %50, %then0__57 ], [ %50, %else__8 ]
  %result2.14 = phi i1 [ %result2.12, %continue__81 ], [ %53, %continue__104 ], [ %53, %then0__57 ], [ %result2.12, %else__8 ]
  %54 = select i1 %result1.6, i1 %result2.14, i1 false
  br i1 %54, label %continue__107, label %then0__58

then0__58:                                        ; preds = %continue__94
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 16 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %55 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 16 to %Result*))
  br i1 %55, label %then0__60, label %continue__111

then0__60:                                        ; preds = %then0__58
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__111

continue__111:                                    ; preds = %then0__60, %then0__58
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %56 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 16 to %Result*))
  br i1 %56, label %else__9, label %then0__61

then0__61:                                        ; preds = %continue__111
  call void @__quantum__qis__t__body(%Qubit* null)
  call void @__quantum__qis__z__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 17 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %57 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 17 to %Result*))
  br i1 %57, label %then0__63, label %continue__117

then0__63:                                        ; preds = %then0__61
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__117

continue__117:                                    ; preds = %then0__63, %then0__61
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %58 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 17 to %Result*))
  br i1 %58, label %then0__64, label %continue__107

then0__64:                                        ; preds = %continue__117
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__z__body(%Qubit* null)
  br label %continue__107

else__9:                                          ; preds = %continue__111
  call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__107

continue__107:                                    ; preds = %else__9, %then0__64, %continue__117, %continue__94
  call void @__quantum__qis__rz__body(double 0x4001B6E192EBBE42, %Qubit* null)
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 18 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 19 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 20 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* null)
  %59 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 20 to %Result*))
  br i1 %59, label %then0__66, label %continue__123

then0__66:                                        ; preds = %continue__107
  call void @__quantum__qis__x__body(%Qubit* null)
  br label %continue__123

continue__123:                                    ; preds = %then0__66, %continue__107
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__rt__tuple_start_record_output()
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 18 to %Result*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @0, i64 0, i64 0))
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 19 to %Result*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @1, i64 0, i64 0))
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 20 to %Result*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @2, i64 0, i64 0))
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

declare void @__quantum__rt__result_record_output(%Result*, i8*)

declare void @__quantum__rt__tuple_end_record_output()

declare void @__quantum__qis__mz__body(%Qubit*, %Result*)

declare i1 @__quantum__qis__read_result__body(%Result*)

attributes #0 = { "EntryPoint" "maxQubitIndex"="2" "maxResultIndex"="20" "requiredQubits"="3" "requiredResults"="21" }
