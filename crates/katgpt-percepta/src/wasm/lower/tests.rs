use super::*;

fn make_func(instrs: Vec<WasmInstr>) -> FuncBody {
    FuncBody {
        locals: vec![],
        num_locals: 0,
        instructions: instrs,
    }
}

fn make_func_with_locals(locals: Vec<(u32, u8)>, instrs: Vec<WasmInstr>) -> FuncBody {
    let num_locals = locals.iter().map(|(c, _)| *c).sum();
    FuncBody {
        locals,
        num_locals,
        instructions: instrs,
    }
}

#[test]
fn test_lower_mul_const() {
    // i32.const 3; i32.mul → should be lowered
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0), // x
        instr(OP_I32_CONST, 3), // C=3
        ni(OP_I32_MUL),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 1);
    // Should not contain MUL
    assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_MUL));
    // Should contain ADD (from multiplication expansion)
    assert!(lowered.instructions.iter().any(|i| i.opcode == OP_I32_ADD));
    // Should have extra temp locals
    assert_eq!(lowered.num_locals, NUM_TEMPS);
}

#[test]
fn test_lower_div_u_const() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_I32_CONST, 4),
        ni(OP_I32_DIV_U),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 1);
    assert!(
        lowered
            .instructions
            .iter()
            .all(|i| i.opcode != OP_I32_DIV_U)
    );
    assert!(lowered.instructions.iter().any(|i| i.opcode == OP_I32_SUB));
}

#[test]
fn test_lower_runtime_mul() {
    // MUL without preceding const → runtime loop
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(OP_I32_MUL),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 2);
    assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_MUL));
    assert!(lowered.instructions.iter().any(|i| i.opcode == OP_I32_ADD));
}

#[test]
fn test_lower_and_255() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_I32_CONST, 255),
        ni(OP_I32_AND),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 1);
    assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_AND));
    assert!(
        lowered
            .instructions
            .iter()
            .any(|i| i.opcode == OP_I32_STORE8)
    );
    assert!(
        lowered
            .instructions
            .iter()
            .any(|i| i.opcode == OP_I32_LOAD8_U)
    );
}

#[test]
fn test_lower_unary_clz() {
    let func = make_func(vec![instr(OP_LOCAL_GET, 0), ni(OP_I32_CLZ), ni(OP_END)]);
    let lowered = lower_hard_ops(&func, 1);
    assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_CLZ));
}

#[test]
fn test_no_lowering_needed() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(OP_I32_ADD),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 2);
    // Should be unchanged (cloned)
    assert_eq!(lowered.instructions.len(), func.instructions.len());
    assert_eq!(lowered.num_locals, 0);
}

#[test]
fn test_check_basic_only_pass() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        ni(OP_END),
    ]);
    let bad = check_basic_only(&func);
    assert!(bad.is_empty());
}

#[test]
fn test_check_basic_only_fail() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(OP_I32_MUL),
        ni(OP_END),
    ]);
    let bad = check_basic_only(&func);
    assert_eq!(bad.get("i32.mul"), Some(&1));
}

#[test]
fn test_check_basic_only_after_lowering() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_I32_CONST, 5),
        ni(OP_I32_MUL),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 1);
    let bad = check_basic_only(&lowered);
    assert!(bad.is_empty(), "unexpected hard ops: {bad:?}");
}

#[test]
fn test_find_const_locals() {
    let instrs = vec![
        instr(OP_I32_CONST, 42),
        instr(OP_LOCAL_SET, 1),
        instr(OP_I32_CONST, 42),
        instr(OP_LOCAL_SET, 1),
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_SET, 2),
    ];
    let cl = find_const_locals(&instrs);
    assert_eq!(cl.get(&1), Some(&42));
    assert!(!cl.contains_key(&2));
}

#[test]
fn test_lower_const_via_local_get() {
    // local.get 2 (which is always const 3) + i32.mul
    let func = make_func_with_locals(
        vec![(1, VALTYPE_I32)],
        vec![
            instr(OP_I32_CONST, 3),
            instr(OP_LOCAL_SET, 2),
            instr(OP_LOCAL_GET, 0),
            instr(OP_LOCAL_GET, 2),
            ni(OP_I32_MUL),
            ni(OP_END),
        ],
    );
    let lowered = lower_hard_ops(&func, 1);
    assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_MUL));
}

#[test]
fn test_lower_shl_const() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SHL),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 1);
    assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_SHL));
}

#[test]
fn test_lower_extend8_s() {
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        ni(OP_I32_EXTEND8_S),
        ni(OP_END),
    ]);
    let lowered = lower_hard_ops(&func, 1);
    assert!(
        lowered
            .instructions
            .iter()
            .all(|i| i.opcode != OP_I32_EXTEND8_S)
    );
    let bad = check_basic_only(&lowered);
    assert!(bad.is_empty(), "unexpected hard ops: {bad:?}");
}

// ── lower_i64_ops tests ─────────────────────────────────

#[test]
fn test_lower_i64_const() {
    // i64.const 42 → i32.const 42
    let func = make_func(vec![
        instr(0x42, 42i64), // i64.const
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    assert!(
        lowered.instructions.iter().all(|i| i.opcode != 0x42),
        "should not contain i64.const"
    );
    assert!(
        lowered
            .instructions
            .iter()
            .any(|i| i.opcode == OP_I32_CONST && i.immediates[0] == 42),
        "should contain i32.const 42"
    );
}

#[test]
fn test_lower_i64_const_truncates() {
    // i64.const with value > 32 bits → truncated to low 32 bits
    let func = make_func(vec![
        instr(0x42, 0x1_0000_0042i64), // i64.const (high bit set)
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    let const_val = lowered
        .instructions
        .iter()
        .find(|i| i.opcode == OP_I32_CONST)
        .map(|i| i.immediates[0])
        .unwrap();
    assert_eq!(const_val, 0x42, "should truncate to low 32 bits");
}

#[test]
fn test_lower_i64_add() {
    // i64.add → i32.add
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(0x7C), // i64.add
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    assert!(
        lowered.instructions.iter().all(|i| i.opcode != 0x7C),
        "should not contain i64.add"
    );
    assert!(
        lowered.instructions.iter().any(|i| i.opcode == OP_I32_ADD),
        "should contain i32.add"
    );
}

#[test]
fn test_lower_i64_mul_to_i32_mul() {
    // i64.mul → i32.mul (further lowered by lower_hard_ops)
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(0x7E), // i64.mul
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    assert!(
        lowered.instructions.iter().all(|i| i.opcode != 0x7E),
        "should not contain i64.mul"
    );
    assert!(
        lowered.instructions.iter().any(|i| i.opcode == OP_I32_MUL),
        "should contain i32.mul"
    );
}

#[test]
fn test_lower_i64_identity_ops() {
    // i32.wrap_i64 (0xA7), i64.extend_i32_s (0xAC), i64.extend_i32_u (0xAD) → removed
    for &op in &[0xA7, 0xAC, 0xAD] {
        let func = make_func(vec![instr(OP_LOCAL_GET, 0), ni(op), ni(OP_END)]);
        let lowered = lower_i64_ops(&func);
        assert!(
            lowered.instructions.iter().all(|i| i.opcode != op),
            "opcode 0x{op:02x} should be removed"
        );
        // Should have local.get + end (2 instructions, identity op removed)
        assert_eq!(
            lowered.instructions.len(),
            2,
            "opcode 0x{op:02x}: should have 2 instructions (local.get + end)"
        );
    }
}

#[test]
fn test_lower_i64_comparison_ops() {
    // i64.eq (0x51) → i32.eq
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(0x51), // i64.eq
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    assert!(
        lowered.instructions.iter().all(|i| i.opcode != 0x51),
        "should not contain i64.eq"
    );
    assert!(
        lowered.instructions.iter().any(|i| i.opcode == OP_I32_EQ),
        "should contain i32.eq"
    );
}

#[test]
fn test_lower_i64_store() {
    // i64.store (0x37) → i32.store (0x36), preserving alignment+offset immediates
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        WasmInstr::with_imms(0x37, vec![2, 0]), // i64.store align=2 offset=0
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    assert!(
        lowered.instructions.iter().all(|i| i.opcode != 0x37),
        "should not contain i64.store"
    );
    let store = lowered
        .instructions
        .iter()
        .find(|i| i.opcode == OP_I32_STORE)
        .expect("should contain i32.store");
    assert_eq!(
        store.immediates,
        vec![2, 0],
        "alignment+offset should be preserved"
    );
}

#[test]
fn test_lower_i64_noop_when_no_i64() {
    // Pure i32 code should be unchanged
    let func = make_func(vec![
        instr(OP_LOCAL_GET, 0),
        instr(OP_LOCAL_GET, 1),
        ni(OP_I32_ADD),
        ni(OP_END),
    ]);
    let lowered = lower_i64_ops(&func);
    assert_eq!(
        lowered.instructions.len(),
        func.instructions.len(),
        "should be unchanged"
    );
}
