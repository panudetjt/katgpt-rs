//! Deterministic AST node-type histogram over Rust source (katgpt-rs Issue
//! 867 T1 — the non-hidden-state canonical construction, Proposal 010 §
//! Feature 1).
//!
//! The extractor half of the cross-arch reopen attempt: source features are
//! architecture-independent by construction (the AST does not know which
//! model will process the code), fixed-dimension (closed vocabulary), and
//! deterministic (same source → bit-identical histogram). This module is the
//! FIXTURE-side counterpart of the riir-train Issue 567 whole-item corpus —
//! every corpus item parses with `syn::parse_file` (generation-time gate),
//! so `ast_histogram` never returns `None` on corpus records by construction.
//!
//! # Vocabulary discipline
//!
//! `N_AST_BINS` and the bin order are FROZEN. Feature vectors computed with
//! this vocabulary (corpus histograms, cached adapters, committed fixtures)
//! are indexed by it; adding bins must APPEND (never reorder, never renumber)
//! so previously computed vectors stay addressable. The bin set is pure node
//! TYPE — no name-based bins (no "calls clone", no identifier strings), per
//! the proposal's structure-not-meaning scope.
//!
//! Setup-time only: `syn::parse_file` allocates and is never on a per-token
//! path (the `canon_source_features` feature's BOUNDARY.md condition).

use syn::visit::Visit;

/// Number of histogram bins. FROZEN vocabulary — append-only (module doc).
pub const N_AST_BINS: usize = 38;

/// A counted AST node-type histogram.
///
/// `counts` is indexed by [`AstBin`] discriminant — the deterministic,
/// ordered form the ridge adapter (Issue 867 T2) consumes directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AstHistogram {
    pub counts: [u32; N_AST_BINS],
}

impl AstHistogram {
    /// Total nodes counted across all bins.
    #[inline]
    pub fn total(&self) -> u64 {
        self.counts.iter().map(|&c| c as u64).sum()
    }

    /// L1-normalized feature vector (each bin as a fraction of the total).
    ///
    /// Scale-invariant form for cross-length comparison; an empty histogram
    /// (parse produced no counted nodes) normalizes to all zeros.
    pub fn normalized(&self) -> [f32; N_AST_BINS] {
        let total = self.total() as f32;
        if total == 0.0 {
            return [0.0; N_AST_BINS];
        }
        let mut out = [0.0f32; N_AST_BINS];
        for (o, &c) in out.iter_mut().zip(self.counts.iter()) {
            *o = c as f32 / total;
        }
        out
    }
}

/// The closed node-type vocabulary. Discriminants ARE the bin indices —
/// keep them contiguous and in order (0..N_AST_BINS), append-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AstBin {
    Fn,
    Impl,
    Trait,
    Struct,
    Enum,
    MatchExpr,
    MatchArm,
    If,
    IfLet,
    While,
    WhileLet,
    ForLoop,
    Loop,
    Closure,
    Async,
    Await,
    Macro,
    MethodCall,
    FieldAccess,
    Index,
    Binary,
    Unary,
    Reference,
    Cast,
    Try,
    Return,
    Literal,
    PathExpr,
    Tuple,
    Range,
    Assign,
    StructLit,
    GenericParam,
    Lifetime,
    WhereClause,
    PatternWild,
    PatternStruct,
    Unsafe,
}

impl AstBin {
    /// Increment this bin in a histogram.
    #[inline]
    fn bump(self, h: &mut AstHistogram) {
        h.counts[self as usize] += 1;
    }
}

/// Compute the AST node-type histogram of a Rust source string.
///
/// Returns `None` if the source does not parse as a Rust file (`syn` does
/// not resolve names or expand macros — an item referencing undefined types
/// still parses; the whole-item corpus is validated with this same function
/// at generation time, riir-train Issue 567 T1).
pub fn ast_histogram(source: &str) -> Option<AstHistogram> {
    let file = syn::parse_file(source).ok()?;
    let mut visitor = HistogramVisitor::new();
    visitor.visit_file(&file);
    Some(visitor.histogram)
}

struct HistogramVisitor {
    histogram: AstHistogram,
}

impl HistogramVisitor {
    fn new() -> Self {
        Self {
            histogram: AstHistogram {
                counts: [0; N_AST_BINS],
            },
        }
    }

    #[inline]
    fn bump(&mut self, bin: AstBin) {
        bin.bump(&mut self.histogram);
    }
}

impl<'ast> Visit<'ast> for HistogramVisitor {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.bump(AstBin::Fn);
        if i.sig.asyncness.is_some() {
            self.bump(AstBin::Async);
        }
        // Walk the body so nested items/expressions are counted too.
        syn::visit::visit_item_fn(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        self.bump(AstBin::Impl);
        syn::visit::visit_item_impl(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.bump(AstBin::Trait);
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.bump(AstBin::Struct);
        syn::visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.bump(AstBin::Enum);
        syn::visit::visit_item_enum(self, i);
    }

    fn visit_expr_match(&mut self, i: &'ast syn::ExprMatch) {
        self.bump(AstBin::MatchExpr);
        syn::visit::visit_expr_match(self, i);
    }

    fn visit_arm(&mut self, i: &'ast syn::Arm) {
        self.bump(AstBin::MatchArm);
        syn::visit::visit_arm(self, i);
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        self.bump(AstBin::If);
        // `if let PAT = expr` is an ExprIf whose cond is an ExprLet (syn 2);
        // the ExprLet visit below records the IfLet bin.
        syn::visit::visit_expr_if(self, i);
    }

    fn visit_expr_let(&mut self, i: &'ast syn::ExprLet) {
        self.bump(AstBin::IfLet);
        syn::visit::visit_expr_let(self, i);
    }

    fn visit_expr_while(&mut self, i: &'ast syn::ExprWhile) {
        self.bump(AstBin::While);
        // syn 2 spells `while let PAT = cond` as ExprWhile whose cond is an
        // ExprLet — same node, one extra bin for the let-form.
        if matches!(*i.cond, syn::Expr::Let(_)) {
            self.bump(AstBin::WhileLet);
        }
        syn::visit::visit_expr_while(self, i);
    }

    fn visit_expr_for_loop(&mut self, i: &'ast syn::ExprForLoop) {
        self.bump(AstBin::ForLoop);
        syn::visit::visit_expr_for_loop(self, i);
    }

    fn visit_expr_loop(&mut self, i: &'ast syn::ExprLoop) {
        self.bump(AstBin::Loop);
        syn::visit::visit_expr_loop(self, i);
    }

    fn visit_expr_closure(&mut self, i: &'ast syn::ExprClosure) {
        self.bump(AstBin::Closure);
        syn::visit::visit_expr_closure(self, i);
    }

    fn visit_expr_async(&mut self, i: &'ast syn::ExprAsync) {
        self.bump(AstBin::Async);
        syn::visit::visit_expr_async(self, i);
    }

    fn visit_expr_await(&mut self, i: &'ast syn::ExprAwait) {
        self.bump(AstBin::Await);
        syn::visit::visit_expr_await(self, i);
    }

    fn visit_expr_macro(&mut self, i: &'ast syn::ExprMacro) {
        self.bump(AstBin::Macro);
        syn::visit::visit_expr_macro(self, i);
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        self.bump(AstBin::Macro);
        syn::visit::visit_item_macro(self, i);
    }

    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        self.bump(AstBin::MethodCall);
        syn::visit::visit_expr_method_call(self, i);
    }

    fn visit_expr_field(&mut self, i: &'ast syn::ExprField) {
        self.bump(AstBin::FieldAccess);
        syn::visit::visit_expr_field(self, i);
    }

    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        self.bump(AstBin::Index);
        syn::visit::visit_expr_index(self, i);
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        self.bump(AstBin::Binary);
        syn::visit::visit_expr_binary(self, i);
    }

    fn visit_expr_unary(&mut self, i: &'ast syn::ExprUnary) {
        self.bump(AstBin::Unary);
        syn::visit::visit_expr_unary(self, i);
    }

    fn visit_expr_reference(&mut self, i: &'ast syn::ExprReference) {
        self.bump(AstBin::Reference);
        syn::visit::visit_expr_reference(self, i);
    }

    fn visit_expr_cast(&mut self, i: &'ast syn::ExprCast) {
        self.bump(AstBin::Cast);
        syn::visit::visit_expr_cast(self, i);
    }

    fn visit_expr_try(&mut self, i: &'ast syn::ExprTry) {
        self.bump(AstBin::Try);
        syn::visit::visit_expr_try(self, i);
    }

    fn visit_expr_return(&mut self, i: &'ast syn::ExprReturn) {
        self.bump(AstBin::Return);
        syn::visit::visit_expr_return(self, i);
    }

    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        self.bump(AstBin::Literal);
        syn::visit::visit_expr_lit(self, i);
    }

    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        self.bump(AstBin::PathExpr);
        syn::visit::visit_expr_path(self, i);
    }

    fn visit_expr_tuple(&mut self, i: &'ast syn::ExprTuple) {
        self.bump(AstBin::Tuple);
        syn::visit::visit_expr_tuple(self, i);
    }

    fn visit_expr_range(&mut self, i: &'ast syn::ExprRange) {
        self.bump(AstBin::Range);
        syn::visit::visit_expr_range(self, i);
    }

    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        self.bump(AstBin::Assign);
        syn::visit::visit_expr_assign(self, i);
    }

    fn visit_expr_struct(&mut self, i: &'ast syn::ExprStruct) {
        self.bump(AstBin::StructLit);
        syn::visit::visit_expr_struct(self, i);
    }

    fn visit_generic_param(&mut self, i: &'ast syn::GenericParam) {
        self.bump(AstBin::GenericParam);
        syn::visit::visit_generic_param(self, i);
    }

    fn visit_lifetime(&mut self, i: &'ast syn::Lifetime) {
        self.bump(AstBin::Lifetime);
        syn::visit::visit_lifetime(self, i);
    }

    fn visit_where_clause(&mut self, i: &'ast syn::WhereClause) {
        self.bump(AstBin::WhereClause);
        syn::visit::visit_where_clause(self, i);
    }

    fn visit_pat_wild(&mut self, i: &'ast syn::PatWild) {
        self.bump(AstBin::PatternWild);
        syn::visit::visit_pat_wild(self, i);
    }

    fn visit_pat_struct(&mut self, i: &'ast syn::PatStruct) {
        self.bump(AstBin::PatternStruct);
        syn::visit::visit_pat_struct(self, i);
    }

    fn visit_expr_unsafe(&mut self, i: &'ast syn::ExprUnsafe) {
        self.bump(AstBin::Unsafe);
        syn::visit::visit_expr_unsafe(self, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_of(h: &AstHistogram, bin: AstBin) -> u32 {
        h.counts[bin as usize]
    }

    #[test]
    fn iterator_chain_vs_index_loop_separates() {
        let idiomatic = ast_histogram(
            "fn total(prices: &[f64]) -> f64 {\n    prices.iter().filter(|p| **p > 0.0).sum()\n}",
        )
        .expect("parses");
        let loop_style = ast_histogram(
            "fn total(prices: &[f64]) -> f64 {\n    let mut t = 0.0;\n    for i in 0..prices.len() {\n        if prices[i] > 0.0 {\n            t += prices[i];\n        }\n    }\n    t\n}",
        )
        .expect("parses");

        assert!(
            bin_of(&idiomatic, AstBin::MethodCall) > bin_of(&loop_style, AstBin::MethodCall),
            "iterator chain must carry more MethodCall nodes"
        );
        assert!(
            bin_of(&loop_style, AstBin::ForLoop) > bin_of(&idiomatic, AstBin::ForLoop),
            "index loop must carry more ForLoop nodes"
        );
        assert!(
            bin_of(&loop_style, AstBin::Index) >= bin_of(&idiomatic, AstBin::Index)
                && bin_of(&loop_style, AstBin::Index) > 0,
            "index loop must carry Index nodes"
        );
    }

    #[test]
    fn try_propagation_vs_unwrap_separates() {
        let idiomatic = ast_histogram(
            "fn load(raw: &str) -> Result<u32, ConfigError> {\n    let n: u32 = raw.trim().parse()?;\n    Ok(n)\n}",
        )
        .expect("parses");
        let unwrap_style = ast_histogram(
            "fn load(raw: &str) -> u32 {\n    let n = raw.trim().parse::<u32>().unwrap();\n    n\n}",
        )
        .expect("parses");

        assert_eq!(bin_of(&idiomatic, AstBin::Try), 1, "`?` counted once");
        assert_eq!(bin_of(&unwrap_style, AstBin::Try), 0, "unwrap is not `?`");
        assert!(
            bin_of(&unwrap_style, AstBin::MethodCall) >= bin_of(&idiomatic, AstBin::MethodCall),
            "unwrap chain is method calls"
        );
    }

    #[test]
    fn deterministic_bit_identical() {
        let src = "trait S { fn area(&self) -> f64; }\nimpl S for C { fn area(&self) -> f64 { 3.0 * self.r } }\n";
        let a = ast_histogram(src).expect("parses");
        let b = ast_histogram(src).expect("parses");
        assert_eq!(a.counts, b.counts, "same source must be bit-identical");
    }

    #[test]
    fn parse_failure_is_none() {
        assert!(ast_histogram("fn {").is_none());
        assert!(ast_histogram("").is_some(), "empty file parses");
    }

    #[test]
    fn normalized_sums_to_one_or_zero() {
        let h = ast_histogram("fn a() -> u8 { 1 }").expect("parses");
        let sum: f32 = h.normalized().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "normalized must sum to 1, got {sum}"
        );
        let empty = AstHistogram {
            counts: [0; N_AST_BINS],
        };
        assert!(
            empty.normalized().iter().all(|&v| v == 0.0),
            "empty must normalize to zeros"
        );
    }

    #[test]
    fn vocabulary_is_append_only_by_construction() {
        // Discriminants must stay contiguous 0..N_AST_BINS — the bin ORDER is
        // the feature-vector contract (module doc). This arm trips if a bin
        // is inserted or reordered.
        let order = [
            AstBin::Fn as u8,
            AstBin::Impl as u8,
            AstBin::Trait as u8,
            AstBin::Struct as u8,
            AstBin::Enum as u8,
            AstBin::MatchExpr as u8,
            AstBin::MatchArm as u8,
            AstBin::If as u8,
            AstBin::IfLet as u8,
            AstBin::While as u8,
            AstBin::WhileLet as u8,
            AstBin::ForLoop as u8,
            AstBin::Loop as u8,
            AstBin::Closure as u8,
            AstBin::Async as u8,
            AstBin::Await as u8,
            AstBin::Macro as u8,
            AstBin::MethodCall as u8,
            AstBin::FieldAccess as u8,
            AstBin::Index as u8,
            AstBin::Binary as u8,
            AstBin::Unary as u8,
            AstBin::Reference as u8,
            AstBin::Cast as u8,
            AstBin::Try as u8,
            AstBin::Return as u8,
            AstBin::Literal as u8,
            AstBin::PathExpr as u8,
            AstBin::Tuple as u8,
            AstBin::Range as u8,
            AstBin::Assign as u8,
            AstBin::StructLit as u8,
            AstBin::GenericParam as u8,
            AstBin::Lifetime as u8,
            AstBin::WhereClause as u8,
            AstBin::PatternWild as u8,
            AstBin::PatternStruct as u8,
            AstBin::Unsafe as u8,
        ];
        for (i, &b) in order.iter().enumerate() {
            assert_eq!(b as usize, i, "AstBin discriminant drift at index {i}");
        }
        assert_eq!(order.len(), N_AST_BINS);
    }
}
