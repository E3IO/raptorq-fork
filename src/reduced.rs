//! Dimension-reduction ("incremental") decoding for RaptorQ.
//!
//! Standard RaptorQ decoding solves for all L = K' + S + H intermediate
//! symbols from the full constraint matrix. When the loss rate is low, the
//! receiver already holds most source symbols: their values are *known*, and
//! the only unknowns are the L_lost missing source symbols plus the LDPC/HDPC
//! symbols that are linear functions of the received ones.
//!
//! This module exploits that structure:
//!
//! 1. The received source symbols plus the implicit zero padding symbols give
//!    K' - L_lost linear constraints on the LT/PI intermediate symbols X.
//! 2. Sparse forward elimination on those constraints reduces X to L_lost
//!    free variables (the "reduced" unknown vector alpha).
//! 3. The repair symbols project onto the free variables, yielding a small
//!    L_lost x L_lost system solved with an O(L_lost^3) Gaussian elimination
//!    instead of the full O((K'+S+H)^2) symbol-level PI solve.
//! 4. Missing source symbols are rebuilt from the free variables.
//!
//! Cost scales with the *number of lost symbols*, not the block size: for a
//! 1% loss on a K = 1000 symbol block only ~10 symbols are rebuilt. The
//! constraint elimination is sparse (each LT row touches ~30 columns), so
//! wall time stays far below the standard decode. Blocks with few symbols
//! (K' < 32) are dominated by fixed overhead and should keep the standard
//! path; callers gate the fast path with [`REDUCED_LOST_THRESHOLD`] and
//! [`REDUCED_MIN_SYMBOLS`].

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::base::{intermediate_tuple, EncodingPacket};
use crate::constraint_matrix::{enc_indices, generate_hdpc_rows};
use crate::octet::Octet;
use crate::symbol::Symbol;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_hdpc_symbols, num_ldpc_symbols,
    num_lt_symbols, num_pi_symbols, systematic_index,
};

/// Maximum number of lost symbols handled by the reduced fast path.
/// Beyond this the small-system solve is no cheaper than the standard
/// structured PI solver.
pub const REDUCED_LOST_THRESHOLD: u32 = 16;

/// Blocks smaller than this keep the standard decode path: the fixed costs of
/// the sparse elimination dominate at tiny K'.
pub const REDUCED_MIN_SYMBOLS: u32 = 24;

/// One row of the sparse constraint matrix: (column, GF(2^8) coefficient)
/// pairs sorted by column, plus the RHS symbol accumulated by row operations.
#[derive(Clone)]
struct SparseRow {
    cols: Vec<(u32, u8)>,
    rhs: Symbol,
}

impl SparseRow {
    fn new(cols: Vec<(u32, u8)>, rhs: Symbol) -> Self {
        let mut row = SparseRow { cols, rhs };
        row.cols.retain(|&(_, c)| c != 0);
        row.cols.sort_unstable_by_key(|&(col, _)| col);
        row
    }

    fn min_col(&self) -> Option<u32> {
        self.cols.first().map(|&(col, _)| col)
    }

    fn coeff_at(&self, col: u32) -> u8 {
        match self.cols.binary_search_by_key(&col, |&(c, _)| c) {
            Ok(i) => self.cols[i].1,
            Err(_) => 0,
        }
    }

    /// row += scalar * other. The RHS symbols are combined with the same
    /// scalar so the equation set stays equivalent.
    fn fma(&mut self, other: &SparseRow, scalar: &Octet) {
        if *scalar == Octet::zero() {
            return;
        }
        let mut merged: Vec<(u32, u8)> = Vec::with_capacity(self.cols.len() + other.cols.len());
        let (mut i, mut j) = (0, 0);
        while i < self.cols.len() && j < other.cols.len() {
            let (ca, va) = self.cols[i];
            let (cb, vb) = other.cols[j];
            if ca < cb {
                merged.push((ca, va));
                i += 1;
            } else if cb < ca {
                merged.push((cb, (scalar.clone() * Octet::new(vb)).byte()));
                j += 1;
            } else {
                let v = va ^ (scalar.clone() * Octet::new(vb)).byte();
                if v != 0 {
                    merged.push((ca, v));
                }
                i += 1;
                j += 1;
            }
        }
        while i < self.cols.len() {
            merged.push(self.cols[i]);
            i += 1;
        }
        while j < other.cols.len() {
            let (cb, vb) = other.cols[j];
            merged.push((cb, (scalar.clone() * Octet::new(vb)).byte()));
            j += 1;
        }
        self.cols = merged;
        if *scalar == Octet::one() {
            self.rhs += &other.rhs;
        } else {
            self.rhs
                .fused_addassign_mul_scalar(&other.rhs, scalar);
        }
    }

    /// Normalize the row so the coefficient at `pivot_col` is 1.
    fn normalize_pivot(&mut self, pivot_col: u32) -> Option<()> {
        let coeff = self.coeff_at(pivot_col);
        if coeff == 0 {
            return None;
        }
        let inv = Octet::one() / Octet::new(coeff);
        for (_, c) in self.cols.iter_mut() {
            *c = (inv.clone() * Octet::new(*c)).byte();
        }
        self.rhs.mulassign_scalar(&inv);
        Some(())
    }
}

fn lt_row(
    esi: u32,
    lt_symbols: u32,
    pi_symbols: u32,
    sys_index: u32,
    p1: u32,
) -> Vec<(u32, u8)> {
    let tuple = intermediate_tuple(esi, lt_symbols, sys_index, p1);
    enc_indices(tuple, lt_symbols, pi_symbols, p1)
        .into_iter()
        .map(|col| (col as u32, 1u8))
        .collect()
}

/// Eliminate every column of `row` that has a pivot row, using the normalized
/// pivot rows (each pivot column appears only in its own pivot row).
fn eliminate_pivots(row: &mut SparseRow, pivots: &[SparseRow], is_free: &[bool]) {
    loop {
        let j = row
            .cols
            .iter()
            .map(|&(c, _)| c)
            .find(|&c| !is_free[c as usize]);
        let Some(j) = j else {
            break;
        };
        let idx = pivots
            .iter()
            .position(|p| p.min_col() == Some(j))
            .expect("non-free column must have a pivot");
        let coeff = row.coeff_at(j);
        let scalar = Octet::new(coeff) / Octet::new(pivots[idx].coeff_at(j));
        let p = pivots[idx].clone();
        row.fma(&p, &scalar);
    }
}

/// Attempt the reduced fast-path decode.
///
/// Returns `Some(bytes)` on success, or `None` when the fast path does not
/// apply (too many lost symbols, too few repair symbols, singular system) and
/// the caller must fall back to the standard decode.
#[allow(clippy::too_many_arguments)]
pub fn try_reduced_decode(
    source_block_symbols: u32,
    symbol_size: u16,
    num_sub_blocks: u16,
    symbol_alignment: u8,
    source_symbols: &[Option<Symbol>],
    repair_packets: &[EncodingPacket],
) -> Option<Vec<u8>> {
    let k = source_block_symbols;
    let kprime = extended_source_block_symbols(k);
    let s = num_ldpc_symbols(k) as usize;
    let h = num_hdpc_symbols(k) as usize;
    let lt_symbols = num_lt_symbols(kprime);
    let pi_symbols = num_pi_symbols(kprime);
    let sys_index = systematic_index(kprime);
    let p1 = calculate_p1(kprime);
    // The full intermediate symbol space is L = K' + S + H columns. LT rows
    // (enc_indices) touch all of them: the "PI" region spans [W .. L).
    let l_total = kprime as usize + s + h;
    let w = lt_symbols as usize;
    let b = w - s;
    let p = l_total - w; // PI region width (== num_pi_symbols(K'))

    // Lost source symbol indices (in ESI order).
    let mut lost: Vec<u32> = Vec::with_capacity(source_symbols.len());
    for (esi, sym) in source_symbols.iter().enumerate() {
        if sym.is_none() {
            lost.push(esi as u32);
        }
    }
    let lost_count = lost.len() as u32;
    if lost_count == 0
        || lost_count > REDUCED_LOST_THRESHOLD
        || k < REDUCED_MIN_SYMBOLS
        || repair_packets.len() < lost_count as usize
    {
        return None;
    }

    let zero = Symbol::zero(symbol_size as usize);

    // 1. Build constraint rows. Received source symbols (RHS = their value),
    //    the K' - K padding symbols (RHS = zero), the S LDPC rows and the H
    //    HDPC rows (RHS = zero). Together these K' - L_lost + S + H rows
    //    reduce the L_total-dimensional intermediate space to L_lost free
    //    columns.
    let mut pivot_rows: Vec<SparseRow> = Vec::with_capacity(l_total);
    for (esi, sym) in source_symbols.iter().enumerate() {
        if let Some(sym) = sym {
            let cols = lt_row(esi as u32, lt_symbols, pi_symbols, sys_index, p1);
            pivot_rows.push(SparseRow::new(cols, sym.clone()));
        }
    }
    for esi in k..kprime {
        let cols = lt_row(esi, lt_symbols, pi_symbols, sys_index, p1);
        if !cols.is_empty() {
            pivot_rows.push(SparseRow::new(cols, zero.clone()));
        }
    }
    // G_LDPC,1 + I_S + G_LDPC,2 rows (section 5.3.3.3): S rows total, each
    // touching 1 + (degree-3 LDPC1 columns) + 2 LDPC2 columns.
    let mut ldpc_cols: Vec<Vec<(u32, u8)>> = vec![Vec::new(); s];
    for i in 0..b {
        let a = 1 + i / s;
        let bm = i % s;
        ldpc_cols[bm].push((i as u32, 1));
        ldpc_cols[(bm + a) % s].push((i as u32, 1));
        ldpc_cols[(bm + 2 * a) % s].push((i as u32, 1));
    }
    for i in 0..s {
        ldpc_cols[i].push(((i + b) as u32, 1));
        ldpc_cols[i].push((((i % p) + w) as u32, 1));
        ldpc_cols[i].push(((((i + 1) % p) + w) as u32, 1));
        pivot_rows.push(SparseRow::new(std::mem::take(&mut ldpc_cols[i]), zero.clone()));
    }
    // HDPC rows (G_HDPC + I_H).
    let hdpc = generate_hdpc_rows(kprime as usize, s as usize, h as usize);
    for i in 0..h {
        let mut cols = Vec::with_capacity(kprime as usize + s + 1);
        for j in 0..l_total {
            let v = hdpc.get(i, j);
            if v != Octet::zero() {
                cols.push((j as u32, v.byte()));
            }
        }
        if !cols.is_empty() {
            pivot_rows.push(SparseRow::new(cols, zero.clone()));
        }
    }

    // 2. Sparse forward elimination: build a set of pivot rows, one per
    //    non-free column. Free columns are those that never became a pivot.
    let mut pivots: Vec<SparseRow> = Vec::new();
    let mut is_free: Vec<bool> = vec![true; l_total];
    for mut row in pivot_rows {
        loop {
            let Some(j) = row.min_col() else {
                break; // zero row (linearly dependent constraint)
            };
            if let Some(idx) = pivots.iter().position(|p| p.min_col() == Some(j)) {
                let coeff = row.coeff_at(j);
                let scalar = Octet::new(coeff) / Octet::new(pivots[idx].coeff_at(j));
                let p = pivots[idx].clone();
                row.fma(&p, &scalar);
            } else {
                row.normalize_pivot(j)?;
                is_free[j as usize] = false;
                pivots.push(row);
                break;
            }
        }
    }

    // 3. Project repair symbols onto the free variables.
    //    Each repair row, after eliminating pivoted columns, yields
    //    sum_k coef[k] * alpha[k] = rhs  with alpha[k] = X[free column k].
    let free_cols: Vec<u32> = (0..l_total)
        .filter(|&c| is_free[c as usize])
        .map(|c| c as u32)
        .collect();
    if free_cols.len() != lost_count as usize {

        return None; // rank deficient constraints
    }
    let free_pos: Vec<usize> = free_cols.iter().map(|&c| c as usize).collect();

    let mut equations: Vec<SparseRow> = Vec::with_capacity(repair_packets.len());

    let num_padding_symbols = kprime - k;
    for packet in repair_packets {
        // Repair symbols live in a separate ISI space: their encoding symbol
        // ID must be offset by the padding symbols (same as the standard
        // decode in decoder.rs Case 3).
        let esi = packet.payload_id.encoding_symbol_id() + num_padding_symbols;
        let mut row = SparseRow::new(
            lt_row(esi, lt_symbols, pi_symbols, sys_index, p1),
            Symbol::new(packet.data.clone()),
        );
        // Eliminate ALL pivoted columns (they may not be the leading ones).
        eliminate_pivots(&mut row, &pivots, &is_free);
        // Keep only free columns (coeffs).
        row.cols.retain(|&(c, _)| is_free[c as usize]);
        equations.push(row);
    }
    if equations.len() < lost_count as usize {

        return None;
    }

    // 4. Solve the reduced system for alpha. Take the first lost_count
    //    equations and Gaussian-eliminate them (symbol RHS).
    let mut system: Vec<SparseRow> = equations.drain(..lost_count as usize).collect();
    for i in 0..lost_count as usize {
        // Find a pivot row for equation i among remaining rows.
        let col_i = free_pos[i];
        let mut pivot = None;
        for r in i..system.len() {
            if system[r].coeff_at(col_i as u32) != 0 {
                pivot = Some(r);
                break;
            }
        }
        let Some(pr) = pivot else {
    
            return None; // singular reduced system
        };
        system.swap(i, pr);
        system[i].normalize_pivot(col_i as u32)?;
        let pi_row = system[i].clone();
        for r in 0..system.len() {
            if r == i {
                continue;
            }
            let coeff = system[r].coeff_at(col_i as u32);
            if coeff != 0 {
                let scalar = Octet::new(coeff);
                system[r].fma(&pi_row, &scalar);
            }
        }
    }

    // alpha[k] = X[free_cols[k]] is now directly the RHS of row k.
    let mut alpha: Vec<Symbol> = Vec::with_capacity(lost_count as usize);
    for i in 0..lost_count as usize {
        alpha.push(system[i].rhs.clone());
    }

    // 5. Rebuild lost source symbols: D_lost = LT row . X, with pivoted
    //    columns eliminated so only free variables remain.
    let mut rebuilt: Vec<(u32, Symbol)> = Vec::with_capacity(lost_count as usize);
    for &esi in &lost {
        let mut row = SparseRow::new(
            lt_row(esi, lt_symbols, pi_symbols, sys_index, p1),
            Symbol::zero(symbol_size as usize),
        );
        eliminate_pivots(&mut row, &pivots, &is_free);
        // D_lost = rhs_accum + sum_k coef[k] * alpha[k]
        let mut symbol = row.rhs;
        for &(col, coeff) in row.cols.iter() {
            if let Ok(k) = free_pos.binary_search(&(col as usize)) {
                if coeff == 1 {
                    symbol += &alpha[k];
                } else if coeff != 0 {
                    symbol.fused_addassign_mul_scalar(&alpha[k], &Octet::new(coeff));
                }
            }
        }
        rebuilt.push((esi, symbol));
    }

    // 6. Assemble the block in sub-block layout (same as the standard path).
    let (tl, ts, nl, ns) = crate::base::partition(
        (symbol_size / symbol_alignment as u16) as u32,
        num_sub_blocks,
    );
    let mut result = vec![0; symbol_size as usize * k as usize];
    let mut rebuilt_iter = rebuilt.into_iter().peekable();
    for i in 0..k as usize {
        let symbol = if let Some(Some(sym)) = source_symbols.get(i) {
            Some(sym.clone())
        } else if rebuilt_iter.peek().is_some_and(|(esi, _)| *esi == i as u32) {
            rebuilt_iter.next().map(|(_, s)| s)
        } else {
            None
        };
        let Some(symbol) = symbol else {
            return None;
        };
        let mut symbol_offset = 0;
        let mut sub_block_offset = 0;
        for sub_block in 0..(nl + ns) {
            let bytes = if sub_block < nl {
                tl as usize * symbol_alignment as usize
            } else {
                ts as usize * symbol_alignment as usize
            };
            let start = sub_block_offset + bytes * i;
            result[start..start + bytes]
                .copy_from_slice(&symbol.as_bytes()[symbol_offset..symbol_offset + bytes]);
            symbol_offset += bytes;
            sub_block_offset += bytes * k as usize;
        }
    }

    Some(result)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::encoder::{SourceBlockEncoder, SourceBlockEncodingPlan};
    use crate::ObjectTransmissionInformation;

    fn plan_for(symbol_count: u16) -> SourceBlockEncodingPlan {
        SourceBlockEncodingPlan::generate(symbol_count)
    }

    #[test]
    fn reduced_matches_standard_for_random_erasures() {
        let symbol_size: u16 = 1350;
        for &k in &[24u16, 32, 40, 64, 100, 200] {
            for &lost in &[1u32, 2, 4, 8] {
                if lost as u16 >= k {
                    continue;
                }
                let mut data: Vec<u8> = vec![0; symbol_size as usize * k as usize];
                for b in data.iter_mut() {
                    *b = rand::random();
                }
                let config =
                    ObjectTransmissionInformation::new(0, symbol_size, 1, 1, 1);
                let plan = plan_for(k);
                let encoder = SourceBlockEncoder::with_encoding_plan(1, &config, &data, &plan);

                let source_packets: Vec<EncodingPacket> = encoder.source_packets();
                let repair_packets: Vec<EncodingPacket> = encoder.repair_packets(0, 64);

                // Drop `lost` source symbols at scattered positions.
                let mut dropped: Vec<u32> = vec![];
                let mut step = k / (lost as u16 + 1);
                if step == 0 {
                    step = 1;
                }
                let mut pos = step;
                while (dropped.len() as u32) < lost && pos < k {
                    dropped.push(pos as u32);
                    pos += step;
                }
                while (dropped.len() as u32) < lost {
                    dropped.push(0);
                }

                let mut sources: Vec<Option<Symbol>> = vec![None; k as usize];
                let mut repairs: Vec<EncodingPacket> = vec![];
                for (i, p) in source_packets.iter().enumerate() {
                    if dropped.contains(&(i as u32)) {
                        continue;
                    }
                    sources[i] = Some(Symbol::new(p.data().to_vec()));
                }
                repairs.extend(
                    repair_packets
                        .iter()
                        .take(lost as usize + 2)
                        .cloned(),
                );

                let fast = try_reduced_decode(
                    k as u32,
                    symbol_size,
                    1,
                    1,
                    &sources,
                    &repairs,
                );
                assert!(
                    fast.is_some(),
                    "reduced decode failed for k={k} lost={lost}"
                );

                let fast_out = fast.unwrap();
                assert_eq!(
                    fast_out,
                    data.as_slice(),
                    "reduced decode output mismatch for k={k} lost={lost}"
                );
            }
        }
    }

    #[test]
    fn reduced_rejects_too_many_lost() {
        let symbol_size: u16 = 64;
        let k: u16 = 32;
        let data: Vec<u8> = vec![7; symbol_size as usize * k as usize];
        let config = ObjectTransmissionInformation::new(0, symbol_size, 1, 1, 1);
        let plan = plan_for(k);
        let encoder = SourceBlockEncoder::with_encoding_plan(1, &config, &data, &plan);
        let source_packets: Vec<EncodingPacket> = encoder.source_packets();
        let repair_packets: Vec<EncodingPacket> = encoder.repair_packets(0, 64);

        // Drop 20 > REDUCED_LOST_THRESHOLD symbols.
        let mut sources: Vec<Option<Symbol>> = vec![None; k as usize];
        for (i, p) in source_packets.iter().enumerate().take(k as usize - 20) {
            sources[i] = Some(Symbol::new(p.data().to_vec()));
        }
        assert!(try_reduced_decode(
            k as u32,
            symbol_size,
            1,
            1,
            &sources,
            &repair_packets[..20],
        )
        .is_none());
    }

    #[test]
    fn reduced_requires_repair_symbols() {
        let symbol_size: u16 = 64;
        let k: u16 = 32;
        let data: Vec<u8> = vec![3; symbol_size as usize * k as usize];
        let config = ObjectTransmissionInformation::new(0, symbol_size, 1, 1, 1);
        let plan = plan_for(k);
        let encoder = SourceBlockEncoder::with_encoding_plan(1, &config, &data, &plan);
        let source_packets: Vec<EncodingPacket> = encoder.source_packets();

        let mut sources: Vec<Option<Symbol>> = vec![None; k as usize];
        for (i, p) in source_packets.iter().enumerate().take(k as usize - 1) {
            sources[i] = Some(Symbol::new(p.data().to_vec()));
        }
        // One lost symbol but zero repair packets: fast path must bail out.
        assert!(try_reduced_decode(k as u32, symbol_size, 1, 1, &sources, &[])
            .is_none());
    }
}
