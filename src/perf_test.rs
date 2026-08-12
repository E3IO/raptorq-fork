#[cfg(test)]
mod perf_test {
    use crate::encoder::{SourceBlockEncoder, SourceBlockEncodingPlan};
    use crate::reduced::try_reduced_decode;
    use crate::{ObjectTransmissionInformation, SourceBlockDecoder};
    use std::iter;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn perf_reduced_vs_standard() {
        let symbol_size: u16 = 1350;
        for &(k, lost) in &[(100u16, 1u32), (100, 8), (500, 5), (1000, 5)] {
            if lost >= k as u32 { continue; }
            let mut data: Vec<u8> = vec![0; symbol_size as usize * k as usize];
            for (i, b) in data.iter_mut().enumerate() { *b = (i * 7 % 251) as u8; }
            let config = ObjectTransmissionInformation::new(data.len() as u64, symbol_size, 1, 1, 1);
            let plan = SourceBlockEncodingPlan::generate(k);
            let encoder = SourceBlockEncoder::with_encoding_plan(1, &config, &data, &plan);
            let sources = encoder.source_packets();
            let repairs = encoder.repair_packets(0, 64);

            // drop `lost` sources at scattered positions
            let mut dropped: Vec<u32> = vec![];
            let mut pos = k / (lost as u16 + 1);
            while dropped.len() < lost as usize && pos < k { dropped.push(pos as u32); pos += k / (lost as u16 + 1); }
            while dropped.len() < lost as usize { dropped.push(0); }

            let mut srcs: Vec<Option<crate::symbol::Symbol>> = vec![None; k as usize];
            for (i, p) in sources.iter().enumerate() {
                if !dropped.contains(&(i as u32)) { srcs[i] = Some(crate::symbol::Symbol::new(p.data().to_vec())); }
            }
            let reps: Vec<crate::base::EncodingPacket> = repairs.iter().take(lost as usize + 2).cloned().collect();

            // warmup
            for _ in 0..3 {
                let _ = try_reduced_decode(k as u32, symbol_size, 1, 1, &srcs, &reps);
            }
            let n = 200;
            let t0 = Instant::now();
            let mut ok = 0;
            for _ in 0..n {
                let r = try_reduced_decode(k as u32, symbol_size, 1, 1, &srcs, &reps).unwrap();
                if r == data { ok += 1; }
            }
            let fast_us = t0.elapsed().as_micros() as f64 / n as f64;

            // standard (SourceBlockDecoder) - only valid where pi_solver correct (k=100 may be buggy)
            let mut dec = SourceBlockDecoder::new(1, &config, symbol_size as u64 * k as u64);
            for (i, p) in sources.iter().enumerate() {
                if !dropped.contains(&(i as u32)) { dec.decode(iter::once(p.clone())); }
            }
            for r in reps.iter() { dec.decode(iter::once(r.clone())); }
            let t1 = Instant::now();
            for _ in 0..(n / 5) {
                let mut d2 = SourceBlockDecoder::new(1, &config, symbol_size as u64 * k as u64);
                for (i, p) in sources.iter().enumerate() {
                    if !dropped.contains(&(i as u32)) { d2.decode(iter::once(p.clone())); }
                }
                for r in reps.iter() { d2.decode(iter::once(r.clone())); }
            }
            let std_us = t1.elapsed().as_micros() as f64 / (n / 5) as f64;
            eprintln!("k={k} lost={lost}: reduced {:.0}us (ok {}/{}), standard {:.0}us, speedup {:.1}x",
                fast_us, ok, n, std_us, std_us / fast_us);
        }
    }
}
