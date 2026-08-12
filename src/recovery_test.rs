#[cfg(test)]
mod recovery_test {
    use crate::encoder::{SourceBlockEncoder, SourceBlockEncodingPlan};
    use crate::{ObjectTransmissionInformation, SourceBlockDecoder};
    use std::iter;

    #[test]
    #[ignore]
    fn recovery_rate() {
        let symbol_size: u16 = 1350;
        // 16:32 = 16 data + 32 repair = 48 symbols, 30% independent loss
        let k: u16 = 16;
        let parity: usize = 32;
        let loss: f64 = 0.30;
        let trials = 2000;
        let mut failed = 0u64;
        let mut rng_state: u64 = 12345;
        let mut rng = move || { rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (rng_state >> 33) as u64 };

        let mut data: Vec<u8> = vec![0; symbol_size as usize * k as usize];
        for (i, b) in data.iter_mut().enumerate() { *b = (i * 7 % 251) as u8; }
        let config = ObjectTransmissionInformation::new(data.len() as u64, symbol_size, 1, 1, 1);
        let plan = SourceBlockEncodingPlan::generate(k);
        let encoder = SourceBlockEncoder::with_encoding_plan(1, &config, &data, &plan);
        let sources = encoder.source_packets();
        let repairs = encoder.repair_packets(0, 64);

        for _ in 0..trials {
            let mut dec = SourceBlockDecoder::new(1, &config, symbol_size as u64 * k as u64);
            let mut got = 0u32;
            let mut out = None;
            for p in sources.iter() {
                if (rng() % 1000) as f64 / 10.0 >= loss {
                    got += 1;
                    out = dec.decode(iter::once(p.clone()));
                }
            }
            for r in repairs.iter().take(parity) {
                if (rng() % 1000) as f64 / 10.0 >= loss {
                    got += 1;
                    out = dec.decode(iter::once(r.clone()));
                }
            }
            match &out {
                Some(v) if v == &data => {}
                Some(_) => { failed += 1; eprintln!("group: got={} symbols, wrong result", got); }
                None => { failed += 1; eprintln!("group: got={} symbols, None", got); }
            }
        }
        eprintln!("RECOVERY: trials={} failed={} ({:.4}%)", trials, failed, failed as f64 * 100.0 / trials as f64);
    }
}
