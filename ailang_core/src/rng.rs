// Centralized RNG for deterministic execution
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct SeededRng {
    rng: ChaCha8Rng,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn gen(&mut self) -> f32 {
        self.rng.gen::<f32>()
    }

    pub fn gen_range(&mut self, low: f32, high: f32) -> f32 {
        self.rng.gen_range(low..high)
    }

    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        // Box-Muller transform for normal distribution
        let u1: f32 = self.rng.gen();
        let u2: f32 = self.rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + std * z
    }
}
