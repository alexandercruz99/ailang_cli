use ailang_core::tensor::Tensor;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct StrategyContextTick {
    #[serde(default)]
    pub now_ts: Option<u64>,
    pub products: Vec<String>,
    pub market: MarketData,
    pub portfolio: Portfolio,
    pub limits: Limits,
}

#[derive(Debug, Deserialize)]
pub struct MarketData {
    pub mid: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub spread_bps: f64,
    pub depth_usd_bid: f64,
    pub depth_usd_ask: f64,
    pub candles_1m: Vec<f64>,
    pub candles_5m: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Portfolio {
    pub balances_usd: f64,
    #[serde(default)]
    pub holdings: HashMap<String, f64>,
    pub open_orders_count: i32,
    pub realized_pnl_today_usd: f64,
}

#[derive(Debug, Deserialize)]
pub struct Limits {
    pub max_gross_notional_usd: f64,
    pub max_notional_per_product_usd: f64,
    pub max_daily_loss_usd: f64,
    pub max_slippage_bps: i32,
    pub max_open_orders: i32,
    pub max_orders_per_min: i32,
    pub max_spread_bps: f64,
    pub min_book_depth_usd: f64,
}

/// Build canonical tensor inputs from StrategyContext.
/// Returns a HashMap mapping input names to Tensor values.
pub fn build_inputs(
    init_products: &[String],
    c1: usize,
    c5: usize,
    tick: StrategyContextTick,
    seq: u64,
    ts: u64,
) -> HashMap<String, Tensor> {
    let mut inputs = HashMap::new();
    let n = init_products.len();

    // Helper to convert f64 to f32, using NaN for invalid values
    let to_f32 = |v: f64| -> f32 {
        if v.is_finite() {
            v as f32
        } else {
            f32::NAN
        }
    };

    // Helper to get market value for a product, using NaN if not found
    let get_market_value = |value: f64| -> f32 { to_f32(value) };

    // 1. product_count: scalar (N as f32)
    inputs.insert(
        "product_count".to_string(),
        Tensor::from_vec(&[], vec![n as f32]),
    );

    // 2-8. Market arrays [N]
    // For single-product market data, we replicate it across all products
    // If products list doesn't match, we use NaN for missing products
    let mut market_mid = Vec::with_capacity(n);
    let mut market_best_bid = Vec::with_capacity(n);
    let mut market_best_ask = Vec::with_capacity(n);
    let mut market_spread_bps = Vec::with_capacity(n);
    let mut market_depth_usd_bid = Vec::with_capacity(n);
    let mut market_depth_usd_ask = Vec::with_capacity(n);

    // For now, we assume the market data is for the first product
    // In a multi-product setup, we'd need per-product market data
    // This is a simplification - in production, market data should be per-product
    for _ in 0..n {
        market_mid.push(get_market_value(tick.market.mid));
        market_best_bid.push(get_market_value(tick.market.best_bid));
        market_best_ask.push(get_market_value(tick.market.best_ask));
        market_spread_bps.push(get_market_value(tick.market.spread_bps));
        market_depth_usd_bid.push(get_market_value(tick.market.depth_usd_bid));
        market_depth_usd_ask.push(get_market_value(tick.market.depth_usd_ask));
    }

    inputs.insert("market_mid".to_string(), Tensor::from_vec(&[n], market_mid));
    inputs.insert(
        "market_best_bid".to_string(),
        Tensor::from_vec(&[n], market_best_bid),
    );
    inputs.insert(
        "market_best_ask".to_string(),
        Tensor::from_vec(&[n], market_best_ask),
    );
    inputs.insert(
        "market_spread_bps".to_string(),
        Tensor::from_vec(&[n], market_spread_bps),
    );
    inputs.insert(
        "market_depth_usd_bid".to_string(),
        Tensor::from_vec(&[n], market_depth_usd_bid),
    );
    inputs.insert(
        "market_depth_usd_ask".to_string(),
        Tensor::from_vec(&[n], market_depth_usd_ask),
    );

    // 8-9. Candles: [N, C1] and [N, C5]
    // Left-pad with NaN if shorter, keep last C1/C5 if longer
    let pad_candle = |candles: &[f64], len: usize| -> Vec<f32> {
        let mut padded = Vec::with_capacity(len);
        if candles.len() < len {
            // Left-pad with NaN
            for _ in 0..(len - candles.len()) {
                padded.push(f32::NAN);
            }
            for &c in candles {
                padded.push(to_f32(c));
            }
        } else {
            // Keep last len values
            for &c in candles.iter().skip(candles.len() - len) {
                padded.push(to_f32(c));
            }
        }
        padded
    };

    let candle_1m_padded = pad_candle(&tick.market.candles_1m, c1);
    let candle_5m_padded = pad_candle(&tick.market.candles_5m, c5);

    // Replicate candles across all products (same simplification as market data)
    let mut candles_1m_close = Vec::with_capacity(n * c1);
    let mut candles_5m_close = Vec::with_capacity(n * c5);
    for _ in 0..n {
        candles_1m_close.extend_from_slice(&candle_1m_padded);
        candles_5m_close.extend_from_slice(&candle_5m_padded);
    }

    inputs.insert(
        "candles_1m_close".to_string(),
        Tensor::from_vec(&[n, c1], candles_1m_close),
    );
    inputs.insert(
        "candles_5m_close".to_string(),
        Tensor::from_vec(&[n, c5], candles_5m_close),
    );

    // 10-19. Portfolio and limits scalars
    inputs.insert(
        "portfolio_realized_pnl_today_usd".to_string(),
        Tensor::from_vec(&[], vec![to_f32(tick.portfolio.realized_pnl_today_usd)]),
    );
    inputs.insert(
        "portfolio_open_orders_count".to_string(),
        Tensor::from_vec(&[], vec![tick.portfolio.open_orders_count as f32]),
    );
    inputs.insert(
        "limits_max_gross_notional_usd".to_string(),
        Tensor::from_vec(&[], vec![to_f32(tick.limits.max_gross_notional_usd)]),
    );
    inputs.insert(
        "limits_max_notional_per_product_usd".to_string(),
        Tensor::from_vec(&[], vec![to_f32(tick.limits.max_notional_per_product_usd)]),
    );
    inputs.insert(
        "limits_max_daily_loss_usd".to_string(),
        Tensor::from_vec(&[], vec![to_f32(tick.limits.max_daily_loss_usd)]),
    );
    inputs.insert(
        "limits_max_slippage_bps".to_string(),
        Tensor::from_vec(&[], vec![tick.limits.max_slippage_bps as f32]),
    );
    inputs.insert(
        "limits_max_open_orders".to_string(),
        Tensor::from_vec(&[], vec![tick.limits.max_open_orders as f32]),
    );
    inputs.insert(
        "limits_max_orders_per_min".to_string(),
        Tensor::from_vec(&[], vec![tick.limits.max_orders_per_min as f32]),
    );
    inputs.insert(
        "limits_max_spread_bps".to_string(),
        Tensor::from_vec(&[], vec![to_f32(tick.limits.max_spread_bps)]),
    );
    inputs.insert(
        "limits_min_book_depth_usd".to_string(),
        Tensor::from_vec(&[], vec![to_f32(tick.limits.min_book_depth_usd)]),
    );

    // 20-21. Tick metadata scalars
    inputs.insert(
        "tick_seq".to_string(),
        Tensor::from_vec(&[], vec![seq as f32]),
    );
    inputs.insert(
        "tick_ts".to_string(),
        Tensor::from_vec(&[], vec![ts as f32]),
    );

    inputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_inputs_basic() {
        let products = vec!["BTC-USD".to_string(), "ETH-USD".to_string()];
        let c1 = 3;
        let c5 = 2;

        let tick = StrategyContextTick {
            now_ts: None,
            products: products.clone(),
            market: MarketData {
                mid: 50000.0,
                best_bid: 49999.0,
                best_ask: 50001.0,
                spread_bps: 4.0,
                depth_usd_bid: 1000000.0,
                depth_usd_ask: 1000000.0,
                candles_1m: vec![49900.0, 49950.0, 50000.0],
                candles_5m: vec![49800.0, 50000.0],
            },
            portfolio: Portfolio {
                balances_usd: 10000.0,
                holdings: HashMap::new(),
                open_orders_count: 2,
                realized_pnl_today_usd: 100.0,
            },
            limits: Limits {
                max_gross_notional_usd: 50000.0,
                max_notional_per_product_usd: 25000.0,
                max_daily_loss_usd: 1000.0,
                max_slippage_bps: 10,
                max_open_orders: 10,
                max_orders_per_min: 5,
                max_spread_bps: 20.0,
                min_book_depth_usd: 5000.0,
            },
        };

        let inputs = build_inputs(&products, c1, c5, tick, 1, 1000);

        // Check product_count (scalar has empty shape)
        assert_eq!(inputs["product_count"].shape().len(), 0);
        assert_eq!(inputs["product_count"].scalar(), 2.0);

        // Check market arrays
        assert_eq!(inputs["market_mid"].shape(), &[2]);
        assert_eq!(inputs["market_best_bid"].shape(), &[2]);
        assert_eq!(inputs["market_best_ask"].shape(), &[2]);

        // Check candles
        assert_eq!(inputs["candles_1m_close"].shape(), &[2, 3]);
        assert_eq!(inputs["candles_5m_close"].shape(), &[2, 2]);

        // Check scalars
        assert_eq!(inputs["tick_seq"].scalar(), 1.0);
        assert_eq!(inputs["tick_ts"].scalar(), 1000.0);
    }

    #[test]
    fn test_build_inputs_candle_padding() {
        let products = vec!["BTC-USD".to_string()];
        let c1 = 5;
        let c5 = 4;

        let tick = StrategyContextTick {
            now_ts: None,
            products: products.clone(),
            market: MarketData {
                mid: 50000.0,
                best_bid: 49999.0,
                best_ask: 50001.0,
                spread_bps: 4.0,
                depth_usd_bid: 1000000.0,
                depth_usd_ask: 1000000.0,
                candles_1m: vec![50000.0], // Shorter than c1=5
                candles_5m: vec![49800.0, 49900.0, 50000.0, 50100.0, 50200.0], // Longer than c5=4
            },
            portfolio: Portfolio {
                balances_usd: 10000.0,
                holdings: HashMap::new(),
                open_orders_count: 0,
                realized_pnl_today_usd: 0.0,
            },
            limits: Limits {
                max_gross_notional_usd: 50000.0,
                max_notional_per_product_usd: 25000.0,
                max_daily_loss_usd: 1000.0,
                max_slippage_bps: 10,
                max_open_orders: 10,
                max_orders_per_min: 5,
                max_spread_bps: 20.0,
                min_book_depth_usd: 5000.0,
            },
        };

        let inputs = build_inputs(&products, c1, c5, tick, 1, 1000);

        // Check candle 1m: should be left-padded with NaN
        let candles_1m = &inputs["candles_1m_close"];
        assert_eq!(candles_1m.shape(), &[1, 5]);
        // First 4 should be NaN, last should be 50000.0
        let data_1m: Vec<f32> = candles_1m.data.iter().copied().collect();
        assert!(data_1m[0].is_nan());
        assert!(data_1m[1].is_nan());
        assert!(data_1m[2].is_nan());
        assert!(data_1m[3].is_nan());
        assert!((data_1m[4] - 50000.0).abs() < 0.01);

        // Check candle 5m: should keep last 4 values
        let candles_5m = &inputs["candles_5m_close"];
        assert_eq!(candles_5m.shape(), &[1, 4]);
        let data_5m: Vec<f32> = candles_5m.data.iter().copied().collect();
        assert!((data_5m[0] - 49900.0).abs() < 0.01);
        assert!((data_5m[1] - 50000.0).abs() < 0.01);
        assert!((data_5m[2] - 50100.0).abs() < 0.01);
        assert!((data_5m[3] - 50200.0).abs() < 0.01);
    }
}
