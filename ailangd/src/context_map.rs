use ailang_core::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct StrategyContextTick {
    #[serde(default)]
    pub now_ts: Option<u64>,
    pub products: Vec<String>,
    pub market: MarketDataTick,
    pub portfolio: PortfolioTick,
    pub limits: LimitsTick,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MarketDataTick {
    pub mid: Option<f64>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub spread_bps: Option<f64>,
    pub depth_usd_bid: Option<f64>,
    pub depth_usd_ask: Option<f64>,
    #[serde(default)]
    pub candles_1m: Vec<f64>,
    #[serde(default)]
    pub candles_5m: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PortfolioTick {
    pub balances_usd: Option<f64>,
    #[serde(default)]
    pub holdings: std::collections::HashMap<String, f64>,
    pub open_orders_count: Option<u64>,
    pub realized_pnl_today_usd: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LimitsTick {
    pub max_gross_notional_usd: Option<f64>,
    pub max_notional_per_product_usd: Option<f64>,
    pub max_daily_loss_usd: Option<f64>,
    pub max_slippage_bps: Option<u64>,
    pub max_open_orders: Option<u64>,
    pub max_orders_per_min: Option<u64>,
    pub max_spread_bps: Option<f64>,
    pub min_book_depth_usd: Option<f64>,
}

/// Build tensor inputs from StrategyContext for AILang execution.
/// 
/// Creates 21 canonical inputs as specified in the contract.
/// Uses f32::NAN for missing/null values.
/// Left-pads candles with NaN if shorter than specified lengths.
pub fn build_inputs(
    init_products: &[String],
    c1: usize,
    c5: usize,
    tick: StrategyContextTick,
    seq: u64,
    ts: u64,
) -> HashMap<String, Tensor> {
    let n = init_products.len();
    let mut inputs = HashMap::new();

    // Helper to convert Option<f64> to f32, using NaN for None
    let opt_to_f32 = |opt: Option<f64>| -> f32 {
        opt.map(|v| v as f32).unwrap_or(f32::NAN)
    };

    // Helper to convert Option<u64> to f32, using NaN for None
    let opt_u64_to_f32 = |opt: Option<u64>| -> f32 {
        opt.map(|v| v as f32).unwrap_or(f32::NAN)
    };

    // 1. product_count: scalar (N)
    inputs.insert(
        "product_count".to_string(),
        Tensor::from_vec(&[], vec![n as f32]),
    );

    // Build per-product arrays (2-7)
    // Product ordering: use INIT products list order
    // For each product in INIT order, look up market data
    // If product not found in tick.market, use NaN for all fields
    
    let mut market_mid = Vec::with_capacity(n);
    let mut market_best_bid = Vec::with_capacity(n);
    let mut market_best_ask = Vec::with_capacity(n);
    let mut market_spread_bps = Vec::with_capacity(n);
    let mut market_depth_usd_bid = Vec::with_capacity(n);
    let mut market_depth_usd_ask = Vec::with_capacity(n);

    // For MVP: assume single product market data (will be enhanced for multi-product)
    // If market data exists, use it for all products; otherwise use NaN
    let mid = opt_to_f32(tick.market.mid);
    let best_bid = opt_to_f32(tick.market.best_bid);
    let best_ask = opt_to_f32(tick.market.best_ask);
    let spread_bps = opt_to_f32(tick.market.spread_bps);
    let depth_bid = opt_to_f32(tick.market.depth_usd_bid);
    let depth_ask = opt_to_f32(tick.market.depth_usd_ask);

    for _ in 0..n {
        market_mid.push(mid);
        market_best_bid.push(best_bid);
        market_best_ask.push(best_ask);
        market_spread_bps.push(spread_bps);
        market_depth_usd_bid.push(depth_bid);
        market_depth_usd_ask.push(depth_ask);
    }

    // 2-7. Market arrays [N]
    inputs.insert("market_mid".to_string(), Tensor::from_vec(&[n], market_mid));
    inputs.insert("market_best_bid".to_string(), Tensor::from_vec(&[n], market_best_bid));
    inputs.insert("market_best_ask".to_string(), Tensor::from_vec(&[n], market_best_ask));
    inputs.insert("market_spread_bps".to_string(), Tensor::from_vec(&[n], market_spread_bps));
    inputs.insert("market_depth_usd_bid".to_string(), Tensor::from_vec(&[n], market_depth_usd_bid));
    inputs.insert("market_depth_usd_ask".to_string(), Tensor::from_vec(&[n], market_depth_usd_ask));

    // 8-9. Candles arrays [N, C1] and [N, C5]
    // Left-pad with NaN if shorter, keep last C1/C5 if longer
    let mut candles_1m_data = tick.market.candles_1m.iter().map(|&v| v as f32).collect::<Vec<_>>();
    if candles_1m_data.len() < c1 {
        // Left-pad with NaN
        let pad_len = c1 - candles_1m_data.len();
        let mut padded = vec![f32::NAN; pad_len];
        padded.extend(candles_1m_data);
        candles_1m_data = padded;
    } else if candles_1m_data.len() > c1 {
        // Keep last C1 values
        candles_1m_data = candles_1m_data[candles_1m_data.len() - c1..].to_vec();
    }

    let mut candles_5m_data = tick.market.candles_5m.iter().map(|&v| v as f32).collect::<Vec<_>>();
    if candles_5m_data.len() < c5 {
        // Left-pad with NaN
        let pad_len = c5 - candles_5m_data.len();
        let mut padded = vec![f32::NAN; pad_len];
        padded.extend(candles_5m_data);
        candles_5m_data = padded;
    } else if candles_5m_data.len() > c5 {
        // Keep last C5 values
        candles_5m_data = candles_5m_data[candles_5m_data.len() - c5..].to_vec();
    }

    // For multi-product: repeat candles for each product (for MVP, use same candles for all)
    let mut candles_1m_expanded = Vec::with_capacity(n * c1);
    let mut candles_5m_expanded = Vec::with_capacity(n * c5);
    for _ in 0..n {
        candles_1m_expanded.extend_from_slice(&candles_1m_data);
        candles_5m_expanded.extend_from_slice(&candles_5m_data);
    }

    inputs.insert("candles_1m_close".to_string(), Tensor::from_vec(&[n, c1], candles_1m_expanded));
    inputs.insert("candles_5m_close".to_string(), Tensor::from_vec(&[n, c5], candles_5m_expanded));

    // 10-19. Portfolio and limits scalars
    inputs.insert(
        "portfolio_realized_pnl_today_usd".to_string(),
        Tensor::from_vec(&[], vec![opt_to_f32(tick.portfolio.realized_pnl_today_usd)]),
    );
    inputs.insert(
        "portfolio_open_orders_count".to_string(),
        Tensor::from_vec(&[], vec![opt_u64_to_f32(tick.portfolio.open_orders_count)]),
    );
    inputs.insert(
        "limits_max_gross_notional_usd".to_string(),
        Tensor::from_vec(&[], vec![opt_to_f32(tick.limits.max_gross_notional_usd)]),
    );
    inputs.insert(
        "limits_max_notional_per_product_usd".to_string(),
        Tensor::from_vec(&[], vec![opt_to_f32(tick.limits.max_notional_per_product_usd)]),
    );
    inputs.insert(
        "limits_max_daily_loss_usd".to_string(),
        Tensor::from_vec(&[], vec![opt_to_f32(tick.limits.max_daily_loss_usd)]),
    );
    inputs.insert(
        "limits_max_slippage_bps".to_string(),
        Tensor::from_vec(&[], vec![opt_u64_to_f32(tick.limits.max_slippage_bps)]),
    );
    inputs.insert(
        "limits_max_open_orders".to_string(),
        Tensor::from_vec(&[], vec![opt_u64_to_f32(tick.limits.max_open_orders)]),
    );
    inputs.insert(
        "limits_max_orders_per_min".to_string(),
        Tensor::from_vec(&[], vec![opt_u64_to_f32(tick.limits.max_orders_per_min)]),
    );
    inputs.insert(
        "limits_max_spread_bps".to_string(),
        Tensor::from_vec(&[], vec![opt_to_f32(tick.limits.max_spread_bps)]),
    );
    inputs.insert(
        "limits_min_book_depth_usd".to_string(),
        Tensor::from_vec(&[], vec![opt_to_f32(tick.limits.min_book_depth_usd)]),
    );

    // 20-21. Tick metadata
    inputs.insert("tick_seq".to_string(), Tensor::from_vec(&[], vec![seq as f32]));
    inputs.insert("tick_ts".to_string(), Tensor::from_vec(&[], vec![ts as f32]));

    inputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_inputs_shapes() {
        let products = vec!["BTC-USD".to_string(), "ETH-USD".to_string()];
        let c1 = 3;
        let c5 = 2;

        let tick = StrategyContextTick {
            now_ts: None,
            products: products.clone(),
            market: MarketDataTick {
                mid: Some(50000.0),
                best_bid: Some(49990.0),
                best_ask: Some(50010.0),
                spread_bps: Some(4.0),
                depth_usd_bid: Some(100000.0),
                depth_usd_ask: Some(100000.0),
                candles_1m: vec![50000.0, 50010.0], // Short, should be padded
                candles_5m: vec![49990.0, 50000.0, 50010.0, 50020.0], // Long, should be truncated
            },
            portfolio: PortfolioTick {
                balances_usd: Some(10000.0),
                holdings: HashMap::new(),
                open_orders_count: Some(2),
                realized_pnl_today_usd: Some(-100.0),
            },
            limits: LimitsTick {
                max_gross_notional_usd: Some(50000.0),
                max_notional_per_product_usd: Some(25000.0),
                max_daily_loss_usd: Some(1000.0),
                max_slippage_bps: Some(10),
                max_open_orders: Some(10),
                max_orders_per_min: Some(5),
                max_spread_bps: Some(5.0),
                min_book_depth_usd: Some(5000.0),
            },
        };

        let inputs = build_inputs(&products, c1, c5, tick, 1, 1000);

        // Verify product_count (scalar tensor with shape [])
        assert_eq!(inputs["product_count"].shape(), &[] as &[usize]);
        assert_eq!(inputs["product_count"].scalar(), 2.0);

        // Verify market arrays [N]
        assert_eq!(inputs["market_mid"].shape(), &[2]);
        assert_eq!(inputs["market_best_bid"].shape(), &[2]);
        assert_eq!(inputs["market_best_ask"].shape(), &[2]);

        // Verify candles shapes [N, C1] and [N, C5]
        assert_eq!(inputs["candles_1m_close"].shape(), &[2, 3]);
        assert_eq!(inputs["candles_5m_close"].shape(), &[2, 2]);

        // Verify candles padding (1m should be left-padded with NaN)
        let candles_1m = inputs["candles_1m_close"].data.as_slice().unwrap();
        assert!(candles_1m[0].is_nan()); // First value should be NaN (padding)
        assert!((candles_1m[1] - 50000.0).abs() < 0.01); // Second value
        assert!((candles_1m[2] - 50010.0).abs() < 0.01); // Third value

        // Verify candles truncation (5m should keep last 2 values)
        let candles_5m = inputs["candles_5m_close"].data.as_slice().unwrap();
        assert!((candles_5m[0] - 50010.0).abs() < 0.01); // Last but one
        assert!((candles_5m[1] - 50020.0).abs() < 0.01); // Last value

        // Verify scalar inputs (scalar tensors with shape [])
        assert_eq!(inputs["tick_seq"].shape(), &[] as &[usize]);
        assert_eq!(inputs["tick_seq"].scalar(), 1.0);
        assert_eq!(inputs["tick_ts"].shape(), &[] as &[usize]);
        assert_eq!(inputs["tick_ts"].scalar(), 1000.0);
    }

    #[test]
    fn test_build_inputs_missing_values() {
        let products = vec!["BTC-USD".to_string()];
        let c1 = 2;
        let c5 = 2;

        let tick = StrategyContextTick {
            now_ts: None,
            products: products.clone(),
            market: MarketDataTick {
                mid: None, // Missing value
                best_bid: None,
                best_ask: None,
                spread_bps: None,
                depth_usd_bid: None,
                depth_usd_ask: None,
                candles_1m: vec![],
                candles_5m: vec![],
            },
            portfolio: PortfolioTick {
                balances_usd: None,
                holdings: HashMap::new(),
                open_orders_count: None,
                realized_pnl_today_usd: None,
            },
            limits: LimitsTick {
                max_gross_notional_usd: None,
                max_notional_per_product_usd: None,
                max_daily_loss_usd: None,
                max_slippage_bps: None,
                max_open_orders: None,
                max_orders_per_min: None,
                max_spread_bps: None,
                min_book_depth_usd: None,
            },
        };

        let inputs = build_inputs(&products, c1, c5, tick, 0, 0);

        // Verify missing values are NaN
        let mid = inputs["market_mid"].data.as_slice().unwrap()[0];
        assert!(mid.is_nan());

        let pnl = inputs["portfolio_realized_pnl_today_usd"].scalar();
        assert!(pnl.is_nan());
    }
}
