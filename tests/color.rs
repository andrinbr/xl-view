use xl_view::color::{
    hlg_inverse_oetf, hlg_oetf, hlg_system_gamma, pq_eotf, pq_oetf, srgb_eotf, srgb_oetf,
};

// Independently evaluated from the normative sRGB, SMPTE ST 2084, and
// BT.2100 equations rather than from the production implementations.
const SRGB_REFERENCES: [(f64, f64); 3] = [(0.0, 0.0), (0.003_130_8, 0.040_449_936), (1.0, 1.0)];
const PQ_REFERENCES: [(f64, f64); 3] = [
    (100.0, 0.508_078_421_517_399),
    (1_000.0, 0.751_827_096_247_041),
    (10_000.0, 1.0),
];
const HLG_REFERENCES: [(f64, f64); 3] = [(0.0, 0.0), (0.083_333_333_333_333_33, 0.5), (1.0, 1.0)];
const HLG_SYSTEM_GAMMA_REFERENCES: [(f32, f32); 6] = [
    (100.0, 0.845_907),
    (400.0, 1.032_865),
    (1_000.0, 1.2),
    (2_000.0, 1.326_433),
    (4_000.0, 1.481_185),
    (10_000.0, 1.702_316),
];

fn assert_near(actual: impl Into<f64>, expected: impl Into<f64>, tolerance: f64) {
    let actual = actual.into();
    let expected = expected.into();
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
#[expect(clippy::cast_possible_truncation)] // The production color pipeline intentionally uses f32.
fn transfer_functions_match_numeric_references() {
    for (linear, encoded) in SRGB_REFERENCES {
        assert_near(srgb_oetf(linear), encoded, 1.0e-9);
        assert_near(srgb_eotf(encoded), linear, 1.0e-9);
    }
    for (nits, encoded) in PQ_REFERENCES {
        let normalized = nits / 10_000.0;
        assert_near(pq_oetf(normalized as f32), encoded, 5.0e-6);
        assert_near(pq_eotf(encoded as f32), normalized, 5.0e-6);
    }
    for (scene_linear, encoded) in HLG_REFERENCES {
        assert_near(hlg_oetf(scene_linear as f32), encoded, 1.0e-6);
        assert_near(hlg_inverse_oetf(encoded as f32), scene_linear, 1.0e-6);
    }
}

#[test]
fn hlg_system_gamma_uses_the_bt2100_extended_formula_outside_the_production_range() {
    for (display_peak_nits, expected_gamma) in HLG_SYSTEM_GAMMA_REFERENCES {
        assert_near(hlg_system_gamma(display_peak_nits), expected_gamma, 1.0e-6);
    }
}
