//! Behaviour pins for the functions converted from `-> Vec<T>` to
//! `-> impl Iterator<Item = T>`.
//!
//! These conversions are meant to remove an allocation and nothing else, so
//! every assertion here describes behaviour that existed before the change.
//! They exist because the failure mode is silent: a flipped index in
//! `triangulate_band_ring` inverts fill normals at render time rather than
//! failing to compile, and a mis-stepped iterator in `dim_override::pairs`
//! drops overrides instead of erroring.

use OpenCADStudio::entities::common::triangulate_band_ring;
use OpenCADStudio::entities::dim_override;
use OpenCADStudio::entities::multileader::catmull_rom_pts;
use OpenCADStudio::scene::model::wire_model::WireModel;
use OpenCADStudio::scene::pipeline::wire_gpu::emit_wire_packed;

use acadrust::xdata::{ExtendedData, ExtendedDataRecord, XDataValue};

/// `ring[i]` as a distinguishable point, so an index mix-up is visible.
fn ring_of(n: usize) -> Vec<[f64; 3]> {
    (0..n).map(|i| [i as f64, 0.0, 0.0]).collect()
}

/// The x components of the emitted triangle stream — i.e. the ring indices in
/// emission order.
fn emitted_indices(ring: &[[f64; 3]]) -> Vec<usize> {
    triangulate_band_ring(ring)
        .map(|p| p[0] as usize)
        .collect()
}

#[test]
fn band_ring_emits_two_triangles_per_quad_in_strip_order() {
    // n = 4: a single straight trapezoid, split 0-1-2 / 0-2-3.
    assert_eq!(emitted_indices(&ring_of(4)), vec![0, 1, 2, 0, 2, 3]);

    // n = 6: m = 3, so j walks 0..2, pairing the outer run against the
    // reversed inner run.
    assert_eq!(
        emitted_indices(&ring_of(6)),
        vec![0, 1, 4, 0, 4, 5, 1, 2, 3, 1, 3, 4]
    );

    // n = 8: m = 4, j walks 0..3.
    assert_eq!(
        emitted_indices(&ring_of(8)),
        vec![0, 1, 6, 0, 6, 7, 1, 2, 5, 1, 5, 6, 2, 3, 4, 2, 4, 5]
    );
}

#[test]
fn band_ring_emits_six_vertices_per_quad() {
    for m in 2..8usize {
        let n = m * 2;
        assert_eq!(
            triangulate_band_ring(&ring_of(n)).count(),
            (m - 1) * 6,
            "banded ring of {n} vertices"
        );
    }
}

#[test]
fn band_ring_falls_back_below_four_vertices() {
    // Degenerate rings cannot form a quad strip; `triangulate_planar` handles
    // them, and it yields nothing below three points.
    for n in 0..3usize {
        assert_eq!(
            triangulate_band_ring(&ring_of(n)).count(),
            0,
            "ring of {n} vertices"
        );
    }
}

#[test]
fn band_ring_falls_back_on_odd_vertex_counts() {
    // An odd count cannot be split into two equal runs, so these take the
    // `triangulate_planar` path rather than the strip path. A real (non
    // collinear) triangle is needed for the planar triangulator to produce
    // anything.
    let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let out: Vec<_> = triangulate_band_ring(&tri).collect();
    assert_eq!(out.len(), 3, "a single triangle emits one triangle");

    // The strip path would have needed an even count; confirm it was not taken
    // by checking the count is not the strip formula's answer for m = 1.
    assert_ne!(out.len(), 0);
}

// ── dim_override::pairs ───────────────────────────────────────────────────

fn dstyle_xdata(app: &str, mut values: Vec<XDataValue>) -> ExtendedData {
    let mut record = ExtendedDataRecord::new(app);
    if app == "ACAD" {
        let mut all = vec![XDataValue::String("DSTYLE".to_string())];
        all.append(&mut values);
        record.values = all;
    } else {
        record.values = values;
    }
    let mut xd = ExtendedData::new();
    xd.add_record(record);
    xd
}

#[test]
fn pairs_reads_code_value_couples_and_skips_control_strings() {
    let xd = dstyle_xdata(
        "ACAD",
        vec![
            XDataValue::ControlString("{".to_string()),
            XDataValue::Integer16(40),
            XDataValue::Real(2.5),
            XDataValue::Integer16(176),
            XDataValue::Integer16(3),
            XDataValue::ControlString("}".to_string()),
        ],
    );

    let got: Vec<_> = dim_override::pairs(&xd).collect();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].0, 40);
    assert!(matches!(got[0].1, XDataValue::Real(r) if (r - 2.5).abs() < 1e-12));
    assert_eq!(got[1].0, 176);
    assert!(matches!(got[1].1, XDataValue::Integer16(3)));
}

#[test]
fn pairs_drops_a_trailing_code_with_no_value() {
    let xd = dstyle_xdata(
        "ACAD",
        vec![
            XDataValue::Integer16(40),
            XDataValue::Real(1.0),
            // A marker with nothing after it is incomplete and is dropped.
            XDataValue::Integer16(41),
        ],
    );

    assert_eq!(dim_override::pairs(&xd).count(), 1);
}

#[test]
fn pairs_falls_back_to_the_legacy_acad_dstyle_record() {
    // Records written by older OCS versions used a dedicated application name
    // and no leading "DSTYLE" marker.
    let xd = dstyle_xdata(
        "ACAD_DSTYLE",
        vec![XDataValue::Integer16(40), XDataValue::Real(3.0)],
    );

    let got: Vec<_> = dim_override::pairs(&xd).collect();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, 40);
}

#[test]
fn pairs_is_empty_without_a_dstyle_record() {
    // An ACAD record whose first value is not the DSTYLE marker carries
    // something else entirely and must not be parsed as overrides.
    let mut record = ExtendedDataRecord::new("ACAD");
    record.values = vec![
        XDataValue::String("SOMETHING_ELSE".to_string()),
        XDataValue::Integer16(40),
        XDataValue::Real(1.0),
    ];
    let mut xd = ExtendedData::new();
    xd.add_record(record);

    assert_eq!(dim_override::pairs(&xd).count(), 0);
    assert_eq!(dim_override::pairs(&ExtendedData::new()).count(), 0);
}

#[test]
fn real_accessor_finds_the_matching_code() {
    let xd = dstyle_xdata(
        "ACAD",
        vec![
            XDataValue::Integer16(40),
            XDataValue::Real(2.5),
            XDataValue::Integer16(dim_override::DIMTXT),
            XDataValue::Real(7.5),
        ],
    );

    // The accessor short-circuits at the match now; the answer must not change.
    assert_eq!(dim_override::real(&xd, dim_override::DIMTXT), Some(7.5));
    assert_eq!(dim_override::real(&xd, 40), Some(2.5));
    assert_eq!(dim_override::real(&xd, 999), None);
}

// ── catmull_rom_pts ───────────────────────────────────────────────────────

#[test]
fn catmull_rom_emits_one_point_per_segment_boundary_per_span() {
    let ctrl = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
    ];
    // Three spans, and the bound is inclusive, so each contributes segs + 1.
    for segs in 1..6u32 {
        assert_eq!(
            catmull_rom_pts(&ctrl, segs).count(),
            3 * (segs as usize + 1),
            "segs_per_span = {segs}"
        );
    }
}

#[test]
fn catmull_rom_interpolates_through_its_control_points() {
    let ctrl = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    let pts: Vec<_> = catmull_rom_pts(&ctrl, 4).collect();

    // t = 0 of the first span is the first control point, and t = 1 of the
    // last span is the last one — Catmull-Rom interpolates its controls.
    let first = pts.first().expect("at least one point");
    let last = pts.last().expect("at least one point");
    for k in 0..3 {
        assert!((first[k] - ctrl[0][k]).abs() < 1e-9, "start component {k}");
        assert!((last[k] - ctrl[2][k]).abs() < 1e-9, "end component {k}");
    }
}

#[test]
fn catmull_rom_is_empty_below_two_control_points() {
    assert_eq!(catmull_rom_pts(&[], 4).count(), 0);
    assert_eq!(catmull_rom_pts(&[[0.0, 0.0, 0.0]], 4).count(), 0);
}

// ── emit_wire_packed ──────────────────────────────────────────────────────

fn wire_through(points: Vec<[f32; 3]>) -> WireModel {
    WireModel {
        points_low: vec![[0.0; 3]; points.len()],
        points,
        ..WireModel::default()
    }
}

#[test]
fn packed_wire_emits_one_instance_per_segment() {
    let wire = wire_through(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]);

    assert_eq!(emit_wire_packed(&wire, [1.0; 4], 0.0).count(), 3);
}

#[test]
fn packed_wire_skips_segments_touching_a_non_finite_point() {
    // A non-finite point is the break marker between disjoint runs; both
    // segments adjacent to it must be dropped.
    let wire = wire_through(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [f32::NAN, f32::NAN, f32::NAN],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ]);

    // Four segments, of which the two touching the NaN are skipped.
    assert_eq!(emit_wire_packed(&wire, [1.0; 4], 0.0).count(), 2);
}

#[test]
fn packed_wire_is_empty_for_a_degenerate_wire() {
    assert_eq!(emit_wire_packed(&wire_through(vec![]), [1.0; 4], 0.0).count(), 0);
    assert_eq!(
        emit_wire_packed(&wire_through(vec![[0.0, 0.0, 0.0]]), [1.0; 4], 0.0).count(),
        0
    );
}

#[test]
fn packed_wire_carries_the_endpoints_of_each_segment() {
    let wire = wire_through(vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
    let instances: Vec<_> = emit_wire_packed(&wire, [1.0; 4], 0.0).collect();

    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].pos_a, [0.0, 0.0, 0.0]);
    assert_eq!(instances[0].pos_b, [1.0, 2.0, 3.0]);
}
