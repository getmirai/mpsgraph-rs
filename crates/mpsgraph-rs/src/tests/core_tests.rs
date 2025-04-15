use crate::core::{
    MPSDataType, ExecutionStage, Optimization, OptimizationProfile,
    Options,
};

#[test]
fn test_mps_data_type() {
    // Test all data types
    let types = [
        MPSDataType::Invalid,
        MPSDataType::Float32,
        MPSDataType::Float16,
        MPSDataType::Int32,
        MPSDataType::Bool,
        MPSDataType::Int8,
        MPSDataType::UInt8,
        MPSDataType::Int16,
        MPSDataType::Float64,
        MPSDataType::Int64,
        MPSDataType::UInt32,
    ];

    for data_type in &types {
        // Just ensure no crashes when getting string representation
        let _desc = format!("{:?}", data_type);
    }
}

#[test]
fn test_graph_options() {
    // Initialize options with default value
    let options = Options::Default;

    // Test with various option combinations
    let with_verbose = Options::Verbose;
    let with_sync = Options::SynchronizeResults;

    // Ensure enums have their values
    assert_eq!(with_verbose as u64, 2);
    assert_eq!(with_sync as u64, 1);
    assert_eq!(Options::None as u64, 0);
    assert_eq!(Options::Default as u64, 3);

    // Test string representation
    let _desc = format!("{:?}", options);
}

#[test]
fn test_graph_optimization() {
    // Test all optimization values
    let opts = [Optimization::Level0, Optimization::Level1];

    for opt in &opts {
        // Check string representation
        let _desc = format!("{:?}", opt);

        // Check raw value is within expected range
        let val = *opt as u64;
        assert!(val <= 1, "Optimization level should be <= 1, got {}", val);
    }
}

#[test]
fn test_execution_stage() {
    // Test execution stage
    let stage = ExecutionStage::Completed;

    // Check string representation
    let _desc = format!("{:?}", stage);

    // Check raw value
    let val = stage as u64;
    assert_eq!(val, 0, "Execution stage Completed should be 0, got {}", val);
}

#[test]
fn test_optimization_profile() {
    // Test optimization profile values
    let profile = OptimizationProfile::Performance;
    assert_eq!(profile as u64, 0);

    let profile = OptimizationProfile::PowerEfficiency;
    assert_eq!(profile as u64, 1);

    // Test string representation
    let _desc = format!("{:?}", profile);
}
