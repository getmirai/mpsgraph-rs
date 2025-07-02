use crate::rnn_ops::{SingleGateRNNDescriptor, LSTMDescriptor, GRUDescriptor, RNNActivation};
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::DataType;
use crate::device::CustomDefault;

#[test]
fn test_single_gate_rnn_descriptor() {
    // Create a SingleGateRNNDescriptor with default settings
    let descriptor = SingleGateRNNDescriptor::new();
    
    // Configure the descriptor
    descriptor
        .set_bidirectional(true)
        .set_reverse(true)
        .set_training(true)
        .set_activation(RNNActivation::TanH);
    
    // Verify the settings
    assert_eq!(descriptor.bidirectional(), true);
    assert_eq!(descriptor.reverse(), true);
    assert_eq!(descriptor.training(), true);
    assert_eq!(descriptor.activation(), RNNActivation::TanH);
}

#[test]
fn test_lstm_descriptor() {
    // Create an LSTMDescriptor with default settings
    let descriptor = LSTMDescriptor::new();
    
    // Configure the descriptor
    descriptor
        .set_bidirectional(true)
        .set_reverse(true)
        .set_training(true)
        .set_produce_cell(true)
        .set_forget_gate_last(true)
        .set_activation(RNNActivation::TanH)
        .set_input_gate_activation(RNNActivation::Sigmoid)
        .set_forget_gate_activation(RNNActivation::Sigmoid)
        .set_cell_gate_activation(RNNActivation::TanH)
        .set_output_gate_activation(RNNActivation::Sigmoid);
    
    // Verify the settings
    assert_eq!(descriptor.bidirectional(), true);
    assert_eq!(descriptor.reverse(), true);
    assert_eq!(descriptor.training(), true);
    assert_eq!(descriptor.produce_cell(), true);
    assert_eq!(descriptor.forget_gate_last(), true);
    assert_eq!(descriptor.activation(), RNNActivation::TanH);
    assert_eq!(descriptor.input_gate_activation(), RNNActivation::Sigmoid);
    assert_eq!(descriptor.forget_gate_activation(), RNNActivation::Sigmoid);
    assert_eq!(descriptor.cell_gate_activation(), RNNActivation::TanH);
    assert_eq!(descriptor.output_gate_activation(), RNNActivation::Sigmoid);
}

#[test]
fn test_gru_descriptor() {
    // Create a GRUDescriptor with default settings
    let descriptor = GRUDescriptor::new();
    
    // Configure the descriptor
    descriptor
        .set_bidirectional(true)
        .set_reverse(true)
        .set_training(true)
        .set_reset_gate_first(true)
        .set_reset_after(true)
        .set_flip_z(true)
        .set_update_gate_activation(RNNActivation::Sigmoid)
        .set_reset_gate_activation(RNNActivation::Sigmoid)
        .set_output_gate_activation(RNNActivation::TanH);
    
    // Verify the settings
    assert_eq!(descriptor.bidirectional(), true);
    assert_eq!(descriptor.reverse(), true);
    assert_eq!(descriptor.training(), true);
    assert_eq!(descriptor.reset_gate_first(), true);
    assert_eq!(descriptor.reset_after(), true);
    assert_eq!(descriptor.flip_z(), true);
    assert_eq!(descriptor.update_gate_activation(), RNNActivation::Sigmoid);
    assert_eq!(descriptor.reset_gate_activation(), RNNActivation::Sigmoid);
    assert_eq!(descriptor.output_gate_activation(), RNNActivation::TanH);
}

// This test only compiles the API but doesn't run actual computations
#[test]
fn test_rnn_api_compiles() {
    let graph = Graph::new();
    
    // Create a batch of 2 sequences, each 10 time steps, with 4 features
    let input_shape = Shape::tensor3d(10, 2, 4);
    let hidden_dim = 8;
    let recurrent_weight_shape = Shape::matrix(hidden_dim, hidden_dim);
    let input_weight_shape = Shape::matrix(hidden_dim, 4);
    
    // Test single-gate RNN API
    let _ = |input, recurrent_weight, input_weight, bias, init_state, mask| {
        let descriptor = SingleGateRNNDescriptor::new();
        descriptor.set_activation(RNNActivation::TanH);
        
        // Test with mask
        let rnn_outputs = graph.single_gate_rnn_with_mask(
            input,
            recurrent_weight,
            Some(input_weight),
            Some(bias),
            Some(init_state),
            Some(mask),
            &descriptor,
            Some("rnn_with_mask")
        );
        
        // Test without mask
        let rnn_outputs2 = graph.single_gate_rnn(
            input,
            recurrent_weight,
            Some(input_weight),
            Some(bias),
            Some(init_state),
            &descriptor,
            Some("rnn")
        );
        
        // Test minimal
        let rnn_outputs3 = graph.single_gate_rnn_minimal(
            input,
            recurrent_weight,
            Some(init_state),
            &descriptor,
            Some("rnn_minimal")
        );
        
        (rnn_outputs, rnn_outputs2, rnn_outputs3)
    };
    
    // Test LSTM API
    let _ = |input, initial_hidden_state, initial_cell_state, weights, recurrent_weights, biases| {
        let descriptor = LSTMDescriptor::new();
        descriptor.set_input_gate_activation(RNNActivation::Sigmoid);
        
        let (output, hidden_state, cell_state) = graph.lstm(
            input,
            initial_hidden_state,
            initial_cell_state,
            weights,
            recurrent_weights,
            Some(biases),
            &descriptor,
            Some("lstm")
        );
        
        (output, hidden_state, cell_state)
    };
    
    // Test GRU API
    let _ = |input, initial_state, weights, recurrent_weights, biases| {
        let descriptor = GRUDescriptor::new();
        descriptor.set_update_gate_activation(RNNActivation::Sigmoid);
        
        let (output, state) = graph.gru(
            input,
            initial_state,
            weights,
            recurrent_weights,
            Some(biases),
            &descriptor,
            Some("gru")
        );
        
        (output, state)
    };
}