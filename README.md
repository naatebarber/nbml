# nbml

A minimal machine learning library built on `ndarray` for low-level ML algorithm development in Rust.

Unlike high-level frameworks, `nbml` provides bare primitives and a lightweight optimizer API for building custom neural networks from scratch. If you want comfortable abstractions, see [Burn](https://github.com/tracel-ai/burn). If you want to understand what's happening under the hood and have full control, `nbml` gives you the building blocks.

## Features

- **Core primitives**: Attention, LSTM, RNN, Conv2D, Feedforward layers, etc
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax, etc
- **Optimizers**: AdamW, SGD
- **Utilities**: Variable Sequence Batching, Gradient Clipping, Gumbel Softmax, Plots, etc
- **Minimal abstractions**: Direct ndarray integration for custom algorithms

## Quick Start
```rust
use nbml::layers::ffn::FFN;
use nbml::f::Activation;
use nbml::optim::adam::AdamW;
use nbml::optim::param::ToParams;

// Build a simple feedforward network
let mut model = FFN::new(vec![(
    (784, 12, Activation::Relu),
    (12, 1, Activation::Sigmoid)
)]);

// Create optimizer
let mut optimizer = AdamW::default().with(&mut model);

// Training loop (simplified)
for batch in training_data {
    let output = model.forward(batch.x, true);
    let loss = cross_entropy(&output, &batch.y);
    let grad = model.backward(loss);
    optimizer.step();
    model.zero_grad();
}
```

## Architecture

### NN Layers (`nbml::nn`)

- **`Layer`**: Single nonlinear projection layer
- **`FFN`**: Feedforward network with configurable layers
- **`LSTM`**: Long Short-Term Memory with merged weight matrices
- **`RNN`**: Vanilla recurrent neural network
- **`LayerNorm`**: Layer normalization
- **`Pooling`**: Sequence mean-pooling
- **`AttentionHead`**: Multi-head self-attention mechanism
- **`TransformerEncoder`**: Pre-norm transformer encoder
- **`TransformerDecoder`**: Pre-norm transformer decoder
- **`Conv2D`**: Explicit Im2Col Conv2D layer (CPU efficient, memory hungry)
- **`PatchwiseConv2D`**: Patchwise Conv2D layer (CPU hungry, memory efficient)

### Optimizers (`nbml::optim`)

Implement the `ToParams` trait for gradient-based optimization:
```rust
pub struct Affine {
    w: Array2<f64>,
    b: Array1<f64>,

    d_w: Array2<f64>,
    d_b: Array1<f64>,
}

// impl Affine {}

impl ToParams for Affine {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::matrix(&mut self.w).with_matrix_grad(&self.d_w),
            Param::vector(&mut self.b).with_vector_grad(&self.d_b),
        ]
    }
}
```

You can bubble params up:
```rust
pub struct AffineAffine {
    affine1: Affine,
    affine2: Affine,
}

// impl AffineAffine {}

impl ToParams for AffineAffine {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        params.append(&mut self.affine1.params());
        params.append(&mut self.affine2.params());
        params
    }
}
```

`ToParams` will also let you zero gradients:
```rust
let mut aa = AffineAffine::new();
aa.forward(x, true) // <- implement this yourself
aa.backward(d_loss) // <- implement this yourself
aa.zero_grads();
```

Available optimizers:
- **`AdamW`**: Adaptive moment estimation with bias correction
- **`SGD`**: Stochastic gradient descent with optional momentum

Use `.with(&mut impl ToParams)` to prepare a stateful optimizer (like AdamW) for your network:
```rust
let mut model = AffineAffine::new();
let mut optim = AdamW::default().with(&mut model); // <- adamw creates momentums, values for all parameters in Model
```

### Activation Functions (`nbml::f`)
```rust
use nbml::f;

let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
let activated = f::relu(&x);
let softmax = f::softmax(&x);
```

Includes derivatives for backpropagation: `d_relu`, `d_tanh`, `d_sigmoid`, etc.

## Design Philosophy

`nbml` is designed for:

- **Experimentation / Research**: Prototyping of novel architectures, through full control of forward and backward passes
- **Transparency**: No hidden magic, every operation is explicit
- **Compute-Constrained Deployment**: Lightweight + no C deps. Very quick for small models.

`nbml` is **not** designed for:

- Large Scale Production deployment (use PyTorch, TensorFlow, or Burn)
- Automatic differentiation (you write the backward pass)
- GPU acceleration (CPU-only via ndarray)
- Plug-and-play models (you build everything yourself)

## Examples

### Custom LSTM Training
```rust
use nbml::layers::lstm::LSTM;
use nbml::optim::adam::Adam;

let mut lstm = LSTM::new(
    128     // d_model or feature dimension
);
let mut optimizer = Adam::default().with(&mut lstm);

// where batch.dim() is (batch_size, seq_len, features)
// and features == lstm.d_model == (128 in this case)

for batch in data {
    let output = lstm.forward(batch, true);
    let loss = compute_loss(&output, &target);
    let grad = lstm.backward(loss);
    optimizer.step();
    lstm.zero_grads();
}
```

### Multi-Head Attention
```rust
use nbml::layers::attention::AttentionHead;

let mut attention = AttentionHead::new(
    512,  // d_in
    64,   // d_head
    8     // n_head
);

// where input.dim() is (batch_size, seq_len, features)
// features == d_in == (512 in this case)
// and mask == (batch_size, seq_len)
// with each element as 1. or 0. depending on whether or not the token
// is padding

let output = attention.forward(
    input, // (batch_size, seq_len, features)
    mask,  // binary mask, (batch_size, seq_len)
    false,  // include causal mask (is this a decoder?)
    true    // grad
);
```
