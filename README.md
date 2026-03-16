# nbml

A minimal machine learning library built on `ndarray` for low-level ML algorithm development in Rust.

Unlike high-level frameworks, `nbml` provides bare primitives and a lightweight optimizer API for building custom neural networks from scratch. If you want comfortable abstractions, see [Burn](https://github.com/tracel-ai/burn). If you want to understand what's happening under the hood and have full control, `nbml` gives you the building blocks.

## Features

- **Core primitives**: Transformers, Attention, LSTM, Conv2D, Feedforward layers, etc
- **Activation functions**: ReLU, Sigmoid, Tanh, etc
- **Layers**: Softmax, LayerNorm, Sequence Pooling
- **Optimizers**: AdamW, SGD
- **Utilities**: Variable Sequence Batching, Gradient Clipping, Gumbel Softmax, Plots, etc
- **Minimal abstractions**: Direct ndarray integration for custom algorithms

## Quick Start
```rust
use nbml::nn::FFN;
use nbml::f::Activation;
use nbml::optim::{AdamW, Optimizer, ToParams};

// Build a simple feedforward network
let mut model = FFN::new(vec![
    (784, 12, Activation::Relu),
    (12, 1, Activation::Sigmoid),
]);

// Create optimizer
let mut optimizer = AdamW::default().with(&mut model);

// Training loop (simplified)
for batch in training_data {
    let output = model.forward(batch.x, true);
    let loss = cross_entropy(&output, &batch.y);
    let grad = model.backward(loss);
    optimizer.step(&mut model);
    model.zero_grads();
}
```

## Architecture

### NN Layers (`nbml::nn`)

- **`Layer`**: Single nonlinear projection layer
- **`FFN`**: Feedforward network with configurable layers
- **`LSTM`**: Long Short-Term Memory Network
- **`RNN`**: Vanilla recurrent neural network
- **`ESN`**: Echo-state network, fixed recurrence + readout
- **`LSM`**: Liquid state machine
- **`RNNReservoir`**: RNN reservoir (used by ESN)
- **`SNNReservoir`**: Spiking neural network reservoir (used by LSM)
- **`Conv2D`**: Explicit Im2Col Conv2D layer (CPU efficient, memory hungry)
- **`PatchwiseConv2D`**: Patchwise Conv2D layer (CPU hungry, memory efficient)
- **`LinearSSM`**: Discrete Linear SSM
- **`Attention`**: Core softmax attention primitive
- **`SelfAttention`**: Multi-head self attention
- **`CrossAttention`**: Multi-head cross attention
- **`LinearAttention`**: Linear self attention with recurrent matrix-valued state. Subquadratic alternative to softmax attention ([Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236))
- **`GatedLinearAttention`**: Gated linear attention with matrix-valued state and outer-product gating ([Yang et al., 2024](https://proceedings.mlr.press/v235/yang24ab.html))
- **`Transformer`**: Transformer encoder/decoder block
- **`LinearTransformer`**: Transformer block using linear self attention instead of softmax attention
- **`GlaTransformer`**: Transformer block using gated linear attention. Similar to Mamba, RWKV, RetNet, etc.

### Layers (`nbml::layers`)

Layers that are only useful as components of other modules:

- **`Linear`**: Affine transformation
- **`Softmax`**: Row-wise softmax
- **`LayerNorm`**: Layer normalization
- **`SequencePooling`**: Sequence mean-pooling

### Optimizers (`nbml::optim`)

#### `ToParams`

`ToParams` connects your model's weights and gradients to the optimizer. Implement `params()` to return a list of `Param` entries, each pairing a weight array with its gradient. The optimizer reads these pointers on each step to update weights in-place — no ownership transfer, no framework magic.

`Param::new` and `with_grad` accept any ndarray dimension (`Array1`, `Array2`, `Array3`, etc.), so you don't need separate methods per shape:

```rust
pub struct Affine {
    w: Array2<f64>,
    b: Array1<f64>,

    d_w: Array2<f64>,
    d_b: Array1<f64>,
}

impl ToParams for Affine {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w).with_grad(&mut self.d_w),
            Param::new(&mut self.b).with_grad(&mut self.d_b),
        ]
    }
}
```

Params compose — bubble them up from sub-modules to build arbitrary architectures:
```rust
pub struct AffineAffine {
    affine1: Affine,
    affine2: Affine,
}

impl ToParams for AffineAffine {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        params.append(&mut self.affine1.params());
        params.append(&mut self.affine2.params());
        params
    }
}
```

`ToParams` also provides `zero_grads()` to reset all gradient arrays after an optimizer step:
```rust
let mut aa = AffineAffine::new();
aa.forward(x, true) // <- implement this yourself
aa.backward(d_loss) // <- implement this yourself
optimizer.step(&mut aa);
aa.zero_grads();
```

Available optimizers:
- **`AdamW`**: Adaptive moment estimation with weight decay
- **`SGD`**: Stochastic gradient descent

Use `.with(&mut impl ToParams)` to initialize a stateful optimizer (like AdamW) for your network:
```rust
let mut model = AffineAffine::new();
let mut optim = AdamW::default().with(&mut model); // creates momentum/variance state for all parameters
```

#### `ToIntermediates`

`ToIntermediates` lets you snapshot and restore a module's cached activations (the values stored during `forward()` that `backward()` needs for gradient computation). This enables training loops that aren't possible in standard frameworks:

- **Recursive / weight-tied depth**: Forward the same module N times, stashing intermediates between each call. During backward, restore each stash in reverse to compute correct weight gradients for every application.
- **Online learning with rollback**: Checkpoint recurrent state mid-sequence, run an optimizer step, then restore and continue from the checkpoint.

Implement `intermediates()` to return mutable references to your cached values:

```rust
impl ToIntermediates for MyLayer {
    fn intermediates(&mut self) -> Vec<&mut dyn Intermediate> {
        vec![&mut self.cache.x, &mut self.cache.z]
    }
}
```

Then `stash_intermediates()` and `apply_intermediates()` work automatically:

```rust
let mut model = MyLayer::new();

// Forward pass A
model.forward(x_a, true);
let stash_a = model.stash_intermediates();

// Forward pass B (overwrites cache)
model.forward(x_b, true);
model.backward(d_loss_b); // correct grads for B

// Restore A's cache, compute A's grads
model.apply_intermediates(stash_a);
model.backward(d_loss_a); // correct grads for A
```

Intermediates are returned from `stash_intermediates` as `Vec<ArrayD<T>>` - aliased as `IntermediateCache`.

### Activation Functions (`nbml::f`)
```rust
use nbml::f;

let x = Array2::from_vec(vec![-1.0, 0.0, 1.0]);
let activated = f::relu(&x);
```

Includes derivatives for backpropagation: `d_relu`, `d_tanh`, `d_sigmoid`, etc.

## Design Philosophy

`nbml` is designed for:

- **Experimentation / Research**: Prototyping of novel architectures, through full control of forward and backward passes
- **Nonstandard Architectures**: A lot more freedom without autograd running the show
- **Transparency**: No hidden magic, every operation is explicit
- **Compute-Constrained Deployment**: Lightweight + no C deps. Very quick for small models

`nbml` is **not** designed for:

- Large Scale Production deployment (use PyTorch, TensorFlow, or Burn)
- Automatic differentiation (you wire up the backward pass for custom modules)
- GPU acceleration (CPU-only via ndarray)

The included `nn` primitives are technically plug and play, but when composing them you will have to wire `backward()` yourself.

```rust
pub struct SequenceClassifier {
    pub transformer: GlaTransformer,
    pub pooling: SequencePooling,
    pub readout: Layer,
}

impl SequenceClassifier {
    pub fn new(d_model: usize) -> Self { ... }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array2<f64> {
        // on-the-spot mask
        let mask = Array3::ones((x.dim().0, x.dim().1, x.dim().1));
        let x = self.transformer.forward(x, mask.clone(), grad); // (B, S, D)
        let x = self.pooling.forward(x, mask); // (B, D)
        let x = self.readout.forward(x); // (B, D)

        x
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array3<f64> {
        let d_loss = self.readout.backward(d_loss); // (B, D)
        let d_loss = self.pooling.backward(d_loss); // (B, S, D)
        let d_loss = self.transformer.backward(d_loss); // (B, S, D)

        d_loss
    }
}
```

## Examples

### Custom LSTM Training
```rust
use nbml::nn::LSTM;
use nbml::optim::{AdamW, Optimizer, ToParams};

let mut lstm = LSTM::new(
    128     // d_model or feature dimension
);
let mut optimizer = AdamW::default().with(&mut lstm);

// where batch.dim() is (batch_size, seq_len, features)
// and features == lstm.d_model == (128 in this case)

for batch in data {
    let output = lstm.forward(batch, true);
    let loss = compute_loss(&output, &target);
    let grad = lstm.backward(loss);
    optimizer.step(&mut lstm);
    lstm.zero_grads();
}
```

### Multi-Head Attention
```rust
use nbml::nn::SelfAttention;

let mut attention = SelfAttention::new(
    512,  // d_in
    64,   // d_head
    8     // n_head
);

// where input.dim() is (batch_size, seq_len, features)
// features == d_in == (512 in this case)
// and mask == (batch_size, seq_len, seq_len)
// with each element as 1. or 0. depending on whether or not the token
// is padding

let output = attention.forward(
    input, // (batch_size, seq_len, features)
    mask,  // binary mask, (batch_size, seq_len, seq_len)
    true    // grad
);
```

### Transformer Decoder
```rust
use nbml::nn::Transformer;
use nbml::ndarray::Array3;

let mut transformer = Transformer::new_decoder(
    512,  // d_in
    64,   // d_head
    8,    // n_head
);

let y_pred = transformer.forward(
    input, // (batch_size, seq_len, features)
    mask,  // binary mask, (batch_size, seq_len, seq_len)
    true    // grad
);

// some bs.
let d_y_pred = Array3::ones(y_pred.dim());
transformer.backward(d_y_pred);
transformer.zero_grads();
```
