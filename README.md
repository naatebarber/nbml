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
use nbml::f::{self, Activation};
use nbml::nn::FFN;
use nbml::optim::{AdamW, Optimizer, ToParams};

// Build a simple feedforward network
let mut model = FFN::new(vec![
    (784, 12, Activation::Relu),
    (12, 1, Activation::Sigmoid),
]);

// Create optimizer
let mut optimizer = AdamW::default().with(&mut model);

// Training loop, where x is (batch_size, 784) and y is (batch_size, 1)
for (x, y) in training_data {
    let y_pred = model.forward(x, true);           // grad = true, cache activations
    let (loss, d_loss) = f::bce_loss(&y_pred, &y); // scalar loss + dL/dy_pred
    model.backward(d_loss);
    optimizer.step(&mut model);
    model.zero_grads();
}
```

Losses in `nbml::f` all return `(loss, d_loss)` — the scalar you log and the gradient you feed
straight back into `backward()`. See `f::mse_loss`, `f::bce_loss`, `f::cross_entropy_loss`, etc, and
their `batch_*` counterparts for 3D (batched sequence) inputs.

## Architecture

### NN Layers (`nbml::nn`)

- **`Layer`**: Single nonlinear projection layer
- **`FFN`**: Feedforward network with configurable layers
- **`PositionwiseFFN`**: Two-layer position-wise feedforward block (used by the transformers)
- **`RNN`**: Vanilla recurrent neural network
- **`LSTM`**: Long Short-Term Memory Network
- **`GRU`**: Gated Recurrent Unit
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
- **`GATv2`**: Sparse v2 graph attention ([Brody et al., 2022](https://arxiv.org/abs/2105.14491))
- **`LinearAttention`**: Linear self attention with recurrent matrix-valued state. Subquadratic alternative to softmax attention ([Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236))
- **`GatedLinearAttention`**: Gated linear attention with matrix-valued state and outer-product gating ([Yang et al., 2024](https://proceedings.mlr.press/v235/yang24ab.html))
- **`DeltaNet`**: Linear attention with delta rule for error-corrected state updates ([Yang et al., 2024](https://arxiv.org/abs/2406.06484))
- **`GatedDeltaNet`**: Gated delta network combining data-dependent gating with delta rule updates ([Yang et al., 2025](https://arxiv.org/abs/2412.06464))
- **`Transformer`**: Transformer encoder/decoder block
- **`LinearTransformerBase<A>`**: Transformer block generic over any `LinearAttentionLike` mixer, with these aliases:
  - **`LinearTransformer`**: `LinearTransformerBase<LinearAttention>`
  - **`GlaTransformer`**: `LinearTransformerBase<GatedLinearAttention>`
  - **`DeltaNetTransformer`**: `LinearTransformerBase<DeltaNet>`
  - **`GdnTransformer`**: `LinearTransformerBase<GatedDeltaNet>`

`nbml::nn::experimental` holds work-in-progress modules (`SelectiveSSM`, `DenseSelectiveSSM`,
`LinearCrossAttention`). It is not re-exported from `nbml::nn`, so import it by full path. Expect
its API to change.

### Layers (`nbml::layers`)

Layers that are only useful as components of other modules:

- **`Dropout`**: Stateful dropout layer
- **`Embedding`**: Learnable token embeddings
- **`L2Norm`**: L2 normalization
- **`Linear`**: Affine transformation
- **`Softmax`**: Row-wise softmax
- **`LayerNorm`**: Layer normalization
- **`SequencePooling`**: Sequence mean-pooling
- **`LastTokenPooling`**: Last-token pooling, selects last token output only. For causal tasks.

### Modeling (`nbml::modeling`)

Thin wrappers that assemble primitives into a full model.

- **`CausalLM<M>`**: Embedding → sequence model → LM head, with `train_step()` wiring up cross
  entropy. `M` is any type you implement `CausalSequenceModel` for (see the example below).

`nbml` also re-exports `ndarray`, `ndarray_rand`, and `ndarray_stats`, so you don't have to pin
matching versions yourself: `use nbml::ndarray::Array3;`

### Optimizers (`nbml::optim`)

#### `ToParams`

`ToParams` connects your model's weights and gradients to the optimizer. Implement `params()` to return a list of `Param` entries, each pairing a weight array with its gradient. The optimizer reads these pointers on each step to update weights in-place — no ownership transfer, no framework magic.

`Param::new` and `with_grad` accept any ndarray dimension (`Array1`, `Array2`, `Array3`, etc.), so you don't need separate methods per shape:

```rust
pub struct Affine {
    w: Array2<f32>,
    b: Array1<f32>,

    d_w: Array2<f32>,
    d_b: Array1<f32>,
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

Because `params()` already enumerates every weight in the tree, `ToParams` gets serialization for
free — `dump()`/`load()` move weights through `Vec<TensorData>`, and `persist()`/`restore()` do the
same via a bincode file:

```rust
let mut lstm = LSTM::new(128);
lstm.persist("./lstm.bin")?;

let mut restored = LSTM::new(128);
restored.restore("./lstm.bin")?; // same architecture, weights copied in place
```

Available optimizers:
- **`AdamW`**: Adaptive moment estimation with weight decay
- **`SGD`**: Stochastic gradient descent

Use `.with(&mut impl ToParams)` to initialize a stateful optimizer (like AdamW) for your network:
```rust
let mut model = AffineAffine::new();
let mut optim = AdamW::default().with(&mut model); // creates momentum/variance state for all parameters
optim.learning_rate = 1e-3;
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

Intermediates are returned from `stash_intermediates` as `Vec<ArrayD<f32>>` - aliased as `IntermediateCache`.

### Activation Functions (`nbml::f`)
```rust
use nbml::f;
use nbml::ndarray::array;

let x = array![[-1.0, 0.0, 1.0]]; // activations operate on 2D arrays
let activated = f::relu(&x);
```

Includes derivatives for backpropagation: `d_relu`, `d_tanh`, `d_sigmoid`, etc.

`f::Activation` is the enum form used by `Layer`/`FFN`. Each variant carries its forward fn,
derivative, and weight initializer: `Relu`, `LeakyRelu`, `SoftmaxCrossEntropy`, `Identity`, `Exp`,
`Sigmoid`, `Tanh`, `Softplus`.

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
use nbml::f::Activation;
use nbml::layers::SequencePooling;
use nbml::ndarray::{Array2, Array3};
use nbml::nn::{GlaTransformer, Layer};
use nbml::optim::{Param, ToParams};

pub struct SequenceClassifier {
    pub transformer: GlaTransformer,
    pub pooling: SequencePooling,
    pub readout: Layer,
}

impl SequenceClassifier {
    pub fn new(d_model: usize, d_head: usize, n_class: usize) -> Self {
        Self {
            transformer: GlaTransformer::new(d_model, d_head),
            pooling: SequencePooling::new(),
            readout: Layer::new(d_model, n_class, Activation::SoftmaxCrossEntropy),
        }
    }

    // mask is the (B, S) pad mask, 1. for real tokens and 0. for padding.
    // Linear attention is inherently causal, so there is no attention mask to build.
    pub fn forward(&mut self, x: Array3<f32>, mask: Array2<f32>, grad: bool) -> Array2<f32> {
        let x = self.transformer.forward(x, grad);   // (B, S, D)
        let x = self.pooling.forward(x, mask, grad); // (B, D)
        self.readout.forward(x, grad)                // (B, n_class)
    }

    pub fn backward(&mut self, d_loss: Array2<f32>) -> Array3<f32> {
        let d_loss = self.readout.backward(d_loss);  // (B, D)
        let d_loss = self.pooling.backward(d_loss);  // (B, S, D)
        self.transformer.backward(d_loss)            // (B, S, D)
    }
}

impl ToParams for SequenceClassifier {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        params.append(&mut self.transformer.params());
        params.append(&mut self.readout.params());
        params
    }
}
```

## Examples

### Custom LSTM Training
```rust
use nbml::f;
use nbml::nn::LSTM;
use nbml::optim::{AdamW, Optimizer, ToParams};

let mut lstm = LSTM::new(
    128     // d_model or feature dimension
);
let mut optimizer = AdamW::default().with(&mut lstm);

// where batch.dim() is (batch_size, seq_len, features)
// and features == lstm.d_model == (128 in this case)

for (batch, target) in data {
    let output = lstm.forward(batch, true);
    let (loss, d_loss) = f::batch_mse_loss(output, target);
    lstm.backward(d_loss);
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
// is attendable

let output = attention.forward(
    input, // (batch_size, seq_len, features)
    mask,  // binary mask, (batch_size, seq_len, seq_len)
    true   // grad
);
```

### Transformer Decoder
```rust
use nbml::nn::Transformer;
use nbml::ndarray::Array3;
use nbml::optim::ToParams;

let mut transformer = Transformer::new_decoder(
    512,  // d_in
    64,   // d_head
    8,    // n_head
);

// Transformer takes a 2D pad mask, not a full attention mask. It broadcasts the
// pad mask to (batch_size, seq_len, seq_len) internally, and for decoders it
// intersects that with a causal mask for you. `pad_mask()` and `causal_mask()`
// are public if you want the expanded masks yourself.
let y_pred = transformer.forward(
    input,    // (batch_size, seq_len, features)
    pad_mask, // binary pad mask, (batch_size, seq_len)
    true      // grad
);

// some bs.
let d_y_pred = Array3::ones(y_pred.dim());
transformer.backward(d_y_pred);
transformer.zero_grads();
```

### KV Cached Decoding

`Transformer::forward_cached` (decoders only) appends to caller-owned K/V caches so incremental
decoding doesn't re-attend over the whole prefix. It runs with `grad = false` throughout — it's for
inference, not training.

```rust
use nbml::nn::Transformer;
use nbml::ndarray::{Array2, Array3, Axis, concatenate, s};

let (batch_size, seq_len, d_model, d_head, n_head) = (2, 8, 64, 16, 4);
let mut model = Transformer::new_decoder(d_model, d_head, n_head);

// x is (batch_size, seq_len, d_model)

// caches are (batch_size * n_head, cached_seq_len, d_head), starting empty
let mut k_cache = Array3::zeros((batch_size * n_head, 0, d_head));
let mut v_cache = Array3::zeros((batch_size * n_head, 0, d_head));

let mut y = Array3::zeros((batch_size, 0, d_model));
for t in 0..seq_len {
    let x_t = x.slice(s![.., t..t + 1, ..]).to_owned();
    let pad_mask = Array2::ones((batch_size, 1));

    // caches grow by one step per call
    let y_t = model.forward_cached(x_t, pad_mask, &mut k_cache, &mut v_cache);
    y = concatenate![Axis(1), y.view(), y_t.view()];
}
```

`SelfAttention::forward_cached` and `Attention::forward_cached` expose the same mechanism one level
down, if you're assembling your own decoder.

### Causal Language Model

`CausalLM<M>` bolts an `Embedding` and an LM head onto any sequence model that implements
`CausalSequenceModel`. Because of the orphan rule you implement that trait on your own type:

```rust
use nbml::modeling::causal_lm::{CausalLM, CausalSequenceModel};
use nbml::ndarray::{Array2, Array3};
use nbml::nn::Transformer;
use nbml::optim::{AdamW, Optimizer, Param, ToParams};

pub struct Decoder {
    blocks: Vec<Transformer>,
}

impl CausalSequenceModel for Decoder {
    fn new(d_model: usize) -> Self {
        Self {
            blocks: (0..4)
                .map(|_| Transformer::new_decoder(d_model, d_model / 4, 4))
                .collect(),
        }
    }

    fn forward(&mut self, mut x: Array3<f32>, mask: Array2<f32>, grad: bool) -> Array3<f32> {
        for block in self.blocks.iter_mut() {
            x = block.forward(x, mask.clone(), grad);
        }
        x
    }

    fn backward(&mut self, mut d_loss: Array3<f32>) -> Array3<f32> {
        for block in self.blocks.iter_mut().rev() {
            d_loss = block.backward(d_loss);
        }
        d_loss
    }
}

impl ToParams for Decoder {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        self.blocks
            .iter_mut()
            .for_each(|b| params.append(&mut b.params()));
        params
    }
}

let mut lm = CausalLM::<Decoder>::new(32_000, 256); // vocab_size, d_model
let mut optim = AdamW::default().with(&mut lm);

// tokens / targets are (batch_size, seq_len) token ids, pad_mask is (batch_size, seq_len)
let loss = lm.train_step(tokens, targets, pad_mask); // forward + cross entropy + backward
optim.step(&mut lm);
lm.zero_grads();
```

### Variable Length Batching
```rust
use nbml::util::batch_with_mask;

// sequences: Vec<Array2<f32>>, each (seq_len_i, features)
let (x, pad_mask) = batch_with_mask(sequences);
// x        -> (batch_size, max_seq_len, features), zero padded
// pad_mask -> (batch_size, max_seq_len), 1. for real steps and 0. for padding
```

### Runnable Examples

`examples/` has end-to-end programs you can run directly:

```
cargo run --example xor
```

## Features / Build Flags

- `blas` — routes ndarray's matmuls through BLAS. On macOS the Accelerate framework is wired up
  by default via `blas-src`.
