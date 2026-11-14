# Deep-learning design

## Supervised networks
1. **Deep MLP:** dense layers, batch normalization, ReLU, dropout and Huber loss.
2. **Residual MLP:** residual skip blocks to support optimization of a deeper dense network.
3. **1D CNN over feature vector:** convolutional filters over encoded tabular features. Feature ordering is an implementation convention; this is an experimental architecture and should be interpreted cautiously.

All three use the same train/test split, training target scaling, optimizer family and metrics for a fair baseline comparison. They are not guaranteed to outperform tree ensembles on small tabular datasets.

## Unsupervised network
A dense autoencoder compresses and reconstructs feature vectors. The 95th percentile of training reconstruction error is used as a review threshold. High error means unusual input patterns, not proof of a disease, pest outbreak or bad farm.

## Sequence learning
LSTM/GRU is deliberately not enabled by default. It requires timestamped repeated measurements grouped by farm/crop, chronological sorting and a sequence-aware split to avoid leakage. If such data becomes available, build a dedicated sequence forecasting module.
