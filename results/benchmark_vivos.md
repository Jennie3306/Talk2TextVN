# ASR benchmark: vivos

Same preprocessing and text normalisation for every model. Compare WER only between rows with the same n;
compare speed only between rows on the same device.

| Model | n | WER | CER | sec / utterance | Real-time factor | Device |
|---|---|---|---|---|---|---|
| **phowhisper:vinai/PhoWhisper-small** | 760 | 0.062 | 0.027 | 0.844 | 0.239 | Tesla T4 |
| phowhisper:vinai/PhoWhisper-base | 760 | 0.080 | 0.036 | 0.361 | 0.102 | Tesla T4 |
| phowhisper:vinai/PhoWhisper-tiny | 760 | 0.096 | 0.043 | 0.242 | 0.069 | Tesla T4 |
| whisper:base | 760 | 0.382 | 0.190 | 0.245 | 0.069 | Tesla T4 |
