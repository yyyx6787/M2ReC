# M2ReC


> **M2Rec: Multi-scale Mamba with Adaptive FFT and LLMs for Sequential Recommendation**

## Usage

### Requirements

* Python 3.7+
* PyTorch 1.12+
* CUDA 11.6+
* Install RecBole:
  * `pip install recbole`
* Install causal Conv1d and the core Mamba package:
  * `pip install causal-conv1d>=1.2.0`
  * `pip install mamba-ssm`

You can also refer to the required environment specifications in `environment.yaml`.

### Run

```python run_mm.py```


Specifying the dataset in `config.yaml` will trigger an automatic download. Please set an appropriate maximum sequence length in `config.yaml` for each dataset before training.





