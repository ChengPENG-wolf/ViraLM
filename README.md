# ViraLM

ViraLM is a Python library for virus identification from metagenomic data.

ViraLM is based on the pre-trained genome foundation model and relies on nucleotide information for prediction.

# Overview

## Required Dependencies

Detailed package information can be found in `viralm.yaml`


## Quick install

*Note*: we suggest you to install all the package using conda ([miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://anaconda.org/)).

Clone the repository to local:

```
git clone https://github.com/ChengPENG-wolf/ViraLM.git
cd ViraLM
```

Install the environment for ViraLM:

```
conda env create -f viralm.yaml -n viralm
```

Once installed, you only need to activate your 'viralm' environment before using ViraLM in the next time.

```
conda activate viralm
```

Download and setup the model

```
gdown --id 1EQVPmFbpLGrBLU0xCtZBpwvXrtrRxic1
tar -xzvf model.tar.gz -C .
rm model.tar.gz
```



## Usage

### Run ViraLM in one command:
*Note*: we recommend to run ViraLM on GPU to accelerate computation.

```
python viralm.py [--input INPUT_FA] [--output OUTPUT_PTH] [--len MINIMUM_LEN] [--threshold THRESHOLD]
```

**Options**

```
--input INPUT_FA
                        The path of your input fasta file
  --output OUTPUT_PTH
			The path of your output directory
  --len MINIMUM_LEN
                        predict only for sequence >= len bp (default 500)
  --threshold THRESHOLD
                        Threshold to reject (default 0.5).
```

**Example**

Prediction on the example file:

```
python viralm.py --input test.fasta --out result --len 500 --threshold 0.5
```

### Contact

If you have any questions, please email us: cpeng29-c@my.cityu.edu.hk


