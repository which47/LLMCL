<h1 align="center">
<span><i>LLMCL</i></span>
</h1>
<h3 align="center">
Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning
</h3>

## Overview
LLMCL is a repository based on the Hugging Face Transformers library, designed to assess the continuous learning capability of large language models. Through this repository, users can easily customize datasets, specify models, and experiment with existing classical continuous learning methods.

## Key Features
- **Continual Learning Methods:** The repository includes several classical continuous learning methods for users to reference and use.
- **Model Customization:** You can easily customize the model you want to use, and the repository will automatically download the corresponding model.

## Quick Start
### 1.Install dependencies
```bash
conda create -n llmcl python==3.10
pip install -r requirements.txt
```
### 2.Start Training
```bash
./scripts/train_seq.sh
```
### 3.Inference
```
./scripts/infer_seq.sh
```
### 4. customize
You can easily customize scripts for your own use:

- Ensure your dataset is organized in JSON format with `prompt` and `answer` as keys.
- Save the dataset file to `<DATA_PATH>/<DATASET_NAME>/<SPLIT>.json`
- For more details, refer to the [get_dataset.py](get_dataset.py) file.
     
## Reproduce
To Reproduce our results, you need \
**1.** Request the access to `llama2` model and download [TRACE Benchmark](https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view?usp=drive_link) , [MedMCQA](https://medmcqa.github.io/),[JEC-QA](https://jecqa.thunlp.org/) to `./data_files` folder.


2.run scripts
customize your training scripts and run it.





## Citation
If you find this repository helpful, please consider citing our work.

```bibtex
@misc{ren2024analyzing,
      title={Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning}, 
      author={Weijieying Ren and Xinlong Li and Lei Wang and Tianxiang Zhao and Wei Qin},
      year={2024},
      eprint={2402.18865},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
