<h1 align="center">
<span><i>LLMCL</i></span>
</h1>
<h3 align="center">
Parameter Efficient Continual Learning For Large language Models
</h3>

## Overview
LLMCL is a repository based on the Hugging Face Transformers library, designed to assess the continuous learning capability of large language models. Through this repository, users can easily customize datasets, specify models, and experiment with existing classical continuous learning methods.

## Key Features
- **Continual Learning Methods:** The repository includes several classical continuous learning methods for users to reference and use.
- **Model Customization:** You can easily customize the model you want to use, and the repository will automatically download the corresponding model.

## Quick Start
1.Clone the repository
```bash
git clone  https://github.com/which47/LLMCL.git
```

2.Install dependencies

```bash
pip install -r requirements.txt
```
3.Start Training

[//]: # (You can use our own scripts or modify it at your convenience.)
```bash
deepspeed main.py \
  --model_name_or_path 'meta-llama/Llama-2-7b-hf' \
  --output_dir "./outputs/models/seq" \
  --dataset_name "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa" \
  --per_device_train_batch_size 16 \
  --adapter lora
```

## Reproduce

1.Request the access to ```llama2``` model and download [TRACE Benchmark](https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view?usp=drive_link) , [MedMCQA](https://medmcqa.github.io/),[JEC-QA](https://jecqa.thunlp.org/) to `./data_files` folder.


2.run scripts
customize your training scripts and run it.





## Acknowledgement

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
