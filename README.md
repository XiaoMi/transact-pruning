# TransAct

This repository is the official implementation of the ACL 2024 paper [Pruning Large Language Models to Intra-module Low-rank Architecture with Transitional Activations](https://aclanthology.org/2024.findings-acl.582/).

| ![transact.png](assets/transact.png) | ![latency.png](assets/latency.png) |
| :----------------------------------: | :--------------------------------: |
|         Pruning architecture         | Latency on Xiaomi 14 mobile phone  |

## Training and Evaluation

1. Prepare environment following `transact.dockerfile`.
2. Create symlink to your data, models, outputs, and HuggingFace cache (mainly for large [datasets](https://github.com/huggingface/datasets)).
   ```sh
   ln -s /path/to/data data
   ln -s /path/to/models models
   ln -s /path/to/outputs outputs
   ln -s /path/to/hf-cache hf-cache
   ```
3. Tweak training configs `train_config.yaml` and `deepspeed.json`.
4. Run the train script `run_trainer.sh`, for example
   ```sh
   bash run_trainer.sh -m all \
     -a 768 -f 1536 \
     -n 128 -k 8 -p acts \
     -l 4096 -t 50 \
     -g 64 -b 4 \
     -d togethercomputer/RedPajama-Data-1T \
     -x llama -y llama2 -z 7B
   ```
   Run `bash run_trainer.sh -h` for help.
5. Run the evaluation script `eval.sh`, for example
   ```sh
   bash eval.sh -m all \
     -a 768 -f 1536 \
     -n 128 -k 8 -p acts \
     -l 4096 -t 50 \
     -d togethercomputer/RedPajama-Data-1T \
     -x llama -y llama2 -z 7B
   ```
   Run `bash eval.sh -h` for help.

## Citations

Please cite the paper if this repository is useful for you.

```
@inproceedings{shen-etal-2024-pruning,
    title = "Pruning Large Language Models to Intra-module Low-rank Architecture with Transitional Activations",
    author = "Shen, Bowen  and
      Lin, Zheng  and
      Zha, Daren  and
      Liu, Wei  and
      Luan, Jian  and
      Wang, Bin  and
      Wang, Weiping",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.582",
    doi = "10.18653/v1/2024.findings-acl.582",
    pages = "9781--9793",
}
```
