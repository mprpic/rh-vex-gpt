# VEX Finetuning Experiment

Red Hat VEX data contains vulnerability information per individual CVE and provides affectedness to Red Hat products and their components. Vulnerabilities include various pieces of metadata such as CVSS scores, CWE IDs, or textual statements and mitigations.

This data was parsed out of the CSAF JSON files at and converted into a training data set at: 🤗 [mprpic/rh-vex-data](https://huggingface.co/datasets/mprpic/rh-vex-data). The data was then used to finetune a model using low-rank adaptino (LoRA) to produce the following artifacts:

- 🤗 [mprpic/rh-vex-mistral-7b-adapter](https://huggingface.co/mprpic/rh-vex-mistral-7b-adapter)
- 🤗 [mprpic/rh-vex-mistral-7b-merged](https://huggingface.co/mprpic/rh-vex-mistral-7b-merged)

## Finetuning

On a machine with at least a 24GB VRAM GPU:

```shell
git clone https://github.com/mprpic/rh-vex-gpt.git && cd rh-vex-gpt/
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
echo HF_TOKEN=<TOKEN> >> .env
```
