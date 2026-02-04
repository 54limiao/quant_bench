#!/bin/bash

cd "$(dirname "$0")"

python run_pipeline.py \
  --trans-config pipline/1-trans/example/original.yaml pipline/1-trans/example/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/example/gptq-w4a8-dynamic.yaml \
                 pipline/2-quant/example/gptq-w4a16-g128.yaml \
                 pipline/2-quant/example/r4-gptq-w4a8-dynamic.yaml \
                 pipline/2-quant/example/r4-gptq-w4a8-static-mix-precision.yaml \
                 pipline/2-quant/example/r4-gptq-w4a8-static.yaml \
                 pipline/2-quant/example/r4-gptq-w4a16-g128.yaml \
                 pipline/2-quant/example/r4-gptq-w4a16.yaml \
  --eval-config pipline/3-eval/example/mmlu_gsm8k.yaml