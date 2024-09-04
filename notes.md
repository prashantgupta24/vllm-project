curl http://localhost:8000/v1/completions \
 -H "Content-Type: application/json" \
 -d '{
"model": "/llama_eval_storage/LLaMa/models/hf/7B",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'

mkdir cache
export TRANSFORMERS_CACHE=cache
export HF_HUB_CACHE=cache
HF_HUB_OFFLINE=0 text-generation-server download-weights facebook/opt-125m
HF_HUB_OFFLINE=0 adapter download-weights facebook/opt-125m


cd /home/develop/.local/lib/python3.11/site-packages/vllm/model_executor/model_loader/


env:
  global:
    - REMOTE_INTEGRATION_TESTS=true
    - REMOTE_INTEGRATION_TEST_IMAGE=quay.io/dtrifiro/vllm:latest
    - REMOTE_INTEGRATION_TEST_CONFIG=product.vllm
  
env:
  global:
    - ORG_NAME=semantic-automation
    - REPO_NAME=dataagent_vllm
    - BRANCH_NAME=master
  
pip install git+https://github.com/opendatahub-io/vllm-tgis-adapter@fix-stat-logger
pip install git+https://github.com/opendatahub-io/vllm-tgis-adapter@ibm-release-prep

python3 -m vllm_tgis_adapter --model-name=/fmaas-integration-tests-pvc/transformers_cache/models--ibm-granite--granite-3b-code-instruct/snapshots/438ae81a509230148f962aaf669bdf42656a31ec


## Adapter stuff

### Prompt

#### Bloomz
export ADAPTER_CACHE=/fmaas-integration-tests-pvc/adapter_cache

python3 -m vllm_tgis_adapter --model=bigscience/bloomz-560m --enable-prompt-adapter --max_prompt_adapter_token=8

python3 -m vllm_tgis_adapter --model=/fmaas-integration-tests-pvc/transformers_cache/models--bigscience--bloomz-560m/snapshots/a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05/ --enable-prompt-adapter --max_prompt_adapter_token=8




grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "What a wonderful day!:\nSentiment:"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 20
      },
      "response": {
          "input_tokens": true,
          "generated_tokens": true,
          "token_logprobs": true,
          "token_ranks": true,
          "top_n_tokens": 2
        }
    },
    "adapter_id" : "bloomz-560m-prompt-adapter"
  }' \
  localhost:8033 fmaas.GenerationService/Generate


#### Bloom
export ADAPTER_CACHE=/fmaas-integration-tests-pvc/adapter_cache

python3 -m vllm_tgis_adapter --model=bigscience/bloom-560m --enable-prompt-adapter --max_prompt_adapter_token=8

grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "What a wonderful day!:\nSentiment:"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 20
      }
    },
    "adapter_id" : "bloom_sentiment_1"
  }' \
  localhost:8033 fmaas.GenerationService/Generate


  grpcurl -insecure -proto proto/generation.proto -d \
    '{
        "model_id": "dummy-model-name",
      "requests": [
          {
            "text": "Hello! My favorite Bible verse is "
        }
      ],
      "params": {
          "method": "SAMPLE",
        "sampling": {
    
        },
        "decoding": {
            "repetition_penalty": 2
        },
        "stopping": {
            "min_new_tokens": 16,
          "max_new_tokens": 20,
          "stop_sequences": ["Peter ", "Timothy ", "joseph", "Corinthians"]
        },
        "response": {
          "input_text": false,
          "input_tokens": true,
          "generated_tokens": false,
          "token_logprobs": false,
          "top_n_tokens": 10,
          "token_ranks": true
        }
      }
    }' \
    localhost:8001 fmaas.GenerationService/Generate

### LoRA stuff

cp /fmaas-integration-tests-pvc/tuning/output/granite-20b-multilingual-base/lora/20240730104547-472a8cd_ubi9_py311/*.* cache/granite-20b-multilingual-base-lora-lm-head/

<!-- cp /fmaas-integration-tests-pvc/tuning/output/granite-20b-multilingual-base/lora/20240722153815-191145a_ubi9_py311/*.* cache/granite-20b-multilingual-base-lora1/ -->

<!-- export ADAPTER_CACHE=cache -->
export ADAPTER_CACHE=/fmaas-integration-tests-pvc/tuning/output/granite-20b-multilingual-base/lora/

python3 -m vllm_tgis_adapter --model=/ibm_llm_alignment/granite.20b.ml5.lab.240217a --enable-lora


grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "Tweet text : @nationalgridus I have no water and the bill is current and paid. Can you do something about this? Label : "
      },
      {
        "text": "Tweet text : @nationalgridus Looks good thanks! Label : "
      }
    ],
    "params": {
      "method": "GREEDY"
    },
    "adapter_id" : "granite-20b-multilingual-base-lora-lm-head"
  }' \
  localhost:8033 fmaas.GenerationService/Generate


### Mixtral stuff

grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "Once upon a time,"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 4
      },
      "response": {
          "generated_tokens": true
        }
    }
  }' \
  localhost:8033 fmaas.GenerationService/Generate
### Qlora

export ADAPTER_CACHE=/fmaas-integration-tests-pvc/tuning/output/llama3-70b/qlora/cc_tone-20240806124230-anhs-qlora/

python3 -m vllm_tgis_adapter --model=/fmaas-integration-tests-pvc/models/llama3-70b-gptq --enable-lora 

grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "Tweet text : @nationalgridus I have no water and the bill is current and paid. Can you do something about this? Label : "
      },
      {
        "text": "Tweet text : @nationalgridus Looks good thanks! Label : "
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 20
      }
    },
    "adapter_id" : "checkpoint-1260"
  }' \
  localhost:8033 fmaas.GenerationService/Generate



lora_dir = "/fmaas-integration-tests-pvc/adapter_cache/granite-20b-multilingual-base-lora/"

import os
import safetensors.torch
lora_dir = "/fmaas-integration-tests-pvc/tuning/output/llama3-70b/qlora/cc_tone-20240806124230-anhs-qlora/checkpoint-1260"
lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
with safetensors.safe_open(lora_tensor_path,framework="pt") as f:
  lora_modules = [lora_module for lora_module in f.keys()]



## Tests
export VLLM_VERSION_OVERRIDE"=git+https://github.com/vllm-project/vllm@v0.5.3.post1"
export VLLM_CPU_DISABLE_AVX512=true
export VLLM_TARGET_DEVICE=cpu
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu


grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "Once upon a time,"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 20
      },
      "response": {
          "input_text": true
        }
    }
  }' \
  localhost:8033 fmaas.GenerationService/Generate

grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "Once upon a time,"
      },
      {
        "text": "When I was little,"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 20
      },
      "response": {
          "input_text": true
        }
    }
  }' \
  localhost:8033 fmaas.GenerationService/Generate

grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "request": [
      {
        "text": "Once upon a time,"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 4
      },
      "response": {
          "input_text": true
        }
    }
  }' \
  localhost:8033 fmaas.GenerationService/GenerateStream

# With headers
grpcurl -plaintext -proto proto/generation.proto -d \
  '{
    "model_id": "dummy-model-name",
    "requests": [
      {
        "text": "Once upon a time,"
      },
      {
        "text": "When I was little,"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 20
      },
      "response": {
          "input_text": true
        }
    }
  }' \
  localhost:8033 fmaas.GenerationService/Generate


grpcurl -plaintext -proto proto/generation.proto -H "x-correlation-id: 12345" -d \
  '{
    "model_id": "dummy-model-name",
    "request": [
      {
        "text": "Once upon a time,"
      }
    ],
    "params": {
      "method": "GREEDY",
      "stopping": {
        "max_new_tokens": 4
      },
      "response": {
          "input_text": true
        }
    }
  }' \
  localhost:8033 fmaas.GenerationService/GenerateStream