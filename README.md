# Llama-Jarvis
![Lint Status](https://github.com/johnsutor/llama-jarvis/workflows/Lint/badge.svg)
![Tests Status](https://github.com/johnsutor/llama-jarvis/workflows/Test/badge.svg)
![contributions welcome](https://img.shields.io/badge/contributions-welcome-blue.svg?style=flat)

![alt text](./assets/llama.png)
Train a speech-to-speech model using your own language model. Currently based on the [Seamless Model](https://huggingface.co/collections/facebook/seamless-communication-6568d486ef451c6ba62c7724), but plan to support more models in the future.

This model is based on speech-to-speech models such as [Llama-Omni](https://github.com/ictnlp/LLaMA-Omni). However, it aims to take advantage of the joint speech-text embeddings of the Seamless Model.

This code is very much a work in progress. Any and all contributions are welcome!  

## Examples
**NOTE** For some of the below, you may have to first [log in to Huggingface](https://huggingface.co/docs/huggingface_hub/main/package_reference/authentication) to gain access to the gated models (especially Llama models).  

### Running Locally 
This code is not yet available via PyPi (I am hesitant to release it without thoroughly testing the code). Thus, to try it locally, please run
```shell 
git clone https://github.com/johnsutor/llama-jarvis
cd llama-jarvis 
pip install -e . 
```

### Phase One Loss
The example code will return the phase one loss (i.e., when training the first phase of Llama-Omni) 
```py 
from llama_jarvis.model import JarvisModel, JarvisConfig, JarvisProcessor

BASE_LLM = "meta-llama/Llama-3.2-1B"
SEAMLESS_MODEL = "facebook/hf-seamless-m4t-medium"
LANGUAGE = "eng"

jarvis_config = JarvisConfig(
    BASE_LLM,
    SEAMLESS_MODEL
)
jarvis_model = JarvisModel(jarvis_config)
jarvis_processor = JarvisProcessor(
    BASE_LLM,
    SEAMLESS_MODEL
)

inputs = processor(
    instruction=["You are a language model who should respond to my speech"],
    text=["What is two plus two?"],
    label=["Two plus two is four"],
    src_lang=LANGUAGE,
    return_tensors="pt",
    padding=True
)

outputs = model.forward(
    **inputs,
    tgt_lang=LANGUAGE
)

print(output.loss)
```

### Phase One Two
The example code will return the phase two loss (i.e., when training the second phase of Llama-Omni) 
```py 
from llama_jarvis.model import JarvisModel, JarvisConfig, JarvisProcessor

BASE_LLM = "meta-llama/Llama-3.2-1B"
SEAMLESS_MODEL = "facebook/hf-seamless-m4t-medium"
LANGUAGE = "eng"

jarvis_config = JarvisConfig(
    BASE_LLM,
    SEAMLESS_MODEL
)
jarvis_model = JarvisModel(jarvis_config)
jarvis_processor = JarvisProcessor(
    BASE_LLM,
    SEAMLESS_MODEL
)

inputs = processor(
    instruction=["You are a language model who should respond to my speech"],
    text=["What is two plus two?"],
    label=["Two plus two is four"],
    src_lang=LANGUAGE,
    return_tensors="pt",
    padding=True
)

outputs = model.forward(
    **inputs,
    tgt_lang=LANGUAGE,
    train_phase=2
)

print(output.loss)
```

## Roadmap
- [ ] Train a baseline model using Llama 3.2 1B and Seamless Medium
- [ ] Provide training example code 
- [ ] Fully document the code 
- [ ] Create an inference script for the model
- [ ] Write thorough tests for the code, and test with a multitude of open-source models 
- [ ] Release the code on PyPi 