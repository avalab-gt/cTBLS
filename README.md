# cTBLS_demo
Demo of the cTBLS system

![](Demo_v1.mov)

### 1. Download the model and data files:

```fine-tuned-encoder.pt``` from https://huggingface.co/avalab/cTBLS_encoder  
```fine-tuned-qa_retriever_distributed_epoch_13_Jan_10_cpu.pt``` from https://huggingface.co/avalab/cTBLS_knowledge_retriever  
```experimental_data.json``` from https://github.com/entitize/HybridDialogue  

### 2. Create a conda environment 

    conda env create --name envname --file=environment.yml

### 2. Generate an openAI API key and export it to .bashrc

### 3. To run the GUI:  

    python app.py  
   
  
  
