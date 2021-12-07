# Vision Beyond Limits 

## Structure

```bash
├── agents 
│   ├── __init__.py    
│   ├── vanillaEnetAgent.py
│   ├── base.py
├── conf
│   ├── vaniallaEnet.json
│   ├── directory
├── data
│   ├── yet another directory
│   ├── directory
├── dataloader
│   ├── __init__.py
│   ├── vanillaEnetLoader.py
├── experiments
│   ├── yet another directory
│   ├── directory
├── losses
│   ├── __init__.py
│   ├── crossEntropy.py
├── models
│   ├── __init__.py
│   ├── enet.py
├── pretrained_weights
│   ├── yet another directory
│   ├── directory
├── utils
│   ├── __init__.py
│   ├── config.py
│   ├── dirs.py
│   ├── metrics.py
│   ├── scripts.py
│   ├── xview.py
│   ├── wandbUtils.py
├── __init__.py        
├── main.py            
├── README.md
```

## Installation
```bash
git clone https://github.com/mradul2/vbl.git
```

## Usage

```bash
python3 vbl/main.py [-h] [--mode MODE] [--wandb_id "WANDB_API_KEY"] vbl/config/file.json 
```