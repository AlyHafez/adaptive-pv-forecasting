import random
import numpy as np # type: ignore[import]
import torch# type: ignore[import]
import os
import logging
import wandb # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
from lightning.pytorch import seed_everything # type: ignore[import]
logging.basicConfig(level=logging.INFO)
def wandb_login():
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY") #load the API key from Environment variables
    if wandb_api_key is None:
        msg = "WANDB_API_KEY not found in environment variables. Please set it in the .env file."
        logging.error(msg)
        raise ValueError(msg)
    else:
        logging.info("WANDB_API_KEY successfully loaded from environment variables.")
        wandb.login(key=wandb_api_key) #  type: ignore[attr-defined]

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
