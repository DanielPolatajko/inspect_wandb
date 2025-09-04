def format_wandb_id_string(id: str) -> str:
    return id.replace("/", "__").replace("-", "_").replace(".", "__").replace(":", "__").replace("@", "__")