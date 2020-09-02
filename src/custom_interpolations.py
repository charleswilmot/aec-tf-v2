from omegaconf import OmegaConf


OmegaConf.register_resolver("decoder_out_size", lambda a,b,c: int(a) * int(b) * int(c))
