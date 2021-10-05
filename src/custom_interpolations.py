from omegaconf import OmegaConf


OmegaConf.register_new_resolver("decoder_out_size", lambda a,b,c: int(a) * int(b) * int(c))
OmegaConf.register_new_resolver("slash_to_dot", lambda s: s.replace("/", "."))
OmegaConf.register_new_resolver("filename", lambda s: s.split("/")[-1].lstrip('.dat'))
