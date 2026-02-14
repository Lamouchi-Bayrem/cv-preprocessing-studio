class ProcessingConfig:
    def __init__(
        self,
        clip_limit=2.0,
        denoise_strength=10,
        block_size=11,
        c=2,
    ):
        self.clip_limit = clip_limit
        self.denoise_strength = denoise_strength
        self.block_size = block_size | 1
        self.c = c
