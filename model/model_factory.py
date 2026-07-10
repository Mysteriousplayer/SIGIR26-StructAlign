class ModelFactory:
    # This repository is centered on the StructAlign pipeline.
    # Additional retrieval backbones are retained as reference implementations
    # and may require extra environment-specific setup for end-to-end use.
    @staticmethod
    def get_model(config):
        if config.arch == 'avg_pool':
            from model.AvgPool.AvgPool import AvgPool
            return AvgPool(config)
        elif config.arch == 'xpool':
            from model.XPool.clip_transformer import XPool
            return XPool(config)
        elif config.arch == 'moe_adapter':
            from model.MoE_Adapter.MoEAdapter import MoEAdapter
            return MoEAdapter(config)
        elif config.arch == 'clip_vip':
            from model.CLIP_ViP.ClipViP import ClipVip
            return ClipVip(config)
        elif config.arch == 'StructAlign':
            from model.StructAlignMoE.SA_model import StructAlign
            return StructAlign(config)
        elif config.arch == 'FrameFusionMoE':
            # FrameFusionMoE is retained as a reference branch in this repository.
            from model.FrameFusionMoE.FrameFusion import FrameFusion
            return FrameFusion(config)
        else:
            raise NotImplemented
