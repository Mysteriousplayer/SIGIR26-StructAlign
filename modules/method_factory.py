class MethodFactory:
    # This repository is centered on the StructAlign training and evaluation path.
    # Additional branches are retained as references and are not the primary
    # maintenance target.
    @staticmethod
    def get_trainer(config):
        if config.arch == 'StructAlign':
            from trainer.StructAlign import Trainer as StructAlignTrainer
            return StructAlignTrainer
        elif config.arch == 'FrameFusionMoE':
            # Reference branch: reuses the StructAlign trainer flow.
            from trainer.StructAlign import Trainer as StructAlignTrainer
            return StructAlignTrainer
        # Add additional methods here as elif branches.
        # elif config.arch == 'other_method':
        #     from trainer.trainer_acc_other import Trainer as OtherTrainer
        #     return OtherTrainer
        else:
            raise NotImplementedError(f"Trainer for '{config.arch}' is not implemented.")

    @staticmethod
    def get_evaluator(config):
        if config.arch == 'StructAlign':
            from trainer.StructAlign import Evaluator as StructAlignEvaluator
            return StructAlignEvaluator
        elif config.arch == 'FrameFusionMoE':
            # Reference branch: reuses the StructAlign evaluator flow.
            from trainer.StructAlign import Evaluator as StructAlignEvaluator
            return StructAlignEvaluator
        # Add additional methods here as elif branches.
        # elif config.arch == 'other_method':
        #     from trainer.trainer_acc_other import Evaluator as OtherEvaluator
        #     return OtherEvaluator
        else:
            raise NotImplementedError(f"Evaluator for '{config.arch}' is not implemented.")
