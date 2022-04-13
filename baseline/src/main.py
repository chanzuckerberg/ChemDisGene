import json
import sys
import os
import random
import click
import torch
import numpy as np
from module.setup import setup
from module.train import Trainer

os.environ['TRANSFORMERS_CACHE'] = os.environ['BIORE_OUTPUT_ROOT'] + \
    '/.cache/huggingface/'


class IntOrPercent(click.ParamType):
    name = "click_union"

    def convert(self, value, param, ctx):
        try:
            float_value = float(value)
            if 0 <= float_value <= 1:
                return float_value
            elif float_value == int(float_value):
                return int(float_value)
            else:
                self.fail(
                    f"expected float between [0,1] or int, got {float_value}",
                    param,
                    ctx,
                )
        except TypeError:
            self.fail(
                "expected string for int() or float() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or float", param, ctx)


@click.command(
    context_settings=dict(show_default=True),
)
@click.option(
    "--mode",
    type=click.Choice(
        ["train", "test"],
        case_sensitive=False,
    ),
    default="test"
)
@click.option(
    "--data_path",
    type=click.Path(),
    default="data/",
    help="directory or file",
)
@click.option(
    "--output_path",
    type=click.Path(),
    default="",
    help="directory to save model",
)
@click.option(
    "--load_path",
    type=click.Path(),
    default="",
    help="directory to load model",
)
@click.option(
    "--encoder_type",
    type=click.Choice(
         ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract","transformer_conv"],
        case_sensitive=False,
    ),
    default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    help="encoder architecture to use",
)
@click.option(
    "--model",
    type=click.Choice(
        ["dot", "biaffine"],
        case_sensitive=False,
    ),
    default="biaffine",
    help="score function",
)
@click.option(
    "--multi_label / --multi_class",
    default=True,
    help="multi_label allows multiple labels during inference; multi_class only allow one label"
)
@click.option(
    "--grad_accumulation_steps",
    type=int,
    default=16,
    help="tricks to have larger batch size with limited memory."
    + " The real batch size = train_batch_size * grad_accumulation_steps",
)
@click.option(
    "--max_text_length",
    type=int,
    default=512,
    help="max doc length of BPE tokens",
)
@click.option(
    "--dim",
    type=int,
    default=128,
    help="dimension of last layer feature before the score function "
    + "(e.g., dimension of biaffine layer, dimension of boxes)",
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-5,
    help="learning rate",
)
@click.option(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="weight decay",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=0.1,
    help="dropout rate",
)
@click.option(
    "--max_grad_norm",
    type=float,
    default=10.0,
    help="gradient norm clip (default 1.0)",
)
@click.option("--epochs", type=int, default=10, help="number of epochs to train")
@click.option(
    "--patience",
    type=int,
    default=5,
    help="patience parameter for early stopping",
)
@click.option(
    "--log_interval",
    type=IntOrPercent(),
    default=0.25,
    help="interval or percentage (as float in [0,1]) of examples to train before logging training metrics "
    "(default: 0, i.e. every batch)",
)
@click.option(
    "--warmup",
    type=float,
    default=-1.0,
    help="number of examples or percentage of training examples for warm up training "
    "(default: -1.0, no warmup, constant learning rate",
)
@click.option(
    "--cuda / --no_cuda",
    default=True,
    help="enable/disable CUDA (eg. no nVidia GPU)",
)
def main(**config):

    config["seed"] = random.randint(1, 10000)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    data, model, device, logger = setup(config)
    trainer = Trainer(data, model, logger, config, device)

    if config["load_path"] != "":
        best_metric_threshold = trainer.load_model()

    if config["mode"] == "train":
        trainer.train()
    else:
        best_metric_threshold = trainer.load_model()
        trainer.model.eval()
        macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf = trainer.test(
            "test_ctd", best_metric_threshold=best_metric_threshold)
        trainer.performance_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
                                    na_acc, not_na_perf, na_perf, per_rel_perf, 0, label="TEST CTD")

        macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf = trainer.test(
            "test_anno_ctd", best_metric_threshold=best_metric_threshold)
        trainer.performance_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
                                    na_acc, not_na_perf, na_perf, per_rel_perf, 0, label="TEST ANNOTATED CTD")

        macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf = trainer.test(
            "test_anno_all", best_metric_threshold=best_metric_threshold)
        trainer.performance_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
                                    na_acc, not_na_perf, na_perf, per_rel_perf, 0, label="TEST ANNOTATED ALL")

    logger.info("Program finished")


if __name__ == "__main__":
    main()
