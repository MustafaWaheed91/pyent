import os
from math import ceil
import logging
from datetime import datetime
import json

import numpy as np
import pandas as pd
from torch.cuda import is_available, is_initialized
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


def train_txt_baseline(X_train_txt: pd.DataFrame, y_train: pd.Series,
                       X_test_txt: pd.DataFrame, y_test: pd.Series, X_val_txt: pd.DataFrame, y_val: pd.Series,
                       **kwargs) -> None:
    """train baseline sentence transformer model 
    """
    try:
        # prepare data splits for algorithm
        X_train_txt['target'] = np.where(y_train == "match", 1, 0)
        X_test_txt['target'] = np.where(y_test == "match", 1, 0)
        X_val_txt['target'] = np.where(y_val == "match", 1, 0)

        # # parameters abd configs for training
        # model_name = 'bert-base-uncased'
        # num_epochs = 1
        # train_batch_size = 64
        # margin = 0.5

        model_save_path = '../output/models/{}-bsz-{}-ep-{}-{}'.format(
            kwargs["model_name"],
            kwargs["train_batch_size"],
            kwargs["num_epochs"],
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(model_save_path, exist_ok=True)
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
        model = SentenceTransformer(kwargs["model_name"])

        # create train and test sample
        train_samples = []
        for row in X_train_txt.iterrows():
            if row[1]['target'] == 1:
                train_samples.append(
                    InputExample(
                        texts=[
                            row[1]['sentence_l'],
                            row[1]['sentence_r']
                        ],
                        label=int(row[1]['target'])
                    )
                )
                train_samples.append(
                    InputExample(
                        texts=[
                            row[1]['sentence_r'],
                            row[1]['sentence_l']
                        ],
                        label=int(row[1]['target'])
                    )
                )
            else:
                train_samples.append(
                    InputExample(
                        texts=[
                            row[1]['sentence_l'],
                            row[1]['sentence_r']
                        ],
                        label=int(row[1]['target'])
                    )
                )

        # create train and test sample
        dev_samples = []
        for row in X_val_txt.iterrows():
            if row[1]['target'] == 1:
                dev_samples.append(
                    InputExample(
                        texts=[
                            row[1]['sentence_l'],
                            row[1]['sentence_r']
                        ],
                        label=int(row[1]['target'])
                    )
                )
                dev_samples.append(
                    InputExample(
                        texts=[
                            row[1]['sentence_r'],
                            row[1]['sentence_l']
                        ],
                        label=int(row[1]['target'])
                    )
                )
            else:
                dev_samples.append(
                    InputExample(
                        texts=[
                            row[1]['sentence_l'],
                            row[1]['sentence_r']
                        ],
                        label=int(row[1]['target'])
                    )
                )

        # initialize data loader and loss definition
        # noinspection PyTypeChecker
        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=kwargs["train_batch_size"]
        )

        train_loss = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=distance_metric,
            margin=kwargs["margin"]
        )

        evaluators = []

        binary_acc_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
            dev_samples,
            show_progress_bar=True
        )

        evaluators.append(binary_acc_evaluator)

        logger.info("Evaluate model without training")
        seq_evaluator = evaluation.SequentialEvaluator(
            evaluators=evaluators,
            main_score_function=lambda scores: scores[-1]
        )
        seq_evaluator(
            model=model,
            epoch=0,
            steps=0,
            output_path=model_save_path
        )

        logger.info("Start Model Training")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=seq_evaluator,
            epochs=kwargs["num_epochs"],
            use_amp=True if is_initialized() and is_available() else False,
            warmup_steps=ceil(len(train_dataloader) * 0.1),
            output_path=model_save_path,
            show_progress_bar=True
        )

        logger.info("Evaluate model performance on test set")
        bi_encoder = SentenceTransformer(model_save_path)

        test_sentence_l = X_test_txt.sentence_l.tolist()
        test_sentence_r = X_test_txt.sentence_r.tolist()
        test_target = X_test_txt.target.tolist()

        test_eval = evaluation.BinaryClassificationEvaluator(
            sentences1=test_sentence_l,
            sentences2=test_sentence_r,
            labels=test_target,
            name=f"test_evaluator_{os.path.basename(model_save_path)}",
            batch_size=32,
            write_csv=True,
            show_progress_bar=True
        )

        test_pref_metrics = test_eval.compute_metrices(bi_encoder)
        logger.info(f"Test Performance Metrics:\n{json.dumps(test_pref_metrics['cossim'], indent=4)}")
    except Exception as e:
        logging.error(e)
        raise
