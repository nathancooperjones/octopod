import copy
import time

from fastprogress.fastprogress import format_time, master_bar, progress_bar
import numpy as np
from PIL import Image
import torch

from octopod.learner_utils import (DEFAULT_LOSSES_DICT,
                                   DEFAULT_METRIC_DICT)
from octopod.vision.config import cropped_transforms, full_img_transforms
from octopod.vision.helpers import center_crop_pil_image, get_number_from_string


class HierarchicalModel(object):
    """
    Class to encapsulate training and validation steps for a pipeline. Based off the fastai learner.

    Parameters
    ----------
    models: torch.nn.Module
        PyTorch `nn.ModuleDict` model to use with the Learner
    train_dataloader: MultiDatasetLoader
        dataloader for all of the training data
    val_dataloader: MultiDatasetLoader
        dataloader for all of the validation data
    task_dict: dict
        dictionary with all of the tasks as keys and the number of unique labels as the values
    loss_function: str
        Loss to use out of the options in
        `octopod.learner_utils.loss_metrics_utils_config.DEFAULT_LOSSES_DICT`
    metric_function: str
        Metric to use out of the options in
        `octopod.learner_utils.metrics_utils.DEFAULT_METRIC_DICT`

    """
    def __init__(self,
                 models,
                 train_dataloader,
                 val_dataloader,
                 task_dict,
                 loss_function='categorical_cross_entropy',
                 metric_function='multi_class_acc'):
        self.models = models
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.task_dict = task_dict
        self.tasks = [*task_dict]
        self.loss_function = DEFAULT_LOSSES_DICT[loss_function]
        self.metric_function = DEFAULT_METRIC_DICT[metric_function]

    def fit(
        self,
        num_epochs,
        scheduler,
        step_scheduler_on_batch,
        optimizer,
        device='cuda:0',
        best_model=False,
        depth=4,
    ):
        """
        Fit the PyTorch model

        Parameters
        ----------
        num_epochs: int
            number of epochs to train
        scheduler: torch.optim.lr_scheduler
            PyTorch learning rate scheduler
        step_scheduler_on_batch: bool
            flag of whether to step scheduler on batch (if True) or on epoch (if False)
        optimizer: torch.optim
            PyTorch optimzer
        device: str (defaults to 'cuda:0')
            device to run calculations on
        best_model: bool (defaults to `False`)
            flag to save best model from a single `fit` training run based on validation loss
            The default is `False`, which will keep the final model from the training run.
            `True` will keep the best model from the training run instead of the model
            from the final epoch of the training cycle.
        depth: int
            Maximum depth to optimize for this `fit` cycle (default 4)

        """
        self.models = self.models.to(device)

        current_best_loss = np.iinfo(np.intp).max

        pbar = master_bar(range(num_epochs))
        headers = ['train_loss', 'val_loss']
        for task in self.tasks:
            headers.append(f'{task}_train_loss')
            headers.append(f'{task}_val_loss')
            headers.append(f'{task}'+'_'+self.metric_function.__name__)
        headers.append('time')
        pbar.write(headers, table=True)

        for epoch in pbar:
            start_time = time.time()
            self.models.train()

            training_loss_dict = {task: 0.0 for task in self.tasks}

            overall_training_loss = 0.0
            number_of_samples = 0

            for step, batch in enumerate(progress_bar(self.train_dataloader, parent=pbar)):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)
                y = y.to(device)

                num_rows = self._get_num_rows(x)

                if num_rows == 1:
                    # skip batches of size 1
                    continue

                level_number = get_number_from_string(task_type)

                if (level_number + 1) > depth:
                    continue

                output = self._hacky_thing(task_type=task_type, x=x, level_number=level_number)

                current_loss = self.loss_function(output, y)

                scaled_loss = current_loss.item() * num_rows

                training_loss_dict[task_type] += scaled_loss

                overall_training_loss += scaled_loss
                number_of_samples += num_rows

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                if step_scheduler_on_batch:
                    scheduler.step()

            overall_training_loss = overall_training_loss/number_of_samples

            for task in self.tasks:
                training_loss_dict[task] = (
                    training_loss_dict[task]
                    / len(self.train_dataloader.loader_dict[task].dataset)
                )

            overall_val_loss, val_loss_dict, metrics_scores = self.validate(
                device,
                pbar,
                depth,
            )

            if not step_scheduler_on_batch:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(overall_val_loss)
                else:
                    scheduler.step()

            str_stats = []
            stats = [overall_training_loss, overall_val_loss]
            for stat in stats:
                str_stats.append(
                    'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                )

            for task in self.tasks:
                str_stats.append(f'{training_loss_dict[task]:.6f}')
                str_stats.append(f'{val_loss_dict[task]:.6f}')
                str_stats.append(
                    f"{metrics_scores[task][self.metric_function.__name__]:.6f}"
                )

            str_stats.append(format_time(time.time() - start_time))

            pbar.write(str_stats, table=True)

            if best_model and overall_val_loss < current_best_loss:
                current_best_loss = overall_val_loss
                best_model_wts = copy.deepcopy(self.models.state_dict())
                best_model_epoch = epoch

        if best_model:
            self.models.load_state_dict(best_model_wts)
            print(f'Epoch {best_model_epoch} best model saved with loss of {current_best_loss}')

    def validate(self, device='cuda:0', pbar=None, depth=4):
        """
        Evaluate the model on a validation set

        Parameters
        ----------
        loss_function: function
            function to calculate loss with in model
        device: str (defaults to 'cuda:0')
            device to run calculations on
        pbar: fast_progress progress bar (defaults to None)
            parent progress bar for all epochs

        Returns
        -------
        overall_val_loss: float
            overall validation loss for all tasks
        val_loss_dict: dict
            dictionary of validation losses for individual tasks
        metrics_scores: dict
            scores for individual tasks
        depth: int
            Maximum depth to optimize for this `fit` cycle (default 4)

        """
        preds_dict = {}

        val_loss_dict = {task: 0.0 for task in self.tasks}
        metrics_scores = (
            {task: {self.metric_function.__name__: 0.0} for task in self.tasks}
        )

        overall_val_loss = 0.0
        number_of_samples = 0

        self.models.eval()

        with torch.no_grad():
            for step, batch in enumerate(
                progress_bar(self.val_dataloader, parent=pbar, leave=(pbar is not None))
            ):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)

                y = y.to(device)

                num_rows = self._get_num_rows(x)

                level_number = get_number_from_string(task_type)

                if (level_number + 1) > depth:
                    continue

                output = self._hacky_thing(task_type=task_type, x=x, level_number=level_number)

                current_loss = self.loss_function(output, y)

                scaled_loss = current_loss.item() * num_rows

                val_loss_dict[task_type] += scaled_loss
                overall_val_loss += scaled_loss
                number_of_samples += num_rows

                y_pred = output.cpu().numpy()
                y_true = y.cpu().numpy()

                preds_dict = self._update_preds_dict(preds_dict, task_type, y_true, y_pred)

        overall_val_loss /= number_of_samples

        for task in self.tasks:
            val_loss_dict[task] = (
                val_loss_dict[task]
                / len(self.val_dataloader.loader_dict[task].dataset)
            )

        for task in metrics_scores.keys():
            try:
                y_true = preds_dict[task]['y_true']
                y_raw_pred = preds_dict[task]['y_pred']

                metric_score, _ = self.metric_function(y_true, y_raw_pred)
            except KeyError:
                metric_score = -1

            metrics_scores[task][self.metric_function.__name__] = metric_score

        return overall_val_loss, val_loss_dict, metrics_scores

    def _hacky_thing(self, task_type, x, level_number):
        # HACKY
        if level_number == 0:
            output = self.models[task_type](x)
        elif level_number == 1:
            initial_output = self.models['level_0_START'].get_embeddings(x)
            output = self.models[task_type](initial_output)
        elif level_number == 2:
            initial_output = self.models['level_0_START'].get_embeddings(x)
            next_output = self.models[
                'level_1_' + task_type.split('|')[0][8:]
            ].get_embeddings(initial_output)
            output = self.models[task_type](next_output)
        elif level_number == 3:
            initial_output = self.models['level_0_START'].get_embeddings(x)
            next_output_1 = self.models[
                'level_1_' + task_type.split('|')[0][8:]
            ].get_embeddings(initial_output)
            next_output_2 = self.models[
                'level_2_' + '|'.join(task_type.split('|')[:-1])[8:]
            ].get_embeddings(next_output_1)
            output = self.models[task_type](next_output_2)

        return output

    def predict(self, image_path, label_mapping_dict, start_level='level_0_START', device='cuda:0'):
        """
        For a single input image, predict down the model hierarchy.

        Parameters
        ----------
        x: torch.tensor, 2-d
            Input should be of shape `batch_size * start`

        Returns
        ----------
        output: torch.tensor, 2-d
            Output will be of shape `batch_size * task_size`

        """
        full_img = Image.open(image_path).convert('RGB')
        cropped_img = center_crop_pil_image(full_img)

        full_img = full_img_transforms['val'](full_img)
        cropped_img = cropped_transforms['val'](cropped_img)

        full_img = torch.autograd.Variable(full_img, requires_grad=False)
        cropped_img = torch.autograd.Variable(cropped_img, requires_grad=False)

        full_img = full_img.unsqueeze(0)
        cropped_img = cropped_img.unsqueeze(0)

        x = {'full_img': full_img, 'crop_img': cropped_img}
        x = self._return_input_on_device(x, device)

        current_layer_list = list()

        with torch.no_grad():
            current_level = 0
            model_input = x

            while True:
                if len(current_layer_list) > 0:
                    current_layer = f'level_{str(current_level)}_' + '|'.join(current_layer_list)
                else:
                    current_layer = start_level

                if label_mapping_dict.get(current_layer) is None:
                    break

                prediction_layer, embedding = self.models[current_layer].predict(model_input)
                prediction_idx = torch.argmax(prediction_layer)
                next_layer = label_mapping_dict[start_level][prediction_idx]

                current_layer_list.append(next_layer)
                current_level += 1
                model_input = embedding

        return current_layer_list

    def _return_input_on_device(self, x, device):
        """
        Send all model inputs to the appropriate device (GPU or CPU)
        when the inputs are in a dictionary format.

        Parameters
        ----------
        x: dict
            Output of a dataloader where the dataset generator groups multiple
            model inputs (such as multiple images) into a dictionary. Example
            `{'full_img':some_tensor,'crop_img':some_tensor}`

        """
        for k, v in x.items():
            x[k] = v.to(device)
        return x

    def _get_num_rows(self, x):
        return x[next(iter(x))].size(0)

    def _update_preds_dict(self, preds_dict, task_type, y_true, y_pred):
        """
        Updates prediction dictionary for a specific task with both true labels
        and predicted labels for a given batch
        """

        if task_type not in preds_dict:
            preds_dict[task_type] = {
                'y_true': y_true,
                'y_pred': y_pred
            }

        else:
            preds_dict[task_type]['y_true'] = (
                np.concatenate((preds_dict[task_type]['y_true'], y_true))
            )

            preds_dict[task_type]['y_pred'] = (
                np.concatenate((preds_dict[task_type]['y_pred'], y_pred))
            )
        return preds_dict
