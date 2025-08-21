import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from torchvision.utils import make_grid
import json

class TensorBoardLogger:
    def __init__(self, experiment_name, hyperparams=None, log_dir=None, results_dir=None):
        from torch.utils.tensorboard import SummaryWriter
        
        self.experiment_name = experiment_name
        self.log_dir = log_dir if log_dir else 'runs'
        self.results_dir = results_dir if results_dir else 'Results'
        self.writer = SummaryWriter(os.path.join(self.log_dir, experiment_name))
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        print(f"Result files will be saved to: {self.results_dir}")
        
        self.hyperparams = hyperparams or {}
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'epoch_time': []
        }
        
        self.current_epoch = -1
        
    def log_metrics(self, metrics, step, prefix=''):
        prefix_clean = prefix.rstrip('/') if prefix else ''
        
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}{name}', value, step)
            
        if step > self.current_epoch or (step == self.current_epoch and prefix_clean == 'val'):
            if prefix_clean == 'train':
                self.current_epoch = step
                if 'epoch' in self.history:
                    self.history['epoch'].append(step + 1)
            
            for name, value in metrics.items():
                history_key = f"{prefix_clean}_{name}" if prefix_clean else name
                
                if history_key not in self.history:
                    self.history[history_key] = []
                    
                self.history[history_key].append(value)
            
    def log_learning_rate(self, lr, step):
        self.writer.add_scalar('train/learning_rate', lr, step)
        
    def log_epoch_time(self, time_str, epoch):
        if epoch >= len(self.history['epoch_time']):
            self.history['epoch_time'].append(time_str)
    
    def get_history(self):
        return self.history
            
    def log_images(self, images, targets, predictions, step, prefix=''):
        with torch.no_grad():
            colored_targets = self._colorize_masks(targets)
            colored_preds = self._colorize_masks(predictions)
            
            colored_targets = torch.from_numpy(colored_targets)
            colored_preds = torch.from_numpy(colored_preds)
            
            img_grid = make_grid(images, pad_value=1)
            target_grid = make_grid(colored_targets, pad_value=1)
            pred_grid = make_grid(colored_preds, pad_value=1)
            
            self.writer.add_image(f'{prefix}input', img_grid, step)
            self.writer.add_image(f'{prefix}target', target_grid, step)
            self.writer.add_image(f'{prefix}prediction', pred_grid, step)
        
    def _colorize_masks(self, masks):
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        
        colors = np.array([
            [0, 0, 0],      
            [255, 0, 0],    
            [0, 255, 0],    
            [0, 0, 255],    
        ])
        
        if masks.ndim == 4:
            if masks.shape[1] == 1:
                masks = np.squeeze(masks, axis=1)
        
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
            
        batch_size, height, width = masks.shape
        
        colored_masks = np.zeros((batch_size, 3, height, width), dtype=np.uint8)
        
        for b in range(batch_size):
            for class_idx in np.unique(masks[b]):
                if class_idx < len(colors):
                    mask = (masks[b] == class_idx)
                    
                    for c in range(3):
                        colored_masks[b, c, mask] = colors[int(class_idx)][c]
        
        return colored_masks
    
    def _generate_colors(self, num_classes):
        import matplotlib.cm as cm
        
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = cm.hsv(hue)[:3]
            colors.append([int(c * 255) for c in rgb])
        return colors
    
    def log_confusion_matrix(self, confusion_matrix, step, class_names=None, prefix=''):
        cm_figure = self._plot_confusion_matrix(confusion_matrix, class_names)
        
        tag = f"{prefix}confusion_matrix" if prefix else "confusion_matrix"
        self.log_plot(cm_figure, step, tag=tag)
        
        plt.close(cm_figure)
        
        return cm_figure
    
    def _plot_confusion_matrix(self, cm, class_names=None):
        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]
            
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if isinstance(cm[i, j], (int, np.integer)) or cm[i, j] == int(cm[i, j]):
                    value_str = format(int(cm[i, j]), 'd')
                else:
                    value_str = format(cm[i, j], '.2f')
                    
                ax.text(j, i, value_str,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig
        
    def log_plot(self, figure, step, tag='custom_plot'):
        self.writer.add_figure(tag, figure, step)
        
    def log_weights_gradients(self, model, step):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param.data, step)
                
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)

    def export_history_to_json(self, total_time=None, best_metric=None, save_dir=None):
        save_directory = save_dir if save_dir is not None else self.results_dir
        os.makedirs(save_directory, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        max_epoch = max(self.history['epoch']) if self.history['epoch'] else 0
        filepath = os.path.join(save_directory, f"{timestamp}_{self.experiment_name}_e{max_epoch}.json")
        
        data = {
            "experiment_name": self.experiment_name,
            "hyperparameters": self.hyperparams,
            "metrics": {}
        }
        
        for metric, values in self.history.items():
            if isinstance(values, (list, tuple)):
                serializable_values = []
                for val in values:
                    if isinstance(val, (int, float, bool, str, type(None))):
                        serializable_values.append(val)
                    elif hasattr(val, 'item'):
                        serializable_values.append(val.item())
                    else:
                        serializable_values.append(float(val))
                data["metrics"][metric] = serializable_values
            else:
                data["metrics"][metric] = values
        
        summary_data = {}
        if total_time is not None:
            summary_data["total_training_time"] = total_time
        if best_metric is not None:
            metric_name, metric_value = best_metric
            summary_data["best_performance"] = {
                "metric": metric_name,
                "value": metric_value
            }
        
        if summary_data:
            data["summary"] = summary_data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"Training history saved to: {filepath}")
        return filepath

    def plot_learning_curves(self, save_dir=None):
        save_directory = save_dir if save_dir is not None else self.results_dir
        os.makedirs(save_directory, exist_ok=True)
        
        if 'epoch' not in self.history or not self.history['epoch']:
            print("History is incomplete, cannot plot learning curves (missing epoch data)")
            return None
        
        metric_groups = {
            'loss': [],
            'accuracy': [],
            'iou': [],
            'f1': [],
            'pr': [],
            'other': []
        }
        
        metric_patterns = {
            'loss': lambda k: 'loss' in k.lower(),
            'accuracy': lambda k: 'acc' in k.lower() or 'accuracy' in k.lower(),
            'iou': lambda k: 'iou' in k.lower() or 'jaccard' in k.lower(),
            'f1': lambda k: 'f1' in k.lower() or 'fscore' in k.lower(),
            'pr': lambda k: 'precision' in k.lower() or 'recall' in k.lower()
        }
        
        prefix_colors = {
            'train': 'b',
            'val': 'r',
            'm': 'g',
            '': 'm'
        }
        
        for key in self.history.keys():
            if key == 'epoch' or key == 'epoch_time' or not isinstance(self.history[key], list):
                continue
                
            if not self.history[key]:
                continue
                
            assigned = False
            for group, pattern in metric_patterns.items():
                if pattern(key):
                    metric_groups[group].append(key)
                    assigned = True
                    break
                    
            if not assigned:
                metric_groups['other'].append(key)
        
        active_groups = [group for group, metrics in metric_groups.items() if metrics]
        num_groups = len(active_groups)
        
        if num_groups == 0:
            print("No plottable metrics found in history.")
            return None
        
        if num_groups <= 3:
            nrows, ncols = num_groups, 1
        else:
            nrows = (num_groups + 1) // 2
            ncols = 2
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
        
        if num_groups == 1:
            axs = np.array([axs])
        
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        
        for i, group in enumerate(active_groups):
            ax = axs[i]
            
            group_titles = {
                'loss': 'Loss Curves',
                'accuracy': 'Accuracy Curves',
                'iou': 'IoU Curves',
                'f1': 'F1 Score Curves',
                'pr': 'Precision & Recall Curves',
                'other': 'Other Metrics'
            }
            
            ax.set_title(group_titles.get(group, f"{group.title()} Metrics"))
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            
            for metric in sorted(metric_groups[group]):
                prefix = ''
                style = '-'
                
                for known_prefix in ['train_', 'val_', 'm']:
                    if metric.startswith(known_prefix):
                        prefix = known_prefix.rstrip('_')
                        break
                
                if 'recall' in metric:
                    style = '--'
                elif 'specificity' in metric:
                    style = ':'
                
                color = prefix_colors.get(prefix, prefix_colors[''])
                label = metric.replace('_', ' ').title()
                
                ax.plot(self.history['epoch'], self.history[metric], 
                        f"{color}{style}", label=label)
            
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        if nrows * ncols > num_groups:
            for i in range(num_groups, nrows * ncols):
                if i < len(axs):
                    fig.delaxes(axs[i])
        
        plt.tight_layout()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        max_epoch = max(self.history['epoch']) if self.history['epoch'] else 0
        save_path = os.path.join(save_directory, f"{timestamp}_{self.experiment_name}_e{max_epoch}_curves.png")
        
        plt.savefig(save_path)
        print(f"Learning curves saved to: {save_path}")
        
        last_epoch = max(self.history['epoch']) if self.history['epoch'] else 0
        self.log_plot(fig, last_epoch, 'learning_curves')
        
        return fig
        
    def close(self):
        self.writer.close()
        
    def __del__(self):
        self.close()


class Evaluator:
    def __init__(self, num_classes=2, device='cuda'):
        self.device = device
        if num_classes == 1 or num_classes == 2:
            self.num_class = 2
        else:
            self.num_class = num_classes
        
        from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
        from torchmetrics.classification import MulticlassJaccardIndex
        from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
        from torchmetrics.classification import CohenKappa, ConfusionMatrix
        from skimage import measure
        from scipy.spatial import distance
        
        self.measure = measure
        self.distance = distance
        
        self.over_seg = None
        self.under_seg = None
        self.total_seg = None
        
        self.metrics = torch.nn.ModuleDict({
            'overall_accuracy': MulticlassAccuracy(num_classes=self.num_class, average='micro').to(device),
            'accuracy': MulticlassAccuracy(num_classes=self.num_class, average='none').to(device),
            'f1_score': MulticlassF1Score(num_classes=self.num_class, average='none').to(device), 
            'iou': MulticlassJaccardIndex(num_classes=self.num_class, average='none').to(device),
            'precision': MulticlassPrecision(num_classes=self.num_class, average='none').to(device),
            'recall': MulticlassRecall(num_classes=self.num_class, average='none').to(device),
            'kappa': CohenKappa(task="multiclass", num_classes=self.num_class).to(device),      
        })
        
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_class).to(device)
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
        self.conf_matrix.reset()
        
        self.over_seg = None
        self.under_seg = None
        self.total_seg = None
    
    def add_batch(self, targets, preds):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)
        elif targets.device != self.device:
            targets = targets.to(self.device)
            
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=self.device)
        elif preds.device != self.device:
            preds = preds.to(self.device)
        
        if len(targets.shape) == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
            
        if len(preds.shape) == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        
        targets = targets.long()
        preds = preds.long()
        
        for name, metric in self.metrics.items():
            metric.update(preds, targets)
        
        self.conf_matrix.update(preds, targets)
    
    def compute(self):
        results = {}
        
        for name, metric in self.metrics.items():
            values = metric.compute()
            results[f'{name}_bg'] = values[0].item()
            results[f'{name}_fg'] = values[1].item()
            
        return results
            
    def Overall_Accuracy(self):
        return self.metrics['overall_accuracy'].compute().item()
        
    def Intersection_over_Union(self, class_index=None):
        values = self.metrics['iou'].compute()
        
        if class_index is None:
            return values[1].item()
        else:
            return values[class_index].item()
        
    def F1Score(self, class_index=None):
        values = self.metrics['f1_score'].compute()
        
        if class_index is None:
            return values[1].item()
        else:
            return values[class_index].item()
        
    def Precision(self, class_index=None):
        values = self.metrics['precision'].compute()
        
        if class_index is None:
            return values[1].item()
        else:
            return values[class_index].item()
    
    def Recall(self, class_index=None):
        values = self.metrics['recall'].compute()
        
        if class_index is None:
            return values[1].item()
        else:
            return values[class_index].item()
        
    def Confusion_Matrix(self):
        return self.conf_matrix.compute()
        
    def Kappa(self):
        return self.metrics['kappa'].compute().item()
        
    def get_classwise_metrics(self):
        results = {}
        class_names = ['background', 'cropland']
        
        for name, metric in self.metrics.items():
            values = metric.compute()
            
            if values.ndim == 0:
                results[name] = values.item()
            else:
                if len(values) >= len(class_names):
                    for i, class_name in enumerate(class_names):
                        results[f'{name}_{class_name}'] = values[i].item()
                else:
                    print(f"Warning: {name} metric returned fewer values than number of classes ({len(values)} < {len(class_names)})")
                    for i in range(min(len(values), len(class_names))):
                        results[f'{name}_{class_names[i]}'] = values[i].item()
                        
        return results
    
    def over_under_classification(self, output, label):
        assert len(output.shape) == 2, "Input should be a single 2D mask"
        assert len(label.shape) == 2, "Label should be a single 2D mask"
        
        global_area_output = np.sum(output)
        
        if global_area_output == 0 or np.sum(label) == 0:
            return None, None, None

        labeled_output = self.measure.label(output, background=0, connectivity=1)
        props_output = self.measure.regionprops(labeled_output)
        labeled_label = self.measure.label(label, background=0, connectivity=1)
        props_label = self.measure.regionprops(labeled_label)
        
        if len(props_output) == 0 or len(props_label) == 0:
            return None, None, None

        results_over = []
        results_under = []
        results_total = []

        for prop_output in props_output:
            min_distance = float('inf')
            closest_prop_label = None

            for prop_label in props_label:
                dist = self.distance.euclidean(prop_output.centroid, prop_label.centroid)
                if dist < min_distance:
                    min_distance = dist
                    closest_prop_label = prop_label

            area_output = prop_output.area
            area_label = closest_prop_label.area

            region_output = np.where(labeled_output == prop_output.label, 1, 0)
            region_label = np.where(labeled_label == closest_prop_label.label, 1, 0)
            area_intersection = np.sum(region_output * region_label)

            over_seg = 1 - (area_intersection / area_label)
            under_seg = 1 - (area_intersection / area_output)
            total_seg = np.sqrt((np.square(over_seg) + np.square(under_seg)) / 2)

            global_over_seg = over_seg * area_output / global_area_output
            global_under_seg = under_seg * area_output / global_area_output
            global_total_seg = total_seg * area_output / global_area_output

            results_over.append(global_over_seg)
            results_under.append(global_under_seg)
            results_total.append(global_total_seg)

        result_over = np.sum(results_over)
        result_under = np.sum(results_under)
        result_total = np.sum(results_total)

        return result_over, result_under, result_total

    def calculate_over_under(self, output, label):
        assert output.shape == label.shape, "Prediction and label shapes must be consistent"
        
        if len(output.shape) == 2:
            over, under, total = self.over_under_classification(output, label)
        elif len(output.shape) == 3:
            batch_size = output.shape[0]
            results_over = []
            results_under = []
            results_total = []
            
            for b in range(batch_size):
                output_batch = output[b]
                label_batch = label[b]
                result = self.over_under_classification(output_batch, label_batch)
                
                if result[0] is not None:
                    result_over, result_under, result_total = result
                    results_over.append(result_over)
                    results_under.append(result_under)
                    results_total.append(result_total)
            
            if results_over:
                over = np.mean(results_over)
                under = np.mean(results_under)
                total = np.mean(results_total)
            else:
                over = under = total = None
        else:
            raise ValueError(f'Unsupported input shape: {output.shape}')
        
        self.over_seg = over
        self.under_seg = under
        self.total_seg = total
        
        return over, under, total
