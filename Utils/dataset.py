from torch.utils.data import Dataset
import os
import numpy as np
import torch
import rasterio
import multiprocessing
from rasterio.errors import NotGeoreferencedWarning
import warnings
from tqdm import tqdm
import torchvision.transforms as transforms
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.utils.data as data
import pandas as pd
import tifffile as tif

os.environ['RASTERIO_NUM_THREADS'] = str(multiprocessing.cpu_count())
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, mode='train', crop_size=512, 
                 transform=None, mean_list=None, std_list=None, model=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.crop_size = crop_size
        self.custom_transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.img'))])
        
        self.mean_list = mean_list if mean_list is not None else [0.485, 0.456, 0.406]
        self.std_list = std_list if std_list is not None else [0.229, 0.224, 0.225]
        
        self.is_swin_model = model.lower() == 'swinunet'
        
        if label_dir:
            self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith(('.tif', '.tiff', '.img'))])
            
            if len(self.image_filenames) != len(self.label_filenames):
                raise ValueError(f"Number of images ({len(self.image_filenames)}) does not match number of labels ({len(self.label_filenames)})")
                
            self.filter_files()
        else:
            self.label_filenames = None

    def __len__(self):
        return len(self.image_filenames)
    
    def filter_files(self):
        image_basenames = [os.path.splitext(f)[0] for f in self.image_filenames]
        label_basenames = [os.path.splitext(f)[0] for f in self.label_filenames]
        common_basenames = set(image_basenames).intersection(set(label_basenames))
        
        image_dict_full = {os.path.splitext(f)[0]: f for f in self.image_filenames}
        label_dict_full = {os.path.splitext(f)[0]: f for f in self.label_filenames}
        
        if len(common_basenames) < min(len(image_basenames) * 0.5, len(label_basenames) * 0.5):
            def extract_id(filename):
                base = os.path.splitext(filename)[0]
                if '_' in base:
                    return base.split('_')[-1]
                return base
            
            image_ids = {extract_id(f): f for f in self.image_filenames}
            label_ids = {extract_id(f): f for f in self.label_filenames}
            common_ids = set(image_ids.keys()).intersection(set(label_ids.keys()))
            
            if len(common_ids) > len(common_basenames):
                print(f"Using underscore number matching, found {len(common_ids)} matching files")
                filtered_image_files = [image_ids[id_] for id_ in common_ids]
                filtered_label_files = [label_ids[id_] for id_ in common_ids]
                
                sorted_pairs = sorted(zip(filtered_image_files, filtered_label_files), 
                                      key=lambda x: extract_id(x[0]))
                filtered_image_files = [pair[0] for pair in sorted_pairs]
                filtered_label_files = [pair[1] for pair in sorted_pairs]
            else:
                print(f"Using complete filename matching, found {len(common_basenames)} matching files")
                filtered_image_files = [image_dict_full[base] for base in common_basenames]
                filtered_label_files = [label_dict_full[base] for base in common_basenames]
        else:
            filtered_image_files = [image_dict_full[base] for base in common_basenames]
            filtered_label_files = [label_dict_full[base] for base in common_basenames]
        
        if len(filtered_image_files) < len(image_basenames):
            print(f"Warning: {len(image_basenames) - len(filtered_image_files)} images excluded due to missing matching labels")
        
        if len(filtered_label_files) < len(label_basenames):
            print(f"Warning: {len(label_basenames) - len(filtered_label_files)} labels excluded due to missing matching images")
        
        self.image_filenames = filtered_image_files
        self.label_filenames = filtered_label_files
        
        print(f"After filtering, {len(filtered_image_files)} valid image-label pairs remain")
              
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        with rasterio.open(image_path) as image_ds:
            image = image_ds.read()
            dtype = image_ds.dtypes[0]
        
        if self.label_dir:
            label_path = os.path.join(self.label_dir, self.label_filenames[idx])
            with rasterio.open(label_path) as label_ds:
                label = label_ds.read(1).astype(np.int8)
                
            if label is None:
                raise FileNotFoundError(f"Label file not found: {label_path}")
        else:
            label = None
        
        self.max_value_tofloat = self.get_max_value(dtype)
        
        if label is not None:
            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}
        
        if self.custom_transform:
            sample = self.custom_transform(sample)
        else:
            if self.mode == 'train':
                sample = self.transform_train(sample)
            elif self.mode == 'val':
                sample = self.transform_val(sample)
            elif self.mode == 'test':
                sample = self.transform_test(sample)
        
        if self.mode == 'test':
            file_name = self.image_filenames[idx].split('.')[0]
            return sample['image'], sample['label'], file_name
        else:
            return sample['image'], sample['label']
    
    def transform_train(self, sample):
        image_np = sample['image']
        label_np = sample['label']
        
        if image_np.shape[0] <= 4:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        if self.is_swin_model:
            aug_pipeline = A.Compose([
                A.VerticalFlip(p=0.5), 
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(
                    blur_limit=(0),
                    sigma_limit=(0.5, 3.0),
                    p=0.5
                ),
                A.RandomResizedCrop(
                    height=image_np.shape[0], width=image_np.shape[1],
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=10,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.CLAHE(
                    clip_limit=(1, 4),
                    tile_grid_size=(8, 8),
                    p=0.5
                ),
                A.CenterCrop(height=224, width=224, p=1.0),
                A.ToFloat(max_value=self.max_value_tofloat, p=1.0),
                A.Normalize(
                    mean=self.mean_list,
                    std=self.std_list,
                    max_pixel_value=1.0,
                    p=1.0
                ),
                ToTensorV2(),
            ])
        else:
            aug_pipeline = A.Compose([
                A.VerticalFlip(p=0.5), 
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(
                    blur_limit=(0),
                    sigma_limit=(0.5, 3.0),
                    p=0.5
                ),
                A.RandomResizedCrop(
                    height=image_np.shape[0], width=image_np.shape[1],
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=10,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.CLAHE(
                    clip_limit=(1, 4),
                    tile_grid_size=(8, 8),
                    p=0.5
                ),
                A.ToFloat(max_value=self.max_value_tofloat, p=1.0),
                A.Normalize(
                    mean=self.mean_list,
                    std=self.std_list,
                    max_pixel_value=1.0,
                    p=1.0
                ),
                ToTensorV2(),
            ])
        
        augmented = aug_pipeline(image=image_np, mask=label_np)
        image = augmented['image']
        if isinstance(augmented['mask'], torch.Tensor):
            label = augmented['mask'].long().unsqueeze(0)
        else:
            label = torch.from_numpy(augmented['mask']).long().unsqueeze(0)

        return {'image': image, 'label': label}
    
    def transform_val(self, sample):
        image_np = sample['image']
        label_np = sample['label']
        
        if image_np.shape[0] <= 4:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        if self.is_swin_model:
            val_pipeline = A.Compose([
                A.CenterCrop(height=224, width=224, p=1.0),
                A.ToFloat(max_value=self.max_value_tofloat, p=1.0),
                A.Normalize(
                    mean=self.mean_list,
                    std=self.std_list,
                    max_pixel_value=1.0,
                    p=1.0
                ),
                ToTensorV2(),
            ])
        else:
            val_pipeline = A.Compose([
                A.ToFloat(max_value=self.max_value_tofloat, p=1.0),
                A.Normalize(
                    mean=self.mean_list,
                    std=self.std_list,
                    max_pixel_value=1.0,
                    p=1.0
                ),
                ToTensorV2(),
            ])
        
        augmented = val_pipeline(image=image_np, mask=label_np)
        image = augmented['image']
        if isinstance(augmented['mask'], torch.Tensor):
            label = augmented['mask'].long().unsqueeze(0)
        else:
            label = torch.from_numpy(augmented['mask']).long().unsqueeze(0)

        return {'image': image, 'label': label}
    
    def transform_test(self, sample):
        return self.transform_val(sample)
    
    def get_max_value(self, dtype_str):
        if 'uint8' in dtype_str:
            return np.iinfo(np.uint8).max
        elif 'uint16' in dtype_str:
            return np.iinfo(np.uint16).max
        elif 'int16' in dtype_str:
            return np.iinfo(np.int16).max 
        elif 'uint32' in dtype_str:
            return np.iinfo(np.uint32).max
        elif 'int32' in dtype_str:
            return np.iinfo(np.int32).max
        elif 'float' in dtype_str:
            return 1.0
        else:
            print(f"Warning: Unknown data type {dtype_str}, assuming data is already normalized")
            return 1.0
    
    @staticmethod
    def transform_infer_array(image_array, mean_list=None, std_list=None):
        if image_array.dtype == np.uint8:
            max_value = 255.0
        elif image_array.dtype == np.uint16:
            max_value = 65535.0
        elif image_array.dtype == np.int16:
            max_value = 32767.0
        elif image_array.dtype == np.uint32:
            max_value = 4294967295.0
        elif image_array.dtype == np.int32:
            max_value = 2147483647.0
        else:
            max_value = 1.0
        
        if len(image_array.shape) == 3 and image_array.shape[0] <= 4:
            image_array = np.transpose(image_array, (1, 2, 0))
        
        infer_pipeline = A.Compose([
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=mean_list,
                std=std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2(),
        ])
        
        augmented = infer_pipeline(image=image_array)
        image_tensor = augmented['image']
        
        return image_tensor
    
class DatasetAnalyzer:
    @staticmethod
    def get_dataset_properties(image_folder, label_folder):
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
        if not image_files:
            raise FileNotFoundError(f"No image files found in folder: {image_folder}")

        first_image_path = os.path.join(image_folder, image_files[0])
        with rasterio.open(first_image_path) as image_ds:
            if image_ds is None:
                raise FileNotFoundError(f"Image not found at path: {first_image_path}")

            bands = image_ds.count
            input_size = (image_ds.height, image_ds.width)

        unique_classes = set()
        label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.tif') or f.endswith('.tiff')])

        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            with rasterio.open(label_path) as dataset:
                if dataset is None:
                    raise FileNotFoundError(f"Label image not found at path: {label_path}")
                label = dataset.read(1)
                unique_classes.update(np.unique(label))

        if unique_classes == {0, 1}:
            classNum = 1
        else:
            classNum = len(unique_classes)

        return input_size, bands, classNum
    
    @staticmethod
    def calculate_means_stds_folder(image_folder, chunk_size=32):
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
        if not image_files:
            raise FileNotFoundError(f"No images found in folder: {image_folder}")

        with rasterio.open(os.path.join(image_folder, image_files[0])) as first_image:
            num_bands = first_image.count

        sum_values = np.zeros(num_bands, dtype=np.float32)
        sum_squares = np.zeros(num_bands, dtype=np.float32)
        pixel_count = 0

        for image_file in tqdm(image_files, desc='Calculate MeanStd: '):
            with rasterio.open(os.path.join(image_folder, image_file)) as raster:
                width = raster.width
                height = raster.height

                for i in range(0, height, chunk_size):
                    y_size = min(chunk_size, height - i)
                    
                    window = rasterio.windows.Window(0, i, width, y_size)
                    chunk = raster.read(window=window)
                    
                    if chunk.ndim == 2:
                        chunk = chunk[np.newaxis, ...]

                    dtype_str = chunk.dtype.name
                    if 'uint8' in dtype_str:
                        max_value = 255
                    elif 'uint16' in dtype_str:
                        max_value = 65535
                    elif 'int16' in dtype_str:
                        max_value = 32767
                    elif 'uint32' in dtype_str:
                        max_value = 4294967295
                    elif 'int32' in dtype_str:
                        max_value = 2147483647
                    elif 'float' in dtype_str:
                        max_value = 1.0
                    else:
                        max_value = 1.0
                        print(f"Warning: Unknown data type {dtype_str}, assuming normalized data")

                    chunk = chunk.astype(np.float32) / max_value

                    sum_values += np.sum(chunk, axis=(1, 2))
                    sum_squares += np.sum(np.square(chunk, dtype=np.float32), axis=(1, 2))
                    pixel_count += chunk.shape[1] * chunk.shape[2]

        means = sum_values / pixel_count
        stds = np.sqrt(sum_squares/pixel_count - np.square(means))

        return means.tolist(), stds.tolist()
    
class CAORSEDataset(data.Dataset):
    def __init__(self, image_dir, pseudodir, aug=False, mean_list=None, std_list=None):
        self.image_dir = image_dir
        self.pseudodir = pseudodir
        self.aug = aug
        
        self.mean_list = [0.485, 0.456, 0.406]
        self.std_list = [0.229, 0.224, 0.225]
        
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.jpg', '.png'))])
        self.label_filenames = sorted([f for f in os.listdir(pseudodir) if f.endswith(('.png', '.tif', '.tiff'))])
        
        self.filter_files()
        
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomGridShuffle(grid=(2, 2), p=0.5),
            A.Rotate(p=0.5),
        ])

    def filter_files(self):
        image_basenames = [os.path.splitext(f)[0] for f in self.image_filenames]
        label_basenames = [os.path.splitext(f)[0] for f in self.label_filenames]
        common_basenames = set(image_basenames).intersection(set(label_basenames))
        
        image_dict_full = {os.path.splitext(f)[0]: f for f in self.image_filenames}
        label_dict_full = {os.path.splitext(f)[0]: f for f in self.label_filenames}
        
        if len(common_basenames) < min(len(image_basenames) * 0.5, len(label_basenames) * 0.5):
            def extract_id(filename):
                base = os.path.splitext(filename)[0]
                if '_' in base:
                    return base.split('_')[-1]
                return base
            
            image_ids = {extract_id(f): f for f in self.image_filenames}
            label_ids = {extract_id(f): f for f in self.label_filenames}
            common_ids = set(image_ids.keys()).intersection(set(label_ids.keys()))
            
            if len(common_ids) > len(common_basenames):
                print(f"Using underscore number matching, found {len(common_ids)} matching files")
                filtered_image_files = [image_ids[id_] for id_ in common_ids]
                filtered_label_files = [label_ids[id_] for id_ in common_ids]
                
                sorted_pairs = sorted(zip(filtered_image_files, filtered_label_files), 
                                      key=lambda x: extract_id(x[0]))
                filtered_image_files = [pair[0] for pair in sorted_pairs]
                filtered_label_files = [pair[1] for pair in sorted_pairs]
            else:
                print(f"Using complete filename matching, found {len(common_basenames)} matching files")
                filtered_image_files = [image_dict_full[base] for base in common_basenames]
                filtered_label_files = [label_dict_full[base] for base in common_basenames]
        else:
            filtered_image_files = [image_dict_full[base] for base in common_basenames]
            filtered_label_files = [label_dict_full[base] for base in common_basenames]
        
        if len(filtered_image_files) < len(image_basenames):
            print(f"Warning: {len(image_basenames) - len(filtered_image_files)} images excluded due to missing matching labels")
        
        if len(filtered_label_files) < len(label_basenames):
            print(f"Warning: {len(label_basenames) - len(filtered_label_files)} labels excluded due to missing matching images")
        
        self.image_filenames = filtered_image_files
        self.label_filenames = filtered_label_files
        
        print(f"After filtering, {len(filtered_image_files)} valid image-label pairs remain")

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.pseudodir, self.label_filenames[index])
        
        img = tif.imread(img_path)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        if mask.ndim > 2:
            mask = mask[:, :, 0]
        
        if self.aug:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        ori_img = img[:, :, :3].copy()
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8')
        
        croppings = np.ones_like(mask, dtype="uint8")

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0

        mean_tensor = torch.tensor(self.mean_list, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(self.std_list, dtype=torch.float32).view(-1, 1, 1)
        img = (img - mean_tensor) / std_tensor
        mask = torch.from_numpy(mask)
        valid_mask = mask < 255
        mask = mask * valid_mask
        
        filename = os.path.basename(self.image_filenames[index])
        return img, mask, ori_img, croppings, filename

    def __len__(self):
        return len(self.image_filenames)
    
class CAOISPRSDataset(Dataset):
    def __init__(self, image_dir, label_dir, update_label_dir=None, mode='train', mean_list=None, std_list=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.update_label_dir = update_label_dir
        self.mode = mode
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.img'))])
        
        self.mean_list = mean_list if mean_list is not None else [0.485, 0.456, 0.406]
        self.std_list = std_list if std_list is not None else [0.229, 0.224, 0.225]
        
        if label_dir:
            self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith(('.tif', '.tiff', '.img'))])
            
            if len(self.image_filenames) != len(self.label_filenames):
                print(f"Warning: Number of images ({len(self.image_filenames)}) does not match number of labels ({len(self.label_filenames)})")
                
            self.filter_files()
        else:
            self.label_filenames = None
        
        if update_label_dir and os.path.exists(update_label_dir):
            self.has_update_labels = True
            self.update_label_filenames = sorted([f for f in os.listdir(update_label_dir) if f.endswith(('.tif', '.tiff', '.img'))])
            
            if self.label_filenames is not None:
                self.filter_update_labels()
        else:
            self.has_update_labels = False
            self.update_label_filenames = None

    def __len__(self):
        return len(self.image_filenames)
    
    def filter_files(self):
        image_basenames = [os.path.splitext(f)[0] for f in self.image_filenames]
        label_basenames = [os.path.splitext(f)[0] for f in self.label_filenames]
        common_basenames = set(image_basenames).intersection(set(label_basenames))
        
        image_dict_full = {os.path.splitext(f)[0]: f for f in self.image_filenames}
        label_dict_full = {os.path.splitext(f)[0]: f for f in self.label_filenames}
        
        if len(common_basenames) < min(len(image_basenames) * 0.5, len(label_basenames) * 0.5):
            def extract_id(filename):
                base = os.path.splitext(filename)[0]
                if '_' in base:
                    return base.split('_')[-1]
                return base
            
            image_ids = {extract_id(f): f for f in self.image_filenames}
            label_ids = {extract_id(f): f for f in self.label_filenames}
            common_ids = set(image_ids.keys()).intersection(set(label_ids.keys()))
            
            if len(common_ids) > len(common_basenames):
                print(f"Using underscore number matching, found {len(common_ids)} matching files")
                filtered_image_files = [image_ids[id_] for id_ in common_ids]
                filtered_label_files = [label_ids[id_] for id_ in common_ids]
                
                sorted_pairs = sorted(zip(filtered_image_files, filtered_label_files), 
                                      key=lambda x: extract_id(x[0]))
                filtered_image_files = [pair[0] for pair in sorted_pairs]
                filtered_label_files = [pair[1] for pair in sorted_pairs]
            else:
                print(f"Using complete filename matching, found {len(common_basenames)} matching files")
                filtered_image_files = [image_dict_full[base] for base in common_basenames]
                filtered_label_files = [label_dict_full[base] for base in common_basenames]
        else:
            filtered_image_files = [image_dict_full[base] for base in common_basenames]
            filtered_label_files = [label_dict_full[base] for base in common_basenames]
        
        if len(filtered_image_files) < len(image_basenames):
            print(f"Warning: {len(image_basenames) - len(filtered_image_files)} images excluded due to missing matching labels")
        
        if len(filtered_label_files) < len(label_basenames):
            print(f"Warning: {len(label_basenames) - len(filtered_label_files)} labels excluded due to missing matching images")
        
        self.image_filenames = filtered_image_files
        self.label_filenames = filtered_label_files
        
        print(f"After filtering, {len(filtered_image_files)} valid image-label pairs remain")
    
    def filter_update_labels(self):
        filtered_update_label_files = []
        
        for label_filename in self.label_filenames:
            label_basename = os.path.splitext(label_filename)[0]
            update_label_filename = None
            
            for update_file in self.update_label_filenames:
                update_basename = os.path.splitext(update_file)[0]
                if update_basename == label_basename:
                    update_label_filename = update_file
                    break
            
            if update_label_filename is None:
                def extract_id(filename):
                    base = os.path.splitext(filename)[0]
                    if '_' in base:
                        return base.split('_')[-1]
                    return base
                
                label_id = extract_id(label_filename)
                for update_file in self.update_label_filenames:
                    if extract_id(update_file) == label_id:
                        update_label_filename = update_file
                        break
            
            if update_label_filename:
                filtered_update_label_files.append(update_label_filename)
            else:
                filtered_update_label_files.append(None)
                print(f"Warning: No updated label found corresponding to label {label_filename}")
        
        self.update_label_filenames = filtered_update_label_files
        valid_count = sum(1 for f in filtered_update_label_files if f is not None)
        print(f"Matched {valid_count}/{len(self.label_filenames)} updated labels")
              
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        filename = self.label_filenames[idx]
        
        with rasterio.open(image_path) as image_ds:
            image = image_ds.read()
            dtype = image_ds.dtypes[0]
        
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        with rasterio.open(label_path) as label_ds:
            label = label_ds.read(1).astype(np.int8)
        
        update_label = None
        if self.has_update_labels and self.update_label_filenames[idx] is not None:
            update_label_path = os.path.join(self.update_label_dir, self.update_label_filenames[idx])
            try:
                with rasterio.open(update_label_path) as update_ds:
                    update_label = update_ds.read(1).astype(np.int8)
            except Exception as e:
                print(f"Error reading updated label: {update_label_path}, error: {e}")
                update_label = None
        
        max_value = self.get_max_value(dtype)
        
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        
        if self.mode == 'train':
            if update_label is not None:
                transform_pipeline = self.get_train_transforms_with_update(max_value)
                augmented = transform_pipeline(image=image, mask=label, mask1=update_label)
            else:
                transform_pipeline = self.get_train_transforms(max_value)
                augmented = transform_pipeline(image=image, mask=label)
        else:
            if update_label is not None:
                transform_pipeline = self.get_val_transforms_with_update(max_value)
                augmented = transform_pipeline(image=image, mask=label, mask1=update_label)
            else:
                transform_pipeline = self.get_val_transforms(max_value)
                augmented = transform_pipeline(image=image, mask=label)
        
        image_tensor = augmented['image']
        
        if isinstance(augmented['mask'], torch.Tensor):
            label_tensor = augmented['mask'].long()
        else:
            label_tensor = torch.from_numpy(augmented['mask']).long()
        
        if label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)
        
        result = {
            'image': image_tensor,
            'label': label_tensor,
            'filename': filename,
        }
        
        if update_label is not None and 'mask1' in augmented:
            if isinstance(augmented['mask1'], torch.Tensor):
                update_label_tensor = augmented['mask1'].long()
            else:
                update_label_tensor = torch.from_numpy(augmented['mask1']).long()
            
            if update_label_tensor.dim() == 3 and update_label_tensor.shape[0] == 1:
                update_label_tensor = update_label_tensor.squeeze(0)
            
            result['update_label'] = update_label_tensor
        
        return result
    
    def get_train_transforms(self, max_value):
        return A.Compose([
            A.VerticalFlip(p=0.5), 
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.5
            ),
            A.RandomResizedCrop(
                height=512, width=512,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            A.CLAHE(
                clip_limit=(1, 4),
                tile_grid_size=(8, 8),
                p=0.5
            ),
            A.GaussianBlur(
                blur_limit=(0),
                sigma_limit=(0.5, 3.0),
                p=0.5
            ),
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=self.mean_list,
                std=self.std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2()
        ])
    
    
    def get_train_transforms_with_update(self, max_value):
        return A.Compose([
            A.VerticalFlip(p=0.5), 
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.5
            ),
            A.RandomResizedCrop(
                height=512, width=512,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            A.CLAHE(
                clip_limit=(1, 4),
                tile_grid_size=(8, 8),
                p=0.5
            ),
            A.GaussianBlur(
                blur_limit=(0),
                sigma_limit=(0.5, 3.0),
                p=0.5
            ),
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=self.mean_list,
                std=self.std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2()
        ], additional_targets={'mask1': 'mask'})
    
    def get_val_transforms(self, max_value):
        return A.Compose([
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=self.mean_list,
                std=self.std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2()
        ])
    
    def get_val_transforms_with_update(self, max_value):
        return A.Compose([
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=self.mean_list,
                std=self.std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2()
        ], additional_targets={'mask1': 'mask'})
    
    def get_max_value(self, dtype_str):
        if 'uint8' in dtype_str:
            return np.iinfo(np.uint8).max
        elif 'uint16' in dtype_str:
            return np.iinfo(np.uint16).max
        elif 'int16' in dtype_str:
            return np.iinfo(np.int16).max 
        elif 'uint32' in dtype_str:
            return np.iinfo(np.uint32).max
        elif 'int32' in dtype_str:
            return np.iinfo(np.int32).max
        elif 'float' in dtype_str:
            return 1.0
        else:
            print(f"Warning: Unknown data type {dtype_str}, assuming data is already normalized")
            return 1.0
    
class JSDAugMixDataset(Dataset):
    def __init__(self, image_dir, label_dir, num_splits=3, mode='train', 
                 mean_list=None, std_list=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.num_splits = num_splits
        
        self.mean_list = mean_list if mean_list is not None else [0.485, 0.456, 0.406]
        self.std_list = std_list if std_list is not None else [0.229, 0.224, 0.225]
        
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.img'))])
        
        if label_dir:
            self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith(('.tif', '.tiff', '.img'))])
            
            if len(self.image_filenames) != len(self.label_filenames):
                print(f"Warning: Number of images ({len(self.image_filenames)}) doesn't match number of labels ({len(self.label_filenames)})")
                
            self.filter_files()
        else:
            self.label_filenames = None

    def filter_files(self):
        image_basenames = [os.path.splitext(f)[0] for f in self.image_filenames]
        label_basenames = [os.path.splitext(f)[0] for f in self.label_filenames]
        common_basenames = set(image_basenames).intersection(set(label_basenames))
        
        image_dict_full = {os.path.splitext(f)[0]: f for f in self.image_filenames}
        label_dict_full = {os.path.splitext(f)[0]: f for f in self.label_filenames}
        
        if len(common_basenames) < min(len(image_basenames) * 0.5, len(label_basenames) * 0.5):
            def extract_id(filename):
                base = os.path.splitext(filename)[0]
                if '_' in base:
                    return base.split('_')[-1]
                return base
            
            image_ids = {extract_id(f): f for f in self.image_filenames}
            label_ids = {extract_id(f): f for f in self.label_filenames}
            common_ids = set(image_ids.keys()).intersection(set(label_ids.keys()))
            
            if len(common_ids) > len(common_basenames):
                print(f"Using ID matching after underscore, found {len(common_ids)} matching pairs")
                filtered_image_files = [image_ids[id_] for id_ in common_ids]
                filtered_label_files = [label_ids[id_] for id_ in common_ids]
                
                sorted_pairs = sorted(zip(filtered_image_files, filtered_label_files), 
                                      key=lambda x: extract_id(x[0]))
                filtered_image_files = [pair[0] for pair in sorted_pairs]
                filtered_label_files = [pair[1] for pair in sorted_pairs]
            else:
                print(f"Using complete filename matching, found {len(common_basenames)} matching pairs")
                filtered_image_files = [image_dict_full[base] for base in common_basenames]
                filtered_label_files = [label_dict_full[base] for base in common_basenames]
        else:
            filtered_image_files = [image_dict_full[base] for base in common_basenames]
            filtered_label_files = [label_dict_full[base] for base in common_basenames]
        
        if len(filtered_image_files) < len(image_basenames):
            print(f"Warning: {len(image_basenames) - len(filtered_image_files)} images excluded due to missing matching labels")
        
        if len(filtered_label_files) < len(label_basenames):
            print(f"Warning: {len(label_basenames) - len(filtered_label_files)} labels excluded due to missing matching images")
        
        self.image_filenames = filtered_image_files
        self.label_filenames = filtered_label_files
        
        print(f"After filtering, {len(filtered_image_files)} valid image-label pairs remain")
    
    def __len__(self):
        return len(self.image_filenames)
    
    def get_base_transforms(self, max_value):
        return A.Compose([
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=self.mean_list,
                std=self.std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2()
        ])
    
    def get_aug_transforms(self, max_value):
        return A.Compose([
            A.VerticalFlip(p=0.5), 
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2,
                p=0.5
            ),
            
            A.RandomResizedCrop(
                height=512, width=512,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.25),
                p=0.5
            ),

            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.5, 3.0),
                p=0.5
            ),
            A.ToFloat(max_value=max_value, p=1.0),
            A.Normalize(
                mean=self.mean_list,
                std=self.std_list,
                max_pixel_value=1.0,
                p=1.0
            ),
            ToTensorV2()
        ])
    
    def get_max_value(self, dtype_str):
        if 'uint8' in dtype_str:
            return np.iinfo(np.uint8).max
        elif 'uint16' in dtype_str:
            return np.iinfo(np.uint16).max
        elif 'int16' in dtype_str:
            return np.iinfo(np.int16).max 
        elif 'uint32' in dtype_str:
            return np.iinfo(np.uint32).max
        elif 'int32' in dtype_str:
            return np.iinfo(np.int32).max
        elif 'float' in dtype_str:
            return 1.0
        else:
            print(f"Warning: Unknown data type {dtype_str}, assuming normalized data")
            return 1.0
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        with rasterio.open(image_path) as image_ds:
            image = image_ds.read()
            dtype = image_ds.dtypes[0]
        
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        with rasterio.open(label_path) as label_ds:
            label = label_ds.read(1).astype(np.int8)
        
        max_value = self.get_max_value(dtype)
        
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        
        base_transform = self.get_base_transforms(max_value)
        
        aug_transform = self.get_aug_transforms(max_value)
        
        base_augmented = base_transform(image=image)
        base_image = base_augmented['image']
        
        image_list = [base_image]
        
        for _ in range(self.num_splits - 1):
            aug_augmented = aug_transform(image=image)
            aug_image = aug_augmented['image']
            image_list.append(aug_image)
        
        if isinstance(label, np.ndarray):
            label_tensor = torch.from_numpy(label).long()
        else:
            label_tensor = label.long()
        
        return tuple(image_list), label_tensor