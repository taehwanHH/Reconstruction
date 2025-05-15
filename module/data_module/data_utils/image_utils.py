import os
import os.path as osp
import glob
import pandas as pd
import numpy as np
from PIL import Image


import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from torch_geometric.data import  Dataset
import torchvision.transforms as transforms

from midastouch.modules.misc import  save_heightmaps, save_contactmasks, remove_and_mkdir

from module.SensingPart import Sensing
from module.TactileUtil import Stiffness
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_image_data(cfg, output_base, mkdir=True, idx_offset=0) -> None:
    DIGIT = Sensing(cfg, output_base, mkdir=mkdir)
    poses = DIGIT.get_points_poses()
    DIGIT.sensing(poses)
    DIGIT.save_results(poses,idx_offset)
    # DIGIT.show_heatmap(poses,visible=False)

def load_all_heightmaps(heightmap_dir, img_size=(32,32), img_extension="jpg") -> torch.Tensor:
    # Get a sorted list of image file paths
    file_list = sorted([
        os.path.join(heightmap_dir, f)
        for f in os.listdir(heightmap_dir)
        if f.lower().endswith(img_extension)
    ])

    # Load each image, resize if needed, and convert to a numpy array
    images = []
    height, width = img_size
    for file_path in file_list:
        img = Image.open(file_path).convert('L')  # convert to grayscale if necessary
        img = img.resize((width, height),resample=Image.Resampling.BICUBIC)  # ensure the size is (width, height)
        img_np = np.array(img)
        images.append(img_np)

    # Stack all images into a single numpy array of shape (N, height, width)
    all_images_np = np.stack(images, axis=0)

    # Convert the numpy array to a torch tensor
    heightmaps_tensor = torch.tensor(all_images_np, dtype=torch.float).unsqueeze(1)

    return heightmaps_tensor

def get_image(base_dir, img_size=(32,32))->(torch.Tensor, torch.Tensor):
    base = base_dir
    heightmap_dir = osp.join(base, "gt_heightmaps")
    mask_dir = osp.join(base, "gt_contactmasks")
    pos_file = osp.join(base, "sensor_poses.npy")

    hm = load_all_heightmaps(heightmap_dir, img_size)  # Modify load_all_heightmaps to output resized images if needed.
    cm = load_all_heightmaps(mask_dir,  img_size)
    cm = ((cm > 128).float())
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)

    masked_hm = hm * cm

    pos_np = np.load(pos_file)
    pos = torch.tensor(pos_np, dtype=torch.float)

    return masked_hm, pos

def save_image_datasets(objects, output_base, cfg) -> None:
    k_class = Stiffness(cfg)
    k_values = k_class.k_values.tolist()

    cfg.sensing.num_samples = 2000
    i = 0
    for obj in objects:
        print(f"\033[1;33m * Sensing object: {obj}\033[0m")
        for k in k_values:
            print(f"\033[1;38m * Sensing stiffness: {k}\033[0m")
            output_dir = osp.join(output_base,f"{i}")
            cfg.obj_model = obj
            cfg.render.k.fixed = k
            collect_image_data(cfg, output_dir)

            # image to dataset
            img, pos = get_image( output_dir)

            dataset_each_obj = TensorDataset(img, pos)
            torch.save(dataset_each_obj, osp.join(output_dir, f"img_dataset_{i}.pt"))
            masked_hm_path = osp.join(output_dir, "origin_masked_hm")
            os.makedirs(masked_hm_path)
            save_heightmaps(img.squeeze(1).numpy(), masked_hm_path, 0)

            i += 1


    pattern = os.path.join(output_base, "*", "img_dataset_*.pt")
    dataset_files = sorted(glob.glob(pattern))

    combined_dataset = ConcatDataset([torch.load(file) for file in dataset_files])
    torch.save(combined_dataset, osp.join(output_base, f"{cfg.data_config.image.file_name}.pt"))



#######################################################################################
# For Dataloader (feature extractor training)
#######################################################################################
class NormalizedImageDatasets(Dataset):
    def __init__(self, images ):
        """
        images: [N, C, H, W] Tensor
        transform1, transform2: 서로 다른 데이터 증강 함수
        """
        super().__init__()
        self.images = images
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        img = self.images[idx]
        view1 = self.transform(img)
        return view1

def get_normalized_dataset(cfg):
    combined_train_dataset = torch.load(osp.join(cfg.train.dir,"image",f"{cfg.image.file_name}.pt"))
    images_train = torch.cat([ds.tensors[0] for ds in combined_train_dataset.datasets], dim=0)
    combined_test_dataset = torch.load(osp.join(cfg.test.dir,"image",f"{cfg.image.file_name}.pt"))
    images_test = torch.cat([ds.tensors[0] for ds in combined_test_dataset.datasets], dim=0)

    aug_train_dataset = NormalizedImageDatasets(images_train)
    aug_test_dataset = NormalizedImageDatasets(images_test)

    train_loader = DataLoader(aug_train_dataset, batch_size=cfg.image.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(aug_test_dataset, batch_size=cfg.image.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

#######################################################################################
# For image autoencoder
#######################################################################################

def image_reconstruction(data_loader, model, result_dir):
    model.eval()

    s_hat =[]
    ch_out=[]
    if model.channel is not None:
        model.channel.channel_param_print()

    with torch.no_grad():
        for batch in data_loader:
            s_batch = batch.to(device)
            recon, co = model.forward_with_channel(s_batch)
            s_hat.extend(recon)
            ch_out.extend(co)

    s_hat_tensor = torch.cat(s_hat)
    ch_out_tensor = torch.cat(ch_out)

    denorm_s = (s_hat_tensor * 0.5 + 0.5).clamp(min=0, max=1)
    images_tensor = denorm_s.mul(255).byte()
    masks_tensor= (images_tensor> 0).to(torch.uint8)


    recon_images = [images_tensor[i].squeeze().cpu().numpy().astype(np.float32) for i in range(images_tensor.size(0))]
    masks = [masks_tensor[i].squeeze().cpu().numpy().astype(np.float32) for i in range(masks_tensor.size(0))]


    image_save_dir = osp.join(result_dir,"recon_image")
    mask_save_dir = osp.join(result_dir,"recon_mask")

    if os.path.exists(image_save_dir):
        remove_and_mkdir(image_save_dir)
    else:
        os.makedirs(image_save_dir)

    if os.path.exists(mask_save_dir):
        remove_and_mkdir(mask_save_dir)
    else:
        os.makedirs(mask_save_dir)


    save_heightmaps(recon_images, image_save_dir, idx_offset=0)
    save_contactmasks(masks, mask_save_dir, idx_offset=0)

    return ch_out_tensor

#######################################################################################
# For Stiffness labeled data
#######################################################################################
def save_stiffness_labeled_datasets(objects, output_base, cfg):
    k_class = Stiffness(cfg)
    k_values = k_class.k_values

    output_csv = osp.join(output_base, "stiffness_labeled_data.csv")
    cfg.sensing.num_samples = 1000
    for k in k_values:
        print(f"\033[1;38m * Sensing stiffness: {k}\033[0m")
        for i, obj in enumerate(objects):
            print(f"\033[1;33m * Sensing object: {obj}\033[0m")
            cfg.obj_model = obj
            output_dir = osp.join(output_base,f"k_{k}",f"{i}")
            DIGIT = Sensing(cfg, output_dir)
            DIGIT.renderer.set_obj_stiffness(k)

            poses = DIGIT.get_points_poses()
            DIGIT.sensing(poses)

            DIGIT.save_results(poses, idx_offset=0)

            # create (32x32 masked heightmaps)
            img, _ = get_image(output_dir)

            masked_hm_path = osp.join(output_dir, "origin_masked_hm")
            os.makedirs(masked_hm_path, exist_ok=True)
            save_heightmaps(img.squeeze(1).numpy(), masked_hm_path, 0)


    records = []
    for s in k_values:
        img_pattern = os.path.join(output_base, f"k_{s}","*", "origin_masked_hm", "*.jpg")
        for file_path in sorted(glob.glob(img_pattern, recursive=True)):
            records.append((file_path, s))
    df = pd.DataFrame(records, columns=["filename", "stiffness"])
    df.to_csv(output_csv, index=False)


class StiffnessImageDataset(Dataset):
    def __init__(self, label_csv, transform=None):
        """
        label_csv: path to CSV with columns ["filename","stiffness"]
        transform: torchvision transforms to apply to the loaded PIL image
        """
        super().__init__()
        self.df = pd.read_csv(label_csv)
        # if no transform passed, use default: [0-255]→[0-1], then Normalize to [-1,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),                      # → [0,1]
            transforms.Normalize((0.5,), (0.5,)),       # → [-1,1]
        ])
        # 후보 stiffness 값을 정렬한 리스트
        self.candidates = sorted(self.df['stiffness'].unique())
        # 각 stiffness 값을 클래스 인덱스로 매핑 (예: {500:0, 1000:1, ...})
        self.mapping = {val: idx for idx, val in enumerate(self.candidates)}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1) lookup
        row = self.df.iloc[idx]
        img_path = row['filename']
        stiffness = float(row['stiffness'])

        # 2) load image
        img = Image.open(img_path).convert('L')  # grayscale
        img_t = self.transform(img)             # Tensor [1,H,W]
        label = torch.tensor(self.mapping[stiffness], dtype=torch.long)

        # 3) return (image, label)
        return img_t, label
    
def get_stiffness_image_dataset(cfg):
    train_label_csv = osp.join(cfg.train.dir, "k_labeled_image","stiffness_labeled_data.csv")
    train_dataset = StiffnessImageDataset(train_label_csv)
    train_loader = DataLoader(train_dataset, batch_size=cfg.image.batch_size, shuffle=True, num_workers=4)

    test_label_csv = osp.join(cfg.test.dir, "k_labeled_image", "stiffness_labeled_data.csv")
    test_dataset = StiffnessImageDataset(test_label_csv)
    test_loader = DataLoader(test_dataset, batch_size=cfg.image.batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader