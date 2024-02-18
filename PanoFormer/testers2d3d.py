import os

import numpy as np
import json
import cv2

import torch
from torch.nn import functional as F
from torchvision import transforms
from utils import colorize

# from metrics import compute_depth_metrics, Evaluator
from network.model import Panoformer as PanoBiT



class Trainer:
    def __init__(self, settings):
        self.settings = settings

        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        # self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        self.model = PanoBiT()
        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())

        if self.settings.load_weights_dir is not None:
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        # self.evaluator = Evaluator()
        # self.save_settings()


    def process_batch(self, img_path, mask_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, dsize=(1024, 512), interpolation=cv2.INTER_CUBIC)

        img = self.to_tensor(img)
        inputs = self.normalize(img)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)

        # Process mask if exists
        try:
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            h, w = mask.shape[:2]
            mask = cv2.resize(mask, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
            mask = self.to_tensor(mask)
            mask = mask.unsqueeze(0)
            mask = mask.to(self.device)
            mask_flag = True
        except:
            print('Mask image not found')
            mask = torch.ones_like(inputs)
            mask_flag = False

        inputs = inputs * mask

        with torch.no_grad():
            outputs = self.model(inputs)
        output_depth = outputs["pred_depth"]
        output_depth = output_depth * mask[:, 0:1]
        # breakpoint()

        output_depth = F.interpolate(output_depth, size=(h, w), mode='bicubic', align_corners=True).cpu()
        output_depth_colorized = colorize(output_depth)

        return output_depth, output_depth_colorized, mask_flag

    # def validate(self):
    #     """Validate the model on the validation set
    #     """
    #     self.model.eval()

    #     self.evaluator.reset_eval_metrics()

    #     pbar = tqdm.tqdm(self.val_loader)
    #     pbar.set_description("Validating Epoch_{}".format(self.epoch))

    #     with torch.no_grad():
    #         for batch_idx, inputs in enumerate(pbar):
    #             outputs, losses = self.process_batch(inputs)
    #             pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
    #             gt_depth = inputs["gt_depth"] * inputs["val_mask"]
    #             #mask = inputs["val_mask"]
    #             self.evaluator.compute_eval_metrics(gt_depth, pred_depth)

    #     self.evaluator.print()

    #     for i, key in enumerate(self.evaluator.metrics.keys()):
    #         losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
    #     self.log("val", inputs, outputs, losses)
    #     del inputs, outputs, losses

    # def log(self, mode, inputs, outputs, losses):
    #     """Write an event to the tensorboard events file
    #     """
    #     outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
    #     inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
    #     writer = self.writers[mode]
    #     for l, v in losses.items():
    #         writer.add_scalar("{}".format(l), v, self.step)

    #     for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
    #         writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
    #         # writer.add_image("cube_rgb/{}".format(j), inputs["cube_rgb"][j].data, self.step)
    #         writer.add_image("gt_depth/{}".format(j),
    #                          inputs["gt_depth"][j].data/inputs["gt_depth"][j].data.max(), self.step)
    #         writer.add_image("pred_depth/{}".format(j),
    #                          outputs["pred_depth"][j].data/outputs["pred_depth"][j].data.max(), self.step)

    # def save_settings(self):
    #     """Save settings to disk so we know what we ran this experiment with
    #     """
    #     models_dir = os.path.join(self.log_path, "models")
    #     if not os.path.exists(models_dir):
    #         os.makedirs(models_dir)
    #     to_save = self.settings.__dict__.copy()

    #     with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
    #         json.dump(to_save, f, indent=2)


    def load_model(self):
        """Load model from disk"""
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


