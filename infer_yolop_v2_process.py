# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import numpy as np
import cv2
import os
import copy
import wget
from ikomia import core, dataprocess
from ikomia.utils import strtobool
from infer_yolop_v2.utils.utils import \
    non_max_suppression, split_for_trace_model,\
    driving_area_mask, lane_line_mask, letterbox, check_img_size


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYolopV2Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "weights", "yolopv2.pt")
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.2
        self.iou_thres = 0.45
        self.update = False
        self.object = True
        self.road_lane = True

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(params["cuda"])
        self.input_size = int(params["input_size"])
        self.conf_thres = float(params["conf_thres"])
        self.iou_thres = float(params["iou_thres"])
        self.update = strtobool(params["update"])
        self.object = strtobool(params["object"])
        self.road_lane = strtobool(params["road_lane"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
                "cuda":str(self.cuda),
                "input_size": str(self.input_size),
                "conf_thres": str(self.conf_thres),
                "iou_thres": str(self.iou_thres),
                "update": str(self.update),
                "object": str(self.object),
                "road_lane": str(self.road_lane)
            }
        return params


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYolopV2(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        # Add input/output of the process here
        dataprocess.CObjectDetectionTask.__init__(self, name)
        self.add_output(dataprocess.CSemanticSegmentationIO())
        self.update = False
        self.device = torch.device("cpu")
        self.model = None
        self.stride = 32
        self.imgsz = 640
        self.img_resize = (1280,720)
        self.colors = [[0,0,255], [255,0,0]]
        self.box_color = [204, 204, 0]
        self.classes = ['road', 'lane']
        self.names = "vehicle"
        # Create parameters class
        if param is None:
            self.set_param_object(InferYolopV2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, src_image):
        param = self.get_param_object()

        # Resize image to 640 and pad if necessary
        h_scr, w_src = src_image.shape[:2]
        scale_ini = [w_src / self.img_resize[0], h_scr / self.img_resize[1]]
        img0 = cv2.resize(src_image, self.img_resize, interpolation=cv2.INTER_LINEAR)
        img, scale, pad = letterbox(img0, self.imgsz, self.stride)
        pad_w, pad_h = pad
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Run inference
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # convert 0 - 255 to 0.0 - 1.0

        # Returns a new tensor with a dimension of size one inserted at the specified position.
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        [pred, anchor_grid], seg, ll = self.model(img)
        # Waste time: the incompatibility of torch.jit.trace
        # causes extra time consumption in demo version
        # but this problem will not appear in offical version
        # Reshape tensor
        pred = split_for_trace_model(pred, anchor_grid)

        # Apply NMS (Non Maximum Suppression)
        pred = non_max_suppression(pred, param.conf_thres, param.iou_thres)
        self.set_names([self.names])
        # Object detection
        if param.object:
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # draw object bounding box around each detection
                    for xyxy in reversed(det):
                        x1 = (xyxy[0] / scale[0] - pad_w) * scale_ini[0]
                        y1 = (xyxy[1] / scale[1] - (pad_h * 2)) * scale_ini[1]
                        x2 = (xyxy[2] / scale[0] - pad_w) * scale_ini[0]
                        y2 = (xyxy[3] / scale[1] - (pad_h * 2)) * scale_ini[1]
                        w = x2 - x1
                        h = y2 - y1
                        self.add_object(i, 0, float(xyxy[4]), float(x1), float(y1), float(w), float(h))
          
        # Segmentation
        semantic_output = self.get_output(2)
        if param.road_lane:
            h_img, w_img = np.shape(src_image)[:2]
            da_seg_mask = driving_area_mask(seg)
            da_seg_mask = da_seg_mask.astype(dtype='uint8')

            ll_seg_mask = lane_line_mask(ll)
            ll_seg_mask = ll_seg_mask.astype(dtype='uint8')

            merge_mask = np.where(ll_seg_mask == 1, 2, da_seg_mask)
            merge_mask = cv2.resize(merge_mask, (w_img, h_img), interpolation = cv2.INTER_NEAREST)
            semantic_output.set_class_names(self.classes)
            semantic_output.set_class_colors(self.colors)
            semantic_output.set_mask(merge_mask)
            self.set_output_color_map(0, 2, self.colors, True)

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        image_in = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = image_in.get_image()

        param = self.get_param_object()

        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            # Load model
            weights_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
            weights = param.model_path

            if not os.path.isdir(weights_folder):
                os.mkdir(weights_folder)

            if not os.path.isfile(weights):
                print("The model YOLOPv2 is downloading...")
                url = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
                wget.download(url, out=param.model_path)
                print("The model is downloaded")

            self.model = torch.jit.load(weights, map_location=self.device)
            self.imgsz = check_img_size(int(param.input_size), s=self.stride)  # check img_size

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).
                           to(self.device).type_as(next(self.model.parameters())))
            half = False
            if half:
                self.model.half()  # to FP16

            # Set dropout and batch normalization layers to evaluation mode before running inference
            self.model.eval()
            param.update = False
            print("Will run on {}".format(self.device.type))

        with torch.no_grad():
            self.infer(src_image)

          # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYolopV2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolop_v2"
        self.info.short_description = "Panoptic driving Perception using YoloPv2"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.2.2"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Cheng Han, Qichao Zhao, Shuyi Zhang, Yinzi Chen,"\
                            "Zhenlin Zhang, Jinwei Yuan"
        self.info.article = "YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception."
        self.info.journal = "arXiv preprint arXiv:2208.1143"
        self.info.year = 2022
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2208.11434"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolop_v2"
        self.info.original_repository = "https://github.com/CAIC-AD/YOLOPv2"
        # Keywords used for search
        self.info.keywords = "YOLOPv2,infer,panoptic,driving,traffic,object detection,segmentation"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION,INSTANCE_SEGMENTATION,SEMANTIC_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferYolopV2(self.info.name, param)
