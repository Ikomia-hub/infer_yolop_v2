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
import os
import copy
import wget
from ikomia import core, dataprocess
from ikomia.utils import strtobool
from infer_yolop_v2.utils.utils import \
    scale_coords, non_max_suppression, split_for_trace_model,\
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

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.update = strtobool(param_map["update"])
        self.object = strtobool(param_map["object"])
        self.road_lane = strtobool(param_map["road_lane"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["iou_thres"] = str(self.iou_thres)
        param_map["update"] = str(self.update)
        param_map["object"] = str(self.object)
        param_map["road_lane"] = str(self.road_lane)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYolopV2(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(dataprocess.CObjectDetectionIO())
        self.addOutput(dataprocess.CSemanticSegIO())
        self.update = False
        self.device = None
        self.model = None
        self.stride = 32
        self.imgsz = 640
        self.box_color = [204, 204, 0]
        self.classes = ['background', 'road', 'lane']
        self.colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]
        # Create parameters class
        if param is None:
            self.setParam(InferYolopV2Param())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, src_image):
        param = self.getParam()

        # Resize image to 640 and pad if necessary
        img = letterbox(src_image, self.imgsz, self.stride)[0]

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

        # Object detection
        obj_det_output = self.getOutput(1)
        if param.object:
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], src_image.shape).round()
                    # draw object bounding box around each detection
                    for xyxy in reversed(det):
                        x1, y1 = (int(xyxy[0]), int(xyxy[1]))
                        x2, y2 = (int(xyxy[2]), int(xyxy[3]))
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        obj_det_output.addObject(i, "vehicle", float(xyxy[4]), x1, y1, w, h, self.box_color)

        # Segmentation
        semantic_output = self.getOutput(2)
        if param.road_lane:
            h, w = np.shape(src_image)[:2]
            da_seg_mask = driving_area_mask(h, w, seg)
            da_seg_mask = da_seg_mask.astype(dtype='uint8')

            ll_seg_mask = lane_line_mask(h, w, ll)
            ll_seg_mask = ll_seg_mask.astype(dtype='uint8')

            merge_mask = np.where(ll_seg_mask == 1, 2, da_seg_mask)

            semantic_output.setMask(merge_mask)
            semantic_output.setClassNames(self.classes, self.colors)
            self.setOutputColorMap(0, 2, self.colors)
        else:
            h, w = np.shape(src_image)[:2]
            semantic_output.setMask(np.zeros((h, w), dtype=np.uint8))
            semantic_output.setClassNames(self.classes, self.colors)
        
    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        image_in = self.getInput(0)

        # Get image from input/output (numpy array):
        src_image = image_in.getImage()

        param = self.getParam()

        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
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

            self.model = torch.jit.load(weights)
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

        # Forward input image
        self.forwardInputImage(0, 0)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYolopV2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolop_v2"
        self.info.shortDescription = "Panoptic driving Perception using YoloPv2"
        self.info.description = "This plugin proposes inference for Panoptic driving Perception "\
                                "This model detects traffic object detection,"\
                                "drivable area segmentation and lane line detection."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Cheng Han, Qichao Zhao, Shuyi Zhang, Yinzi Chen,"\
                            "Zhenlin Zhang, Jinwei Yuan"
        self.info.article = "YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception."
        self.info.journal = "arXiv preprint arXiv:2208.1143"
        self.info.year = 2022
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = "https://arxiv.org/abs/2208.11434"
        # Code source repository
        self.info.repository = "https://github.com/CAIC-AD/YOLOPv2"
        # Keywords used for search
        self.info.keywords = "YOLOPv2,infer,panoptic,driving,traffic,object detection,segmentation"

    def create(self, param=None):
        # Create process object
        return InferYolopV2(self.info.name, param)
