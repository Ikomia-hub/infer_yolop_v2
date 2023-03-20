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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_yolop_v2.infer_yolop_v2_process import InferYolopV2Param

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferYolopV2Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)
        if param is None:
            self.parameters = InferYolopV2Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda",
                        self.parameters.cuda and is_available())

        # Input size
        self.check_input_sz = pyqtutils.append_double_spin(
                            self.gridLayout, "Input size",
                            self.parameters.input_size,min = 160., max = 640., step = 1.)

        # Conf_thres
        self.spin_thr_conf = pyqtutils.append_double_spin(
                                self.gridLayout, "Confidence threshold",
                                self.parameters.conf_thres, min = 0., max = 1., step = 1e-1)

        # Iou_thres
        self.spin_iou_conf = pyqtutils.append_double_spin(
                                self.gridLayout, "IoU threshold",
                                self.parameters.iou_thres, min = 0., max = 1., step = 1e-1)

        # Object detection
        self.check_object = pyqtutils.append_check(
                        self.gridLayout, "Vehicule",
                        self.parameters.object)

        # Driving area and lane detection
        self.check_road_lane = pyqtutils.append_check(
                        self.gridLayout, "Road and lane",
                        self.parameters.road_lane)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.check_input_sz.value()
        self.parameters.conf_thres = self.spin_thr_conf.value()
        self.parameters.iou_thres = self.spin_iou_conf.value()
        self.parameters.object = self.check_object.isChecked()
        self.parameters.road_lane = self.check_road_lane.isChecked()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferYolopV2WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolop_v2"

    def create(self, param):
        # Create widget object
        return InferYolopV2Widget(param, None)
