<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_yolop_v2/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_yolop_v2</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_yolop_v2">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_yolop_v2">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_yolop_v2/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_yolop_v2.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run YOLOP_v2 for Panoptic driving Perception. This model detects traffic object detection, drivable area segmentation and lane line detection.

![Road object detection](https://raw.githubusercontent.com/Ikomia-hub/infer_yolop_v2/feat/new_readme/icons/output1.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolop_v2", auto_connect=True)

# Run on your image  
wf.run_on(url="https://www.cnet.com/a/img/resize/4797a22dd672697529df19c2658364a85e0f9eb4/hub/2023/02/14/9406d927-a754-4fa9-8251-8b1ccd010d5a/ring-car-cam-2023-02-14-14h09m20s720.png?auto=webp&width=1920")

# Inpect your result
display(algo.get_image_with_graphics())
display(algo.get_output(0).get_overlay_mask())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **input_size** (int) - default '640': Size of the input image.
- **conf_thres** (float) default '0.2': Box threshold for the prediction [0,1].
- **iou_thres** (float) - default '0.45': Intersection over Union, degree of overlap between two boxes [0,1].
- **object** (bool) - default 'True': Detect vehicles.
- **road_lane** (bool) - default 'True': Detect road and line.

**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolop_v2", auto_connect=True)


algo.set_parameters({
    "input_size": "640",
    "conf_thres": "0.2",
    "iou_thres": "0.45",
    "object": "True",
    "road_lane": "True"
})

# Run on your image  
wf.run_on(url="https://www.cnet.com/a/img/resize/4797a22dd672697529df19c2658364a85e0f9eb4/hub/2023/02/14/9406d927-a754-4fa9-8251-8b1ccd010d5a/ring-car-cam-2023-02-14-14h09m20s720.png?auto=webp&width=1920")

# Inpect your result
display(algo.get_image_with_graphics())
display(algo.get_output(0).get_overlay_mask())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolop_v2", auto_connect=True)

# Run on your image  
wf.run_on(url="https://www.cnet.com/a/img/resize/4797a22dd672697529df19c2658364a85e0f9eb4/hub/2023/02/14/9406d927-a754-4fa9-8251-8b1ccd010d5a/ring-car-cam-2023-02-14-14h09m20s720.png?auto=webp&width=1920")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
