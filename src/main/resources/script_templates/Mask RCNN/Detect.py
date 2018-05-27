# @Dataset data
# @CommandService cs
# @ModuleService ms

from sc.fiji.maskrcnn import ObjectsDetector

inputs = {"model": None,
          "modelName": "Microtubule",
          "dataset": data}
module = ms.waitFor(cs.run(ObjectsDetector, True, inputs))

rois = module.getOutput("roisList")
table = module.getOutput("table")
masks = module.getOutput("masksImage")
