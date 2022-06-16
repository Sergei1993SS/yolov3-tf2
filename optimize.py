from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import logging
import numpy as np
import cv2
import os

input_saved_model_dir = '/home/sergei/PycharmProjects/yolov3-tf2/checkpoints/yolov3_train_97.ckpt'
output_saved_model_dir = '/home/sergei/PycharmProjects/yolov3-tf2/optimize_model'
quantize_mode = 'float16'
input_size = 1024
path_img = '/home/sergei/HS_TRAIN/JPEGImages/'


def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    print(image_resized.shape)

    image_paded = np.full(shape=[ih, iw, 1], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    print(image_paded.shape)

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def representative_data_gen():
  fimage = os.listdir(path_img)
  fimage = path_img + fimage[0]
  print(fimage)
  batched_input = np.zeros((1, input_size, input_size, 1), dtype=np.float32)
  for input_value in range(1):
    if os.path.exists(fimage):
      original_image=cv2.imread(fimage, cv2.IMREAD_GRAYSCALE)
      #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
      print(image_data.shape)
      img_in = image_data[..., np.newaxis].astype(np.float32)
      batched_input[input_value, :] = img_in
      # batched_input = tf.constant(img_in)
      print(input_value)
      # yield (batched_input, )
      # yield tf.random.normal((1, 416, 416, 3)),
    else:
      continue
  batched_input = tf.constant(batched_input)
  #print((batched_input,))
  yield (batched_input,)

def image_preprocess(image, target_size, gt_boxes=None):
  ih, iw = target_size
  h, w = image.shape

  scale = min(iw / w, ih / h)
  nw, nh = int(scale * w), int(scale * h)
  image_resized = cv2.resize(image, (nw, nh))

  #image_paded = np.full(shape=[ih, iw], fill_value=128.0)
  #dw, dh = (iw - nw) // 2, (ih - nh) // 2
  #image_paded[dh:nh + dh, dw:nw + dw] = image_resized
  image_paded = image_resized / 255.

  if gt_boxes is None:
    return image_paded

  else:
    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
    return image_paded, gt_boxes

def run():
    # Conversion Parameters
    if quantize_mode == 'int8':
      conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.INT8,
        max_workspace_size_bytes=4000000000,
        use_calibration=True)
      print('convert')
      converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params)

      converter.convert(calibration_input_fn=representative_data_gen)
    elif quantize_mode == 'float16':
      conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=4000000000)
      converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir, conversion_params=conversion_params)
      converter.convert()
    else:
      conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP32,
        max_workspace_size_bytes=4000000000)
      converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir, conversion_params=conversion_params)
      converter.convert()

      # converter.build(input_fn=representative_data_gen)
    converter.save(output_saved_model_dir=output_saved_model_dir)
    print('Done Converting to TF-TRT')

    saved_model_loaded = tf.saved_model.load(output_saved_model_dir)
    graph_func = saved_model_loaded.signatures[tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    trt_graph = graph_func.graph.as_graph_def()
    for n in trt_graph.node:
      print(n.op)
      if n.op == "TRTEngineOp":
        print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
      else:
        print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))
    logging.info("model saved to: {}".format(output_saved_model_dir))

    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    print("numb. of all_nodes in TensorRT graph:", all_nodes)

if __name__ == '__main__':
    run()