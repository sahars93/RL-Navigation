import onnxruntime as ort
import numpy as np
import onnx
import netron
import torch


# onnx_model = onnx.load("test_onnx_jetbot.onnx")
# print(onnx_model)
# observation = np.random.rand(1, 4).astype(np.float32)
# ort_model = ort.InferenceSession("test_onnx_cartpole.onnx")
# ort_model = ort.InferenceSession("saved_checkpoints/last_Jetbot_ep_200_rew__30.939707_.onnx")

# outputs = ort_model.run(None, {"obs": observation})
# print(outputs)
# mu = outputs[0]
# sigma = np.exp(outputs[1])
# action = np.random.normal(mu, sigma)
# print(action)
# base_action = action[:2]
# arm_action = action[2:]

# print(action[2:])
# print(base_action)
# input_details = onnx_model.graph.input
# output_details = onnx_model.graph.output
# for input_info in output_details:
#     print(f"Name: {input_info.name}")

netron.start("saved_checkpoints/last_Jetbot_ep_200_rew_27.83471.pth")



# 

# # Create a tensor with dimensions [1, 2, 1, 3, 1]
# x = torch.randn(1, 12)
# # Apply squeeze to remove dimensions with size 1
# # y = x.squeeze()
# print(x)
# print("Original tensor shape:", x.shape)
# # print("Squeezed tensor shape:", y.shape)
