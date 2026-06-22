import torch

ckpt = torch.load(r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\weights\model_only_age_imdb_4.29.pth.tar", map_location="cpu")

# Si es un state_dict
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
print(ckpt.keys())

for name, tensor in state_dict.items():
    print(name, tensor.dtype)
    break