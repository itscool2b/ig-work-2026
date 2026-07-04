#imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from integrated_gradients import integrated_gradients

#model
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).cuda()
model.train(False)
for p in model.parameters():
    p.requires_grad_(False)

#preprocess
img = Image.open('image.jpg')
preprocess = weights.transforms()
input_tensor = preprocess(img).unsqueeze(0).cuda()

#predicted class
with torch.no_grad():
    logits = model(input_tensor).squeeze(0)
    class_id = logits.argmax().item()
    class_name = weights.meta["categories"][class_id]
    prob = torch.softmax(logits, dim=0)[class_id].item() * 100
    print(f"predicted: {class_name} (logit={logits[class_id].item():.2f}, {prob:.1f}%)")

#forward_fn — log_softmax of predicted class
def forward_fn(x):
    logits = model(x).squeeze(0)
    return torch.log_softmax(logits, dim=-1)[class_id]

#ig — black image baseline (zeros in pixel space, preprocessed), m=300
black_img = Image.new('RGB', (224, 224), (0, 0, 0))
baseline = preprocess(black_img).unsqueeze(0).cuda()
attr = integrated_gradients(forward_fn, input_tensor, baseline, m=300)

#visualize
attr_map = attr.squeeze(0).sum(dim=0).abs().detach().cpu().numpy()
attr_map = attr_map / attr_map.max()

original = np.array(img.resize((224, 224)))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"ResNet-50  |  predicted: {class_name}  ({prob:.1f}%)", fontsize=13, fontweight="bold", y=1.02)

ax1.imshow(original)
ax1.set_title("original", fontsize=11)
ax1.axis("off")

im2 = ax2.imshow(attr_map, cmap="hot")
ax2.set_title("IG attribution (|sum over channels|)", fontsize=11)
ax2.axis("off")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="normalized attribution")

ax3.imshow(original)
ax3.imshow(attr_map, cmap="hot", alpha=0.5)
ax3.set_title(f"overlay ({class_name})", fontsize=11)
ax3.axis("off")

fig.text(0.5, -0.02,
         "IG params: m=300 steps  |  baseline: black image (preprocessed zeros)  |  target: log_softmax  |  aggregation: abs sum over RGB channels",
         ha="center", fontsize=9, fontstyle="italic", color="0.4")

plt.tight_layout()
plt.savefig("output/ig_resnet50.png", dpi=150, bbox_inches="tight")
print("saved ig_resnet50.png")

#cleanup
del model
torch.cuda.empty_cache()
